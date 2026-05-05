from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from utils import get_st_blacklist_pl, load_daily_data_full
from utils.active_market_value_regime import build_active_market_value_regime_frame


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_bull_pool_random_baseline"


def _parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons:
        raise argparse.ArgumentTypeError("horizons must not be empty")
    if any(h <= 0 for h in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive integers")
    return horizons


def _max_drawdown(nav: np.ndarray) -> float:
    running_max = np.maximum.accumulate(nav)
    dd = (nav - running_max) / running_max
    return float(dd.min())


def _rolling_sleeve_nav(daily_ret: np.ndarray, horizon: int) -> tuple[float, float]:
    n_complete_epochs = len(daily_ret) // horizon
    if n_complete_epochs == 0:
        return 0.0, 0.0

    sleeve_navs = np.empty((horizon, n_complete_epochs), dtype=np.float64)
    for sleeve_idx in range(horizon):
        sleeve_rets = daily_ret[sleeve_idx::horizon][:n_complete_epochs]
        sleeve_navs[sleeve_idx] = np.cumprod(1.0 + sleeve_rets)

    nav = sleeve_navs.mean(axis=0)
    return float(nav[-1] - 1.0), _max_drawdown(nav)


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "mean": float(values.mean()),
    }


def build_dataset(args: argparse.Namespace) -> pl.DataFrame:
    conn = duckdb.connect(str(args.qmt_db), read_only=True)
    try:
        st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
        st_blacklist_df = pl.DataFrame(
            {"code": st_blacklist},
            schema={"code": pl.Utf8},
        ).lazy()

        q_full = (
            load_daily_data_full(conn)
            .filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
        )

        max_horizon = max(args.horizons)
        future_exprs = []
        derived_exprs = []
        temp_cols = []
        for step in range(1, max_horizon + 1):
            high_col = f"_fwd_high_{step}"
            low_col = f"_fwd_low_{step}"
            close_col = f"_fwd_close_{step}"
            temp_cols.extend([high_col, low_col, close_col])
            future_exprs.extend(
                [
                    pl.col("high_adj").shift(-step).over("code").alias(high_col),
                    pl.col("low_adj").shift(-step).over("code").alias(low_col),
                    pl.col("close_adj").shift(-step).over("code").alias(close_col),
                ]
            )

        for horizon in args.horizons:
            high_cols = [f"_fwd_high_{step}" for step in range(1, horizon + 1)]
            low_cols = [f"_fwd_low_{step}" for step in range(1, horizon + 1)]
            derived_exprs.extend(
                [
                    (pl.col(f"_fwd_close_{horizon}") / pl.col("close_adj") - 1).alias(
                        f"fwd_ret_{horizon}d"
                    ),
                    (pl.max_horizontal(*high_cols) / pl.col("close_adj") - 1).alias(
                        f"fwd_mfe_{horizon}d"
                    ),
                    (pl.min_horizontal(*low_cols) / pl.col("close_adj") - 1).alias(
                        f"fwd_mae_{horizon}d"
                    ),
                ]
            )

        keep_cols = [
            "date",
            "code",
            "close_adj",
            "market_cap_100m",
            "amount_ma20",
        ]
        for horizon in args.horizons:
            keep_cols.extend(
                [
                    f"fwd_ret_{horizon}d",
                    f"fwd_mfe_{horizon}d",
                    f"fwd_mae_{horizon}d",
                ]
            )

        df = (
            q_full.with_columns(future_exprs)
            .with_columns(derived_exprs)
            .with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
            .drop(temp_cols)
            .filter(pl.col(f"fwd_ret_{max_horizon}d").is_not_null())
            .select(keep_cols)
            .collect()
        )

        df_regime = build_active_market_value_regime_frame(
            bull_trigger_pct=args.amv_bull_trigger_pct,
            bull_lookback_days=args.amv_bull_lookback_days,
            bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
            effective_lag_days=args.amv_effective_lag_days,
            date_col="date",
        ).select(["date", "is_bull_regime", "amv_mechanical_regime"])

        return (
            df.join(df_regime, on="date", how="left")
            .with_columns(
                [
                    pl.col("is_bull_regime").fill_null(False),
                    pl.col("amv_mechanical_regime").fill_null("unknown"),
                ]
            )
            .filter(
                (pl.col("market_cap_100m") >= args.mv_min)
                & (pl.col("amount_ma20") >= args.amount_ma20_min)
            )
        )
    finally:
        conn.close()


def evaluate_pool(
    df_pool: pl.DataFrame,
    *,
    pool_name: str,
    horizons: list[int],
    n_picks: int,
    n_trials: int,
    seed: int,
    stop_loss_pct: float,
) -> dict:
    daily_counts = (
        df_pool.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= n_picks)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    if not eligible_dates:
        return {
            "pool": pool_name,
            "eligible_days": 0,
            "error": f"no dates with at least {n_picks} candidates",
        }

    df_eligible = (
        df_pool.filter(pl.col("date").is_in(eligible_dates))
        .sort(["date", "code"])
        .with_row_index("row_idx")
    )

    date_index_lookup: dict[object, np.ndarray] = {}
    for date_key, sub in df_eligible.group_by("date", maintain_order=True):
        date_value = date_key[0] if isinstance(date_key, tuple) else date_key
        date_index_lookup[date_value] = sub["row_idx"].to_numpy()

    trade_dates = sorted(date_index_lookup)
    rng = np.random.default_rng(seed)

    horizon_arrays = {
        horizon: {
            "ret": df_eligible[f"fwd_ret_{horizon}d"].to_numpy(),
            "mfe": df_eligible[f"fwd_mfe_{horizon}d"].to_numpy(),
            "mae": df_eligible[f"fwd_mae_{horizon}d"].to_numpy(),
        }
        for horizon in horizons
    }

    trial_stats = {
        horizon: {
            "mean_ret_buyhold": np.empty(n_trials),
            "mean_ret_stoploss": np.empty(n_trials),
            "hit15": np.empty(n_trials),
            "nav_end_buyhold": np.empty(n_trials),
            "nav_end_stoploss": np.empty(n_trials),
            "max_dd_buyhold": np.empty(n_trials),
            "max_dd_stoploss": np.empty(n_trials),
        }
        for horizon in horizons
    }

    for trial_idx in range(n_trials):
        picks_by_day = []
        for date_value in trade_dates:
            pool_idx = date_index_lookup[date_value]
            picks_by_day.append(rng.choice(pool_idx, size=n_picks, replace=False))
        picks_flat = np.concatenate(picks_by_day)

        for horizon in horizons:
            arrays = horizon_arrays[horizon]
            ret = arrays["ret"][picks_flat]
            mfe = arrays["mfe"][picks_flat]
            mae = arrays["mae"][picks_flat]
            ret_stop = np.where(mae <= -stop_loss_pct, -stop_loss_pct, ret)

            ret_by_day = ret.reshape(len(trade_dates), n_picks)
            ret_stop_by_day = ret_stop.reshape(len(trade_dates), n_picks)
            mfe_by_day = mfe.reshape(len(trade_dates), n_picks)

            daily_ret = ret_by_day.mean(axis=1)
            daily_ret_stop = ret_stop_by_day.mean(axis=1)
            nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
            nav_end_stop, max_dd_stop = _rolling_sleeve_nav(daily_ret_stop, horizon)

            stats = trial_stats[horizon]
            stats["mean_ret_buyhold"][trial_idx] = ret.mean()
            stats["mean_ret_stoploss"][trial_idx] = ret_stop.mean()
            stats["hit15"][trial_idx] = (mfe_by_day >= 0.15).mean()
            stats["nav_end_buyhold"][trial_idx] = nav_end
            stats["nav_end_stoploss"][trial_idx] = nav_end_stop
            stats["max_dd_buyhold"][trial_idx] = max_dd
            stats["max_dd_stoploss"][trial_idx] = max_dd_stop

    horizon_results = {}
    for horizon in horizons:
        per_day = (
            df_eligible.group_by("date")
            .agg(
                [
                    pl.col(f"fwd_ret_{horizon}d").mean().alias("daily_mean_ret"),
                    (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("daily_hit15"),
                ]
            )
            .sort("date")
        )
        stats = trial_stats[horizon]
        horizon_results[str(horizon)] = {
            "date_equal_mean_ret": float(per_day["daily_mean_ret"].mean()),
            "date_equal_hit15": float(per_day["daily_hit15"].mean()),
            "row_equal_mean_ret": float(df_eligible[f"fwd_ret_{horizon}d"].mean()),
            "row_equal_hit15": float((df_eligible[f"fwd_mfe_{horizon}d"] >= 0.15).mean()),
            "mc_mean_ret_buyhold": _quantile_summary(stats["mean_ret_buyhold"]),
            "mc_mean_ret_stoploss": _quantile_summary(stats["mean_ret_stoploss"]),
            "mc_hit15": _quantile_summary(stats["hit15"]),
            "mc_nav_end_buyhold": _quantile_summary(stats["nav_end_buyhold"]),
            "mc_nav_end_stoploss": _quantile_summary(stats["nav_end_stoploss"]),
            "mc_max_dd_buyhold": _quantile_summary(stats["max_dd_buyhold"]),
            "mc_max_dd_stoploss": _quantile_summary(stats["max_dd_stoploss"]),
        }

    return {
        "pool": pool_name,
        "rows": df_pool.height,
        "eligible_rows": df_eligible.height,
        "eligible_days": len(trade_dates),
        "date_min": str(trade_dates[0]),
        "date_max": str(trade_dates[-1]),
        "candidate_count": {
            "median": float(daily_counts["n_candidates"].median()),
            "mean": float(daily_counts["n_candidates"].mean()),
            "q05": float(daily_counts["n_candidates"].quantile(0.05)),
            "q95": float(daily_counts["n_candidates"].quantile(0.95)),
            "min": int(daily_counts["n_candidates"].min()),
            "max": int(daily_counts["n_candidates"].max()),
        },
        "horizons": horizon_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool random-pick baseline")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_horizons, default=[5, 10, 20])
    parser.add_argument("--n-picks", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-loss-pct", type=float, default=0.03)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.n_picks <= 0:
        raise ValueError("--n-picks must be positive")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")

    started_at = datetime.now()
    print("Building AMV bull pool random baseline dataset...")
    df_lf2 = build_dataset(args)
    print(f"LF2 rows: {df_lf2.height:,}")
    print(f"Date range: {df_lf2['date'].min()} -> {df_lf2['date'].max()}")
    print(f"Unique codes: {df_lf2['code'].n_unique():,}")
    print(f"AMV bull rows: {int(df_lf2['is_bull_regime'].sum()):,}")

    pools = {
        "lf2_all_days": df_lf2,
        "lf2_amv_bull": df_lf2.filter(pl.col("is_bull_regime")),
        "lf2_amv_non_bull": df_lf2.filter(~pl.col("is_bull_regime")),
    }

    results = {}
    for pool_name, pool_df in pools.items():
        print(f"\nEvaluating pool: {pool_name} ({pool_df.height:,} rows)")
        results[pool_name] = evaluate_pool(
            pool_df,
            pool_name=pool_name,
            horizons=args.horizons,
            n_picks=args.n_picks,
            n_trials=args.n_trials,
            seed=args.seed,
            stop_loss_pct=args.stop_loss_pct,
        )

    output_dir = args.output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "horizons": args.horizons,
            "n_picks": args.n_picks,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "stop_loss_pct": args.stop_loss_pct,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    print("\nQuick view (median MC buy-hold mean return):")
    for pool_name, pool_result in results.items():
        print(f"- {pool_name}:")
        for horizon in args.horizons:
            h = pool_result["horizons"][str(horizon)]
            print(
                f"  {horizon}d: ret={h['mc_mean_ret_buyhold']['median'] * 100:+.3f}% "
                f"navB={h['mc_nav_end_buyhold']['median'] * 100:+.2f}% "
                f"dd={h['mc_max_dd_buyhold']['median'] * 100:.2f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
