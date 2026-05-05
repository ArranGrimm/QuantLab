from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from utils import get_st_blacklist_pl, load_daily_data_full
from utils.active_market_value_regime import build_active_market_value_regime_frame
from utils.alpha158_factors import calc_alpha158_factors, resolve_alpha158_group_config
from utils.rotation_factors import calc_rotation_factors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_topn"
ALPHA158_KBAR_COLS = tuple(resolve_alpha158_group_config("kbar_shape")["factor_cols"])


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _rel_path(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace("\\", "/")


def _finite_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan()


def _score_component(factor: str, *, higher_is_better: bool, weight: float) -> pl.Expr:
    # Polars rank descending=False gives the best item rank 1 for smaller-is-better signals.
    rank_descending = not higher_is_better
    return (
        pl.col(factor).rank(method="average", descending=rank_descending).over("date")
        / pl.len().over("date")
        * weight
    )


def build_feature_frame(args: argparse.Namespace) -> pl.DataFrame:
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

        q_factor = calc_alpha158_factors(
            calc_rotation_factors(q_full),
            use_kbar=True,
            price_fields=(),
            include=(),
        )

        keep_cols = [
            "date",
            "code",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "pre_close_adj",
            "market_cap_100m",
            "amount",
            "price_pos_20d",
            "close_to_high_20d",
            *ALPHA158_KBAR_COLS,
        ]

        df = (
            q_factor
            .with_columns(pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"))
            .select([*keep_cols, "amount_ma20"])
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
            .sort(["date", "code"])
        )
    finally:
        conn.close()


def build_signal_export(args: argparse.Namespace) -> tuple[pl.DataFrame, dict[str, Any]]:
    df = build_feature_frame(args)

    component_checks = [
        _finite_expr("price_pos_20d"),
        _finite_expr("close_to_high_20d"),
        _finite_expr("KLEN"),
        _finite_expr("KMID2"),
    ]
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= args.mv_min)
        & (pl.col("amount_ma20") >= args.amount_ma20_min)
        & component_checks[0]
        & component_checks[1]
        & component_checks[2]
        & component_checks[3]
    )

    df_scored = (
        df.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(
                (
                    _score_component("price_pos_20d", higher_is_better=True, weight=args.price_weight)
                    + _score_component("close_to_high_20d", higher_is_better=False, weight=args.price_weight)
                    + _score_component("KLEN", higher_is_better=False, weight=args.kbar_weight)
                    + _score_component("KMID2", higher_is_better=True, weight=args.kbar_weight)
                )
                / (2.0 * args.price_weight + 2.0 * args.kbar_weight)
            )
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(
            pl.col("_signal_score")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("_signal_rank")
        )
    )

    signal_rows = (
        df_scored
        .filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )

    trading_dates = df.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(
        pl.col("date").shift(-1).alias("execution_date")
    ).drop_nulls("execution_date")

    execution_signals = (
        signal_rows
        .join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .rename({"execution_date": "date"})
        .select(["date", "code", "signal_date", "score", "rank"])
    )

    export = (
        df.select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "is_bull_regime",
                "amv_mechanical_regime",
                "market_cap_100m",
                "amount_ma20",
            ]
        )
        .join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            [
                pl.col("signal_date").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
            ]
        )
        .sort(["date", "code"])
    )

    signal_count = int(export["is_signal"].sum())
    signal_days = export.filter(pl.col("is_signal")).select("date").n_unique()
    signal_rows_raw = signal_rows.height
    blocked_by_execution_regime = int(
        export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
    )

    summary = {
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": signal_rows_raw,
        "signal_rows_after_shift": signal_count,
        "signal_days_after_shift": signal_days,
        "signals_blocked_by_execution_bear_regime": blocked_by_execution_regime,
    }
    return export, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMV topN signals for bt-amv-topn")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--price-weight", type=float, default=2.0)
    parser.add_argument("--kbar-weight", type=float, default=0.5)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    output_dir = args.output_root / started_at.strftime("%Y%m%d_%H%M%S")
    signal_path = output_dir / "signal.parquet"
    meta_path = output_dir / "signal.meta.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building AMV topN signal export...")
    export, summary = build_signal_export(args)
    export.write_parquet(signal_path)

    meta = {
        "strategy": "amv_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": "top3_high_pos_kbar_p2_k0p5_r0",
        "model_name": "manual_combo_ranker",
        "feature_mode": "amv_topn_manual_combo",
        "feature_count": 4,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "top_n": args.top_n,
            "price_weight": args.price_weight,
            "kbar_weight": args.kbar_weight,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "summary": summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved signal: {signal_path}")
    print(f"Saved meta:   {meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Relative signal: {_rel_path(signal_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
