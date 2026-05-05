from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from utils import get_st_blacklist_pl, load_daily_data_full
from utils.active_market_value_regime import build_active_market_value_regime_frame
from utils.alpha158_factors import calc_alpha158_factors, resolve_alpha158_group_config
from utils.rotation_factors import FACTOR_COLS, calc_rotation_factors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QMT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_bull_pool_ranker_lab"
ALPHA158_KBAR_COLS = tuple(resolve_alpha158_group_config("kbar_shape")["factor_cols"])


RANKERS: list[dict[str, Any]] = [
    {"id": "ret_5d_desc", "label": "5日动量强", "factor": "ret_5d", "descending": True, "group": "动量"},
    {"id": "ret_5d_asc", "label": "5日短反转", "factor": "ret_5d", "descending": False, "group": "反转"},
    {"id": "ret_20d_desc", "label": "20日动量强", "factor": "ret_20d", "descending": True, "group": "动量"},
    {"id": "ret_20d_asc", "label": "20日反转", "factor": "ret_20d", "descending": False, "group": "反转"},
    {"id": "ma_bias_20_desc", "label": "20日均线强势", "factor": "ma_bias_20", "descending": True, "group": "位置"},
    {"id": "ma_bias_20_asc", "label": "20日均线回调", "factor": "ma_bias_20", "descending": False, "group": "位置"},
    {"id": "price_pos_20d_desc", "label": "20日高位强势", "factor": "price_pos_20d", "descending": True, "group": "位置"},
    {"id": "price_pos_20d_asc", "label": "20日低位反转", "factor": "price_pos_20d", "descending": False, "group": "位置"},
    {"id": "near_high_20d", "label": "接近20日新高", "factor": "close_to_high_20d", "descending": False, "group": "位置"},
    {"id": "far_from_high_20d", "label": "远离20日高点", "factor": "close_to_high_20d", "descending": True, "group": "位置"},
    {"id": "turnover_ma_ratio_desc", "label": "换手放大", "factor": "turnover_ma_ratio", "descending": True, "group": "量能"},
    {"id": "abnormal_vol_desc", "label": "异常放量", "factor": "abnormal_vol", "descending": True, "group": "量能"},
    {"id": "turnover_accel_desc", "label": "换手加速", "factor": "turnover_accel", "descending": True, "group": "量能"},
    {"id": "vol_price_corr_20d_desc", "label": "量价同涨", "factor": "vol_price_corr_20d", "descending": True, "group": "量价"},
    {"id": "vol_20d_asc", "label": "20日低波动", "factor": "vol_20d", "descending": False, "group": "风险"},
    {"id": "vol_compress_asc", "label": "波动压缩", "factor": "vol_compress", "descending": False, "group": "风险"},
    {"id": "atr_14_pct_asc", "label": "ATR低风险", "factor": "atr_14_pct", "descending": False, "group": "风险"},
    {"id": "panic_vol_ratio_20d_asc", "label": "卖压低", "factor": "panic_vol_ratio_20d", "descending": False, "group": "风险"},
    {"id": "intraday_pos_desc", "label": "收盘靠高", "factor": "intraday_pos", "descending": True, "group": "日内"},
    {"id": "intraday_pos_asc", "label": "收盘靠低", "factor": "intraday_pos", "descending": False, "group": "日内"},
    {"id": "disp_bias_20_desc", "label": "成本线上强势", "factor": "disp_bias_20", "descending": True, "group": "行为"},
    {"id": "disp_bias_20_asc", "label": "成本线下回归", "factor": "disp_bias_20", "descending": False, "group": "行为"},
    {"id": "KMID_desc", "label": "K线实体偏强", "factor": "KMID", "descending": True, "group": "Alpha158 K线"},
    {"id": "KMID_asc", "label": "K线实体偏弱", "factor": "KMID", "descending": False, "group": "Alpha158 K线"},
    {"id": "KLEN_asc", "label": "K线振幅收缩", "factor": "KLEN", "descending": False, "group": "Alpha158 K线"},
    {"id": "KLEN_desc", "label": "K线振幅放大", "factor": "KLEN", "descending": True, "group": "Alpha158 K线"},
    {"id": "KMID2_desc", "label": "实体占比偏强", "factor": "KMID2", "descending": True, "group": "Alpha158 K线"},
    {"id": "KMID2_asc", "label": "实体占比偏弱", "factor": "KMID2", "descending": False, "group": "Alpha158 K线"},
    {"id": "KUP_asc", "label": "上影线短", "factor": "KUP", "descending": False, "group": "Alpha158 K线"},
    {"id": "KUP_desc", "label": "上影线长", "factor": "KUP", "descending": True, "group": "Alpha158 K线"},
    {"id": "KUP2_asc", "label": "上影占比短", "factor": "KUP2", "descending": False, "group": "Alpha158 K线"},
    {"id": "KUP2_desc", "label": "上影占比长", "factor": "KUP2", "descending": True, "group": "Alpha158 K线"},
    {"id": "KLOW_asc", "label": "下影线短", "factor": "KLOW", "descending": False, "group": "Alpha158 K线"},
    {"id": "KLOW_desc", "label": "下影线长", "factor": "KLOW", "descending": True, "group": "Alpha158 K线"},
    {"id": "KLOW2_asc", "label": "下影占比短", "factor": "KLOW2", "descending": False, "group": "Alpha158 K线"},
    {"id": "KLOW2_desc", "label": "下影占比长", "factor": "KLOW2", "descending": True, "group": "Alpha158 K线"},
    {"id": "KSFT_desc", "label": "收盘位置偏高", "factor": "KSFT", "descending": True, "group": "Alpha158 K线"},
    {"id": "KSFT_asc", "label": "收盘位置偏低", "factor": "KSFT", "descending": False, "group": "Alpha158 K线"},
    {"id": "KSFT2_desc", "label": "收盘区间偏高", "factor": "KSFT2", "descending": True, "group": "Alpha158 K线"},
    {"id": "KSFT2_asc", "label": "收盘区间偏低", "factor": "KSFT2", "descending": False, "group": "Alpha158 K线"},
]


COMBO_RANKERS: list[dict[str, Any]] = [
    {
        "id": "combo_near_high_klen_lowrisk",
        "label": "新高+缩振+低风险",
        "group": "组合",
        "components": [
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 1.0},
            {"factor": "atr_14_pct", "higher_is_better": False, "weight": 1.0},
            {"factor": "panic_vol_ratio_20d", "higher_is_better": False, "weight": 1.0},
        ],
    },
    {
        "id": "combo_high_pos_kmid2_lowrisk",
        "label": "高位+实体强+低风险",
        "group": "组合",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "KMID2", "higher_is_better": True, "weight": 1.0},
            {"factor": "atr_14_pct", "higher_is_better": False, "weight": 1.0},
            {"factor": "panic_vol_ratio_20d", "higher_is_better": False, "weight": 1.0},
        ],
    },
    {
        "id": "combo_high_pos_kbar_confirm",
        "label": "高位+K线确认",
        "group": "组合",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 1.0},
            {"factor": "KMID2", "higher_is_better": True, "weight": 1.0},
        ],
    },
    {
        "id": "combo_near_high_kmid2_short_upper",
        "label": "新高+实体强+短上影",
        "group": "组合",
        "components": [
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KMID2", "higher_is_better": True, "weight": 1.0},
            {"factor": "KUP2", "higher_is_better": False, "weight": 1.0},
            {"factor": "panic_vol_ratio_20d", "higher_is_better": False, "weight": 1.0},
        ],
    },
    {
        "id": "combo_high_pos_ksft2_lowvol",
        "label": "高位+收盘强+低波",
        "group": "组合",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "KSFT2", "higher_is_better": True, "weight": 1.0},
            {"factor": "vol_20d", "higher_is_better": False, "weight": 1.0},
            {"factor": "atr_14_pct", "higher_is_better": False, "weight": 1.0},
        ],
    },
    {
        "id": "combo_near_high_klen_short_lower",
        "label": "新高+缩振+短下影",
        "group": "组合",
        "components": [
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 1.0},
            {"factor": "KLOW2", "higher_is_better": False, "weight": 1.0},
            {"factor": "atr_14_pct", "higher_is_better": False, "weight": 1.0},
        ],
    },
    {
        "id": "combo_reversal_lowrisk",
        "label": "反转+收低+低风险",
        "group": "组合",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": False, "weight": 1.5},
            {"factor": "ret_20d", "higher_is_better": False, "weight": 1.5},
            {"factor": "intraday_pos", "higher_is_better": False, "weight": 1.0},
            {"factor": "atr_14_pct", "higher_is_better": False, "weight": 1.0},
        ],
    },
]


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


def _finite_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan()


def _ranker_factor_cols(ranker: dict[str, Any]) -> list[str]:
    if "components" not in ranker:
        return [ranker["factor"]]
    return [str(component["factor"]) for component in ranker["components"]]


def _component_score_expr(component: dict[str, Any]) -> pl.Expr:
    factor = str(component["factor"])
    higher_is_better = bool(component.get("higher_is_better", True))
    weight = float(component.get("weight", 1.0))
    rank_descending = not higher_is_better
    return (
        pl.col(factor).rank(method="average", descending=rank_descending).over("date")
        / pl.len().over("date")
        * weight
    )


def add_combo_scores(df_pool: pl.DataFrame, rankers: list[dict[str, Any]]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for ranker in rankers:
        components = ranker.get("components")
        if not components:
            continue
        score_col = f"_score_{ranker['id']}"
        total_weight = sum(float(component.get("weight", 1.0)) for component in components)
        if total_weight <= 0:
            raise ValueError(f"combo ranker {ranker['id']} has non-positive total weight")
        ranker["factor"] = score_col
        ranker["factor_cols"] = _ranker_factor_cols(ranker)
        ranker["descending"] = True
        component_exprs = [_component_score_expr(component) for component in components]
        score_expr = component_exprs[0]
        for component_expr in component_exprs[1:]:
            score_expr = score_expr + component_expr
        exprs.append((score_expr / total_weight).alias(score_col))

    if not exprs:
        return df_pool
    return df_pool.with_columns(exprs)


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

        q_factor = calc_alpha158_factors(
            calc_rotation_factors(q_full),
            use_kbar=True,
            price_fields=(),
            include=(),
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
            *FACTOR_COLS,
            *ALPHA158_KBAR_COLS,
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
            q_factor.with_columns(future_exprs)
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
            .filter(pl.col("is_bull_regime"))
            .filter(
                (pl.col("market_cap_100m") >= args.mv_min)
                & (pl.col("amount_ma20") >= args.amount_ma20_min)
            )
        )
    finally:
        conn.close()


def _baseline_for_dates(df_pool: pl.DataFrame, dates: list[object], horizon: int) -> dict[str, float]:
    df_dates = df_pool.filter(pl.col("date").is_in(dates))
    per_day = (
        df_dates.group_by("date")
        .agg(
            [
                pl.col(f"fwd_ret_{horizon}d").mean().alias("daily_ret"),
                (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("hit15"),
            ]
        )
        .sort("date")
    )
    daily_ret = per_day["daily_ret"].to_numpy()
    nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
    return {
        "mean_ret": float(per_day["daily_ret"].mean()),
        "hit15": float(per_day["hit15"].mean()),
        "nav_end": nav_end,
        "max_dd": max_dd,
    }


def evaluate_ranker(
    df_pool: pl.DataFrame,
    ranker: dict[str, Any],
    *,
    horizons: list[int],
    top_n: int,
) -> dict[str, Any]:
    factor = ranker["factor"]
    finite_checks = [_finite_expr(col_name) for col_name in ranker.get("factor_cols", [factor])]
    valid_expr = finite_checks[0]
    for finite_check in finite_checks[1:]:
        valid_expr = valid_expr & finite_check
    df_valid = df_pool.filter(valid_expr & _finite_expr(factor))
    daily_counts = (
        df_valid.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= top_n)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    if not eligible_dates:
        return {
            **ranker,
            "eligible_days": 0,
            "error": f"no dates with at least {top_n} finite factor values",
        }

    selected = (
        df_valid.filter(pl.col("date").is_in(eligible_dates))
        .sort(["date", factor, "code"], descending=[False, ranker["descending"], False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "code"])
    )

    result: dict[str, Any] = {
        **ranker,
        "eligible_days": len(eligible_dates),
        "selected_rows": selected.height,
        "candidate_count": {
            "median": float(daily_counts["n_candidates"].median()),
            "mean": float(daily_counts["n_candidates"].mean()),
            "q05": float(daily_counts["n_candidates"].quantile(0.05)),
            "q95": float(daily_counts["n_candidates"].quantile(0.95)),
            "min": int(daily_counts["n_candidates"].min()),
            "max": int(daily_counts["n_candidates"].max()),
        },
        "horizons": {},
    }

    for horizon in horizons:
        per_day = (
            selected.group_by("date")
            .agg(
                [
                    pl.col(f"fwd_ret_{horizon}d").mean().alias("daily_ret"),
                    (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("hit15"),
                ]
            )
            .sort("date")
        )
        daily_ret = per_day["daily_ret"].to_numpy()
        nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
        mean_ret = float(per_day["daily_ret"].mean())
        hit15 = float(per_day["hit15"].mean())
        baseline = _baseline_for_dates(df_pool, eligible_dates, horizon)

        result["horizons"][str(horizon)] = {
            "mean_ret": mean_ret,
            "hit15": hit15,
            "nav_end": nav_end,
            "max_dd": max_dd,
            "random_baseline": baseline,
            "edge_ret": mean_ret - baseline["mean_ret"],
            "edge_hit15": hit15 - baseline["hit15"],
            "edge_nav_end": nav_end - baseline["nav_end"],
            "edge_max_dd": max_dd - baseline["max_dd"],
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool single-factor top-N ranking lab")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_horizons, default=[5, 10, 20])
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    started_at = datetime.now()
    print("Building AMV bull pool factor dataset...")
    df_pool = build_dataset(args)
    rankers = [dict(ranker) for ranker in [*RANKERS, *COMBO_RANKERS]]
    df_pool = add_combo_scores(df_pool, rankers)
    print(f"AMV bull LF2 rows: {df_pool.height:,}")
    print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
    print(f"Unique codes: {df_pool['code'].n_unique():,}")

    results = []
    for ranker in rankers:
        print(f"Evaluating {ranker['id']} ({ranker['label']})...")
        results.append(
            evaluate_ranker(
                df_pool,
                ranker,
                horizons=args.horizons,
                top_n=args.top_n,
            )
        )

    best_by_horizon = {}
    for horizon in args.horizons:
        horizon_key = str(horizon)
        valid = [r for r in results if "error" not in r]
        best_by_horizon[horizon_key] = sorted(
            [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "group": r["group"],
                    **r["horizons"][horizon_key],
                }
                for r in valid
            ],
            key=lambda row: (row["edge_ret"], row["mean_ret"]),
            reverse=True,
        )[:10]

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
            "top_n": args.top_n,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "pool": {
            "rows": df_pool.height,
            "date_min": str(df_pool["date"].min()),
            "date_max": str(df_pool["date"].max()),
            "unique_codes": df_pool["code"].n_unique(),
        },
        "rankers": results,
        "best_by_horizon": best_by_horizon,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    for horizon in args.horizons:
        print(f"\nTop rankers by {horizon}d edge over random baseline:")
        for row in best_by_horizon[str(horizon)][:5]:
            print(
                f"- {row['label']:<12} "
                f"ret={row['mean_ret'] * 100:+.3f}% "
                f"edge={row['edge_ret'] * 100:+.3f}pp "
                f"nav={row['nav_end'] * 100:+.2f}% "
                f"dd={row['max_dd'] * 100:.2f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
