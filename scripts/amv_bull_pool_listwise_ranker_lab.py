from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
    _rolling_sleeve_nav,
)
from utils.active_market_value_regime import build_active_market_value_regime_frame

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked.*")


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_listwise_ranker"

STOCK_FEATURES = [
    "price_pos_20d",
    "close_to_high_20d",
    "KLEN",
    "KMID2",
    "ret_5d",
    "ret_20d",
    "atr_14_pct",
    "panic_vol_ratio_20d",
    "intraday_pos",
    "turnover_ma_ratio",
    "market_cap_100m",
    "amount_ma20",
]
CORE_STOCK_FEATURES = [
    "price_pos_20d",
    "close_to_high_20d",
    "KLEN",
    "KMID2",
]
KBAR_STOCK_FEATURES = [
    "KLEN",
    "KMID2",
]
MOMENTUM_STOCK_FEATURES = [
    "ret_5d",
    "ret_20d",
]
RISK_STOCK_FEATURES = [
    "atr_14_pct",
    "panic_vol_ratio_20d",
]
LIQUIDITY_STOCK_FEATURES = [
    "market_cap_100m",
    "amount_ma20",
]
EXECUTION_STOCK_FEATURES = [
    "intraday_pos",
    "turnover_ma_ratio",
]
STATE_FEATURES = [
    "bull_day_scaled",
    "bull_phase_code",
    "amv_ret_1d_scaled",
    "amv_ret_2d_scaled",
    "amv_bull_trigger_ret_scaled",
    "trail_pool_ret_5d",
    "pool_up_ratio_5d",
    "pool_ret_5d_p90",
    "pool_topn_ret_5d",
    "pool_topn_ret_20d",
    "pool_amount_ma5_vs_20",
    "pool_candidate_count_scaled",
]
NEW_STATE_FEATURES = [
    "pool_topn_ret_5d",
    "pool_topn_ret_20d",
    "pool_amount_ma5_vs_20",
]
OLD_STATE_FEATURES = [col for col in STATE_FEATURES if col not in set(NEW_STATE_FEATURES)]
BASELINE_SCORE_COLS = {
    "baseline_p2_k0p5_r0": "_score_combo_p2_k0p5_r0",
    "baseline_p3_k0p5_r0": "_score_combo_p3_k0p5_r0",
    "baseline_klen": "_score_klen",
    "baseline_kmid2": "_score_kmid2",
    "baseline_ret_5d": "_score_ret_5d",
}
DEFAULT_VARIANTS = [
    "full",
    "no_risk",
    "stock_only",
    "core_state",
    "kbar_momentum_state",
    "momentum_state",
]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def combo_rankers() -> list[dict[str, Any]]:
    return [
        {
            "id": "combo_p2_k0p5_r0",
            "label": "当前组合 P2/K0.5/R0",
            "group": "baseline",
            "components": [
                {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
                {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
                {"factor": "KLEN", "higher_is_better": False, "weight": 0.5},
                {"factor": "KMID2", "higher_is_better": True, "weight": 0.5},
            ],
        },
        {
            "id": "combo_p3_k0p5_r0",
            "label": "高位+K线确认 P3/K0.5/R0",
            "group": "baseline",
            "components": [
                {"factor": "price_pos_20d", "higher_is_better": True, "weight": 3.0},
                {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 3.0},
                {"factor": "KLEN", "higher_is_better": False, "weight": 0.5},
                {"factor": "KMID2", "higher_is_better": True, "weight": 0.5},
            ],
        },
    ]


def rank_cols(features: list[str]) -> list[str]:
    return [f"rank_{col}" for col in features]


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("at least one variant is required")
    unknown = sorted(set(variants) - set(feature_variants().keys()))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {', '.join(unknown)}")
    return variants


def feature_variants() -> dict[str, list[str]]:
    stock_cols = rank_cols(STOCK_FEATURES)
    risk_cols = set(rank_cols(RISK_STOCK_FEATURES))
    no_risk_cols = [col for col in stock_cols if col not in risk_cols]
    return {
        "full": [*stock_cols, *STATE_FEATURES],
        "full_old_state": [*stock_cols, *OLD_STATE_FEATURES],
        "no_risk": no_risk_cols + STATE_FEATURES,
        "no_risk_old_state": no_risk_cols + OLD_STATE_FEATURES,
        "stock_only": stock_cols,
        "core_state": [*rank_cols(CORE_STOCK_FEATURES), *STATE_FEATURES],
        "kbar_momentum_state": [
            *rank_cols(KBAR_STOCK_FEATURES),
            *rank_cols(MOMENTUM_STOCK_FEATURES),
            *STATE_FEATURES,
        ],
        "kbar_momentum_old_state": [
            *rank_cols(KBAR_STOCK_FEATURES),
            *rank_cols(MOMENTUM_STOCK_FEATURES),
            *OLD_STATE_FEATURES,
        ],
        "momentum_state": [
            *rank_cols(MOMENTUM_STOCK_FEATURES),
            *STATE_FEATURES,
        ],
        "kbar_state": [
            *rank_cols(KBAR_STOCK_FEATURES),
            *STATE_FEATURES,
        ],
        "core_only": rank_cols(CORE_STOCK_FEATURES),
        "momentum_only": rank_cols(MOMENTUM_STOCK_FEATURES),
        "state_only": STATE_FEATURES,
        "no_liquidity": [
            col for col in stock_cols if col not in set(rank_cols(LIQUIDITY_STOCK_FEATURES))
        ]
        + STATE_FEATURES,
        "no_execution": [
            col for col in stock_cols if col not in set(rank_cols(EXECUTION_STOCK_FEATURES))
        ]
        + STATE_FEATURES,
    }


def build_bull_phase_frame(args: argparse.Namespace) -> pl.DataFrame:
    regime = (
        build_active_market_value_regime_frame(
            bull_trigger_pct=args.amv_bull_trigger_pct,
            bull_lookback_days=args.amv_bull_lookback_days,
            bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
            effective_lag_days=args.amv_effective_lag_days,
            date_col="date",
        )
        .select(
            [
                "date",
                "is_bull_regime",
                "amv_ret_1d",
                "amv_ret_2d",
                "amv_bull_trigger_ret",
            ]
        )
        .sort("date")
    )
    bull_run_id = 0
    bull_day = 0
    prev_bull = False
    run_ids: list[int | None] = []
    run_days: list[int | None] = []
    phases: list[str] = []
    phase_codes: list[int] = []

    for row in regime.select("is_bull_regime").to_dicts():
        is_bull = bool(row["is_bull_regime"])
        if is_bull and not prev_bull:
            bull_run_id += 1
            bull_day = 1
        elif is_bull:
            bull_day += 1
        else:
            bull_day = 0

        if not is_bull:
            run_ids.append(None)
            run_days.append(None)
            phases.append("non_bull")
            phase_codes.append(0)
        else:
            run_ids.append(bull_run_id)
            run_days.append(bull_day)
            if bull_day <= 5:
                phases.append("early")
                phase_codes.append(1)
            elif bull_day <= 20:
                phases.append("middle")
                phase_codes.append(2)
            else:
                phases.append("late")
                phase_codes.append(3)
        prev_bull = is_bull

    return regime.with_columns(
        [
            pl.Series("bull_run_id", run_ids, dtype=pl.Int64),
            pl.Series("bull_day", run_days, dtype=pl.Int64),
            pl.Series("bull_phase", phases),
            pl.Series("bull_phase_code", phase_codes, dtype=pl.Int8),
            (pl.Series("bull_day", run_days, dtype=pl.Int64).fill_null(0) / 100.0).alias(
                "bull_day_scaled"
            ),
            (pl.col("amv_ret_1d").fill_null(0.0) / 10.0).alias("amv_ret_1d_scaled"),
            (pl.col("amv_ret_2d").fill_null(0.0) / 10.0).alias("amv_ret_2d_scaled"),
            (pl.col("amv_bull_trigger_ret").fill_null(0.0) / 10.0).alias(
                "amv_bull_trigger_ret_scaled"
            ),
        ]
    )


def rank_normalize_expr(col_name: str) -> pl.Expr:
    valid = pl.col(col_name).is_not_null() & ~pl.col(col_name).is_nan()
    valid_count = pl.when(valid).then(1).otherwise(0).sum().over("date").cast(pl.Float64)
    rank = pl.col(col_name).rank(method="average").over("date").cast(pl.Float64)
    return (
        pl.when(valid & (valid_count > 1))
        .then((rank - 1.0) / (valid_count - 1.0) * 2.0 - 1.0)
        .otherwise(0.0)
        .alias(f"rank_{col_name}")
    )


def build_ltr_dataset(args: argparse.Namespace) -> tuple[pl.DataFrame, list[str]]:
    if args.label_mode == "next_open_to_close":
        args.horizons = sorted({args.horizon, args.horizon + args.execution_lag_days})
    else:
        args.horizons = [args.horizon]
    df_pool = add_combo_scores(build_dataset(args), combo_rankers())
    label_col = f"fwd_ret_{args.horizon}d"
    mfe_col = f"fwd_mfe_{args.horizon}d"
    mae_col = f"fwd_mae_{args.horizon}d"
    raw_label_col = f"raw_{label_col}"
    raw_mfe_col = f"raw_{mfe_col}"
    raw_mae_col = f"raw_{mae_col}"
    if args.label_mode == "next_open_to_close":
        df_pool = df_pool.with_columns(
            [
                pl.col(label_col).alias(raw_label_col),
                pl.col(mfe_col).alias(raw_mfe_col),
                pl.col(mae_col).alias(raw_mae_col),
                pl.col(f"fwd_exec_ret_{args.horizon}d").alias(label_col),
                pl.col(f"fwd_exec_mfe_{args.horizon}d").alias(mfe_col),
                pl.col(f"fwd_exec_mae_{args.horizon}d").alias(mae_col),
            ]
        )
        if args.exclude_limit_up_entry:
            df_pool = df_pool.filter(~pl.col("fwd_exec_entry_limit_up").fill_null(True))

    top_momentum = (
        df_pool.sort(["date", "ret_5d", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(args.state_top_n)
        .group_by("date")
        .agg(
            [
                pl.col("ret_5d").mean().alias("pool_topn_ret_5d"),
                pl.col("ret_20d").mean().alias("pool_topn_ret_20d"),
            ]
        )
    )
    daily_env = (
        df_pool.group_by("date")
        .agg(
            [
                pl.col("ret_5d").mean().alias("trail_pool_ret_5d"),
                (pl.col("ret_5d") > 0).mean().alias("pool_up_ratio_5d"),
                pl.col("ret_5d").quantile(0.90).alias("pool_ret_5d_p90"),
                (pl.col("amount_ma5").sum() / pl.col("amount_ma20").sum() - 1.0).alias(
                    "pool_amount_ma5_vs_20"
                ),
                (pl.len() / 2000.0).alias("pool_candidate_count_scaled"),
            ]
        )
        .join(top_momentum, on="date", how="left")
        .sort("date")
    )
    df = (
        df_pool.join(daily_env, on="date", how="left")
        .join(build_bull_phase_frame(args), on="date", how="left")
        .with_columns(
            [
                (-pl.col("KLEN")).alias("_score_klen"),
                pl.col("KMID2").alias("_score_kmid2"),
                pl.col("ret_5d").alias("_score_ret_5d"),
                pl.col("date").dt.year().alias("year"),
            ]
        )
        .with_columns([rank_normalize_expr(col) for col in STOCK_FEATURES])
        .with_columns(
            [
                pl.col(col).fill_null(0.0).fill_nan(0.0)
                for col in [*STATE_FEATURES, *(f"rank_{col}" for col in STOCK_FEATURES)]
            ]
        )
        .filter(pl.col(label_col).is_not_null() & ~pl.col(label_col).is_nan())
    )

    label_rank = pl.col(label_col).rank(method="ordinal", descending=True).over("date")
    df = df.with_columns(
        [
            label_rank.alias("_label_rank"),
            pl.when(label_rank <= args.relevance_top_k)
            .then(args.relevance_top_k - label_rank + 1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("ltr_label"),
        ]
    ).sort(["date", "code"])

    feature_cols = [f"rank_{col}" for col in STOCK_FEATURES] + STATE_FEATURES
    keep_cols = [
        "date",
        "code",
        "year",
        label_col,
        mfe_col,
        mae_col,
        "ltr_label",
        *feature_cols,
        *BASELINE_SCORE_COLS.values(),
    ]
    if args.label_mode == "next_open_to_close":
        keep_cols.extend(
            [
                raw_label_col,
                raw_mfe_col,
                raw_mae_col,
                "fwd_exec_entry_gap",
                "fwd_exec_entry_limit_up",
            ]
        )
    return df.select(keep_cols), feature_cols


def group_sizes(df: pl.DataFrame) -> list[int]:
    return df.group_by("date", maintain_order=True).agg(pl.len().alias("n"))["n"].to_list()


def to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy(allow_copy=True).astype(np.float32)


def split_by_year(df: pl.DataFrame, start_year: int, end_year: int) -> pl.DataFrame:
    return df.filter((pl.col("year") >= start_year) & (pl.col("year") <= end_year)).sort(
        ["date", "code"]
    )


def evaluate_scores(
    df: pl.DataFrame,
    *,
    score_col: str,
    strategy_id: str,
    fold_id: str,
    horizon: int,
    top_n: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    label_col = f"fwd_ret_{horizon}d"
    mfe_col = f"fwd_mfe_{horizon}d"
    eligible_dates = (
        df.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= top_n)
        .select("date")
    )
    scored = df.join(eligible_dates, on="date", how="inner")
    selected = (
        scored.sort(["date", score_col, "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .select(["date", "code", score_col, label_col, mfe_col])
    )
    actual_top = (
        scored.sort(["date", label_col, "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .select(["date", "code"])
        .with_columns(pl.lit(True).alias("_is_actual_top"))
    )
    daily = (
        selected.join(actual_top, on=["date", "code"], how="left")
        .with_columns(pl.col("_is_actual_top").fill_null(False))
        .group_by("date")
        .agg(
            [
                pl.col(label_col).mean().alias("top_ret"),
                (pl.col(mfe_col) >= 0.15).mean().alias("top_hit15"),
                pl.col("_is_actual_top").mean().alias("precision_at_topn"),
            ]
        )
        .join(
            scored.group_by("date").agg(
                [
                    pl.col(label_col).mean().alias("baseline_ret"),
                    (pl.col(mfe_col) >= 0.15).mean().alias("baseline_hit15"),
                ]
            ),
            on="date",
            how="inner",
        )
        .with_columns(
            [
                (pl.col("top_ret") - pl.col("baseline_ret")).alias("edge_ret"),
                (pl.col("top_hit15") - pl.col("baseline_hit15")).alias("edge_hit15"),
                pl.lit(strategy_id).alias("strategy_id"),
                pl.lit(fold_id).alias("fold_id"),
                pl.col("date").dt.year().alias("year"),
            ]
        )
        .sort("date")
    )
    daily_ret = daily["top_ret"].to_numpy()
    baseline_ret = daily["baseline_ret"].to_numpy()
    nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
    baseline_nav_end, baseline_max_dd = _rolling_sleeve_nav(baseline_ret, horizon)
    metrics = {
        "strategy_id": strategy_id,
        "fold_id": fold_id,
        "days": daily.height,
        "mean_ret": float(daily["top_ret"].mean()),
        "baseline_mean_ret": float(daily["baseline_ret"].mean()),
        "edge_ret": float(daily["edge_ret"].mean()),
        "positive_edge_ratio": float((daily["edge_ret"] > 0).mean()),
        "precision_at_topn": float(daily["precision_at_topn"].mean()),
        "top_hit15": float(daily["top_hit15"].mean()),
        "baseline_hit15": float(daily["baseline_hit15"].mean()),
        "edge_hit15": float(daily["edge_hit15"].mean()),
        "nav_end": nav_end,
        "baseline_nav_end": baseline_nav_end,
        "edge_nav_end": nav_end - baseline_nav_end,
        "max_dd": max_dd,
        "baseline_max_dd": baseline_max_dd,
    }
    return metrics, daily


def train_fold(
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    variant_id: str,
    fold: dict[str, int],
    args: argparse.Namespace,
    include_baselines: bool,
) -> tuple[list[dict[str, Any]], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fold_id = str(fold["id"])
    train_df = split_by_year(df, fold["train_start"], fold["train_end"])
    valid_df = split_by_year(df, fold["valid_year"], fold["valid_year"])
    test_df = split_by_year(df, fold["test_year"], fold["test_year"])
    if train_df.is_empty() or valid_df.is_empty() or test_df.is_empty():
        raise ValueError(f"fold {fold_id} has empty train/valid/test set")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        label_gain=list(range(args.relevance_top_k + 1)),
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            to_numpy(train_df, feature_cols),
            train_df["ltr_label"].to_numpy().astype(np.int32),
            group=group_sizes(train_df),
            eval_set=[
                (
                    to_numpy(valid_df, feature_cols),
                    valid_df["ltr_label"].to_numpy().astype(np.int32),
                )
            ],
            eval_group=[group_sizes(valid_df)],
            eval_at=[args.top_n, 10],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        ltr_scores = model.predict(to_numpy(test_df, feature_cols))
    test_scored = test_df.with_columns(pl.Series("ltr_score", ltr_scores))
    metrics: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    score_cols = {f"ltr_{variant_id}": "ltr_score"}
    if include_baselines:
        score_cols.update(BASELINE_SCORE_COLS)
    for strategy_id, score_col in score_cols.items():
        strategy_metrics, daily = evaluate_scores(
            test_scored,
            score_col=score_col,
            strategy_id=strategy_id,
            fold_id=fold_id,
            horizon=args.horizon,
            top_n=args.top_n,
        )
        strategy_metrics.update(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_year": fold["valid_year"],
                "test_year": fold["test_year"],
                "feature_variant": variant_id if strategy_id.startswith("ltr_") else "baseline",
                "feature_count": len(feature_cols) if strategy_id.startswith("ltr_") else None,
            }
        )
        metrics.append(strategy_metrics)
        daily_frames.append(
            daily.with_columns(
                pl.lit(variant_id if strategy_id.startswith("ltr_") else "baseline").alias(
                    "feature_variant"
                )
            )
        )

    importance = pl.DataFrame(
        {
            "fold_id": [fold_id] * len(feature_cols),
            "test_year": [fold["test_year"]] * len(feature_cols),
            "feature_variant": [variant_id] * len(feature_cols),
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain").tolist(),
            "importance_split": model.booster_.feature_importance(importance_type="split").tolist(),
        }
    )
    prediction_cols = [
        "date",
        "code",
        "year",
        "ltr_score",
        f"fwd_ret_{args.horizon}d",
        f"fwd_mfe_{args.horizon}d",
        f"fwd_mae_{args.horizon}d",
    ]
    if args.label_mode == "next_open_to_close":
        prediction_cols.extend(
            [
                f"raw_fwd_ret_{args.horizon}d",
                f"raw_fwd_mfe_{args.horizon}d",
                f"raw_fwd_mae_{args.horizon}d",
                "fwd_exec_entry_gap",
                "fwd_exec_entry_limit_up",
            ]
        )
    predictions_top = (
        test_scored.sort(["date", "ltr_score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(args.top_n)
        .select(prediction_cols)
        .with_columns(pl.lit(fold_id).alias("fold_id"))
        .with_columns(pl.lit(variant_id).alias("feature_variant"))
    )
    return metrics, pl.concat(daily_frames, how="vertical"), importance, predictions_top


def default_folds() -> list[dict[str, int]]:
    return [
        {"id": 1, "train_start": 2019, "train_end": 2021, "valid_year": 2022, "test_year": 2023},
        {"id": 2, "train_start": 2019, "train_end": 2022, "valid_year": 2023, "test_year": 2024},
        {"id": 3, "train_start": 2019, "train_end": 2023, "valid_year": 2024, "test_year": 2025},
        {"id": 4, "train_start": 2019, "train_end": 2024, "valid_year": 2025, "test_year": 2026},
    ]


def summarize_metrics(metrics_df: pl.DataFrame) -> dict[str, Any]:
    summary = (
        metrics_df.group_by("strategy_id")
        .agg(
            [
                pl.col("edge_ret").mean().alias("avg_edge_ret"),
                pl.col("mean_ret").mean().alias("avg_mean_ret"),
                pl.col("precision_at_topn").mean().alias("avg_precision_at_topn"),
                pl.col("edge_hit15").mean().alias("avg_edge_hit15"),
                pl.col("positive_edge_ratio").mean().alias("avg_positive_edge_ratio"),
                pl.col("test_year").n_unique().alias("test_years"),
            ]
        )
        .sort("avg_edge_ret", descending=True)
    )
    by_year = metrics_df.sort(["test_year", "edge_ret"], descending=[False, True])
    return {
        "by_strategy": summary.to_dicts(),
        "by_year": by_year.to_dicts(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool listwise long ranker lab")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--label-mode",
        choices=["close_to_close", "next_open_to_close"],
        default="close_to_close",
        help="LTR label definition. next_open_to_close matches T+1 open entry and max-hold close exit.",
    )
    parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=1,
        help="Trading-day lag from signal date to executable entry date.",
    )
    parser.add_argument(
        "--exclude-limit-up-entry",
        action="store_true",
        help="Drop samples whose executable entry open is limit-up and therefore not buyable.",
    )
    parser.add_argument(
        "--price-limit-tolerance",
        type=float,
        default=0.001,
        help="Tolerance used when identifying A-share limit-up entry opens.",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument(
        "--state-top-n",
        type=int,
        default=20,
        help="Top-N pool size used by market-state momentum features.",
    )
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=DEFAULT_VARIANTS,
        help=(
            "Comma-separated feature variants. Available: "
            f"{', '.join(sorted(feature_variants().keys()))}"
        ),
    )
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.execution_lag_days < 1:
        raise ValueError("--execution-lag-days must be >= 1")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")

    started_at = datetime.now()
    print("Building AMV bull pool LTR dataset...")
    df, all_feature_cols = build_ltr_dataset(args)
    variant_feature_map = {
        variant_id: feature_variants()[variant_id]
        for variant_id in args.variants
    }
    print(
        f"Rows={df.height:,}, dates={df['date'].min()} -> {df['date'].max()}, "
        f"codes={df['code'].n_unique():,}, features={len(all_feature_cols)}, "
        f"variants={', '.join(args.variants)}"
    )

    all_metrics: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    importance_frames: list[pl.DataFrame] = []
    selected_frames: list[pl.DataFrame] = []
    for variant_idx, (variant_id, feature_cols) in enumerate(variant_feature_map.items()):
        print(f"\nVariant {variant_id}: features={len(feature_cols)}")
        for fold in default_folds():
            print(
                f"  Fold {fold['id']}: train={fold['train_start']}-{fold['train_end']} "
                f"valid={fold['valid_year']} test={fold['test_year']}"
            )
            metrics, daily, importance, selected = train_fold(
                df,
                feature_cols,
                variant_id=variant_id,
                fold=fold,
                args=args,
                include_baselines=variant_idx == 0,
            )
            all_metrics.extend(metrics)
            daily_frames.append(daily)
            importance_frames.append(importance)
            selected_frames.append(selected)
            ltr_row = next(row for row in metrics if row["strategy_id"] == f"ltr_{variant_id}")
            print(
                f"    LTR: edge={ltr_row['edge_ret'] * 100:+.3f}pp "
                f"precision@{args.top_n}={ltr_row['precision_at_topn']:.3f}"
            )

    metrics_df = pl.DataFrame(all_metrics)
    daily_df = pl.concat(daily_frames, how="vertical")
    importance_df = pl.concat(importance_frames, how="vertical")
    selected_df = pl.concat(selected_frames, how="vertical")
    feature_importance = (
        importance_df.group_by(["feature_variant", "feature"])
        .agg(
            [
                pl.col("importance_gain").mean().alias("mean_gain"),
                pl.col("importance_split").mean().alias("mean_split"),
            ]
        )
        .sort(["feature_variant", "mean_gain"], descending=[False, True])
    )

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.write_csv(output_dir / "fold_metrics.csv")
    daily_df.write_csv(output_dir / "daily_metrics.csv")
    selected_df.write_csv(output_dir / "ltr_topn_selected.csv")
    importance_df.write_csv(output_dir / "feature_importance_by_fold.csv")
    feature_importance.write_csv(output_dir / "feature_importance.csv")
    config = {
        "qmt_db": str(args.qmt_db),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizon": args.horizon,
        "label_mode": args.label_mode,
        "execution_lag_days": args.execution_lag_days,
        "exclude_limit_up_entry": args.exclude_limit_up_entry,
        "price_limit_tolerance": args.price_limit_tolerance,
        "top_n": args.top_n,
        "relevance_top_k": args.relevance_top_k,
        "state_top_n": args.state_top_n,
        "model": {
            "library": "LightGBM",
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": "linear",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
        },
        "features": all_feature_cols,
        "all_features": all_feature_cols,
        "feature_variants": variant_feature_map,
        "folds": default_folds(),
    }
    summary = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": config,
        "pool": {
            "rows": df.height,
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "unique_codes": df["code"].n_unique(),
            "unique_dates": df["date"].n_unique(),
        },
        "files": {
            "fold_metrics": "fold_metrics.csv",
            "daily_metrics": "daily_metrics.csv",
            "ltr_topn_selected": "ltr_topn_selected.csv",
            "feature_importance": "feature_importance.csv",
            "feature_importance_by_fold": "feature_importance_by_fold.csv",
        },
        "metrics": summarize_metrics(metrics_df),
        "top_features": (
            feature_importance.group_by("feature_variant", maintain_order=True)
            .head(10)
            .to_dicts()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_dir / 'summary.json'}")
    print("\nAverage edge by strategy:")
    for row in summary["metrics"]["by_strategy"]:
        print(
            f"- {row['strategy_id']}: edge={row['avg_edge_ret'] * 100:+.3f}pp "
            f"precision={row['avg_precision_at_topn']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
