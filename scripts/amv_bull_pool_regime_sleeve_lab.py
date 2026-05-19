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

from scripts.amv_bull_pool_listwise_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as LTR_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    OLD_STATE_FEATURES,
    build_ltr_dataset,
    default_folds,
)
from scripts.amv_bull_pool_ranker_lab import _rolling_sleeve_nav

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked.*")


DEFAULT_OUTPUT_ROOT = LTR_OUTPUT_ROOT.parent / "amv_bull_pool_regime_sleeve"


SLEEVE_SCORE_COLS = {
    "ret_5d": "_sleeve_ret_5d",
    "ret_20d": "_sleeve_ret_20d",
    "klen": "_sleeve_klen",
    "kmid2": "_sleeve_kmid2",
    "kbar": "_sleeve_kbar",
    "kbar_momentum": "_sleeve_kbar_momentum",
    "manual_p2_k0p5_r0": "_score_combo_p2_k0p5_r0",
    "manual_p3_k0p5_r0": "_score_combo_p3_k0p5_r0",
}


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_sleeves(value: str) -> list[str]:
    sleeves = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(sleeves) - set(SLEEVE_SCORE_COLS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown sleeves: {', '.join(unknown)}")
    if not sleeves:
        raise argparse.ArgumentTypeError("at least one sleeve is required")
    return sleeves


def add_sleeve_scores(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("_score_ret_5d").alias("_sleeve_ret_5d"),
            pl.col("rank_ret_20d").alias("_sleeve_ret_20d"),
            (-pl.col("rank_KLEN")).alias("_sleeve_klen"),
            pl.col("rank_KMID2").alias("_sleeve_kmid2"),
            ((-pl.col("rank_KLEN") + pl.col("rank_KMID2")) / 2.0).alias("_sleeve_kbar"),
            (
                (
                    -pl.col("rank_KLEN")
                    + pl.col("rank_KMID2")
                    + pl.col("rank_ret_5d")
                    + pl.col("rank_ret_20d")
                )
                / 4.0
            ).alias("_sleeve_kbar_momentum"),
        ]
    )


def build_daily_sleeves(
    df: pl.DataFrame,
    *,
    sleeves: list[str],
    horizon: int,
    top_n: int,
) -> pl.DataFrame:
    label_col = f"fwd_ret_{horizon}d"
    mfe_col = f"fwd_mfe_{horizon}d"
    baseline = df.group_by("date").agg(
        [
            pl.col(label_col).mean().alias("baseline_ret"),
            (pl.col(mfe_col) >= 0.15).mean().alias("baseline_hit15"),
        ]
    )
    frames: list[pl.DataFrame] = []
    for sleeve_id in sleeves:
        score_col = SLEEVE_SCORE_COLS[sleeve_id]
        selected = (
            df.filter(pl.col(score_col).is_not_null() & ~pl.col(score_col).is_nan())
            .sort(["date", score_col, "code"], descending=[False, True, False])
            .group_by("date", maintain_order=True)
            .head(top_n)
        )
        daily = (
            selected.group_by("date")
            .agg(
                [
                    pl.col(label_col).mean().alias("top_ret"),
                    (pl.col(mfe_col) >= 0.15).mean().alias("top_hit15"),
                    pl.len().alias("selected_rows"),
                ]
            )
            .join(baseline, on="date", how="inner")
            .with_columns(
                [
                    pl.lit(sleeve_id).alias("sleeve_id"),
                    (pl.col("top_ret") - pl.col("baseline_ret")).alias("edge_ret"),
                    (pl.col("top_hit15") - pl.col("baseline_hit15")).alias("edge_hit15"),
                    pl.col("date").dt.year().alias("year"),
                ]
            )
        )
        frames.append(daily)
    return pl.concat(frames, how="vertical").sort(["date", "sleeve_id"])


def build_state_by_date(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("date")
        .agg(
            [
                pl.col("year").first(),
                *[pl.col(col).first().alias(col) for col in OLD_STATE_FEATURES],
            ]
        )
        .sort("date")
    )


def calc_daily_metrics(
    daily: pl.DataFrame,
    *,
    strategy_id: str,
    fold_id: str,
    horizon: int,
) -> dict[str, Any]:
    daily_ret = daily["top_ret"].to_numpy()
    baseline_ret = daily["baseline_ret"].to_numpy()
    nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
    baseline_nav_end, baseline_max_dd = _rolling_sleeve_nav(baseline_ret, horizon)
    return {
        "strategy_id": strategy_id,
        "fold_id": fold_id,
        "test_year": int(daily["year"].max()) if daily.height else None,
        "days": daily.height,
        "mean_ret": float(daily["top_ret"].mean()),
        "baseline_mean_ret": float(daily["baseline_ret"].mean()),
        "edge_ret": float(daily["edge_ret"].mean()),
        "positive_edge_ratio": float((daily["edge_ret"] > 0).mean()),
        "top_hit15": float(daily["top_hit15"].mean()),
        "baseline_hit15": float(daily["baseline_hit15"].mean()),
        "edge_hit15": float(daily["edge_hit15"].mean()),
        "nav_end": nav_end,
        "baseline_nav_end": baseline_nav_end,
        "edge_nav_end": nav_end - baseline_nav_end,
        "max_dd": max_dd,
        "baseline_max_dd": baseline_max_dd,
    }


def split_by_year(df: pl.DataFrame, start_year: int, end_year: int) -> pl.DataFrame:
    return df.filter((pl.col("year") >= start_year) & (pl.col("year") <= end_year)).sort("date")


def model_daily_choices(
    daily_sleeves: pl.DataFrame,
    state_by_date: pl.DataFrame,
    *,
    sleeve_ids: list[str],
    horizon: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pl.DataFrame, pl.DataFrame]:
    label_map = {sleeve_id: idx for idx, sleeve_id in enumerate(sleeve_ids)}
    inverse_label_map = {idx: sleeve_id for sleeve_id, idx in label_map.items()}
    best_by_date = (
        daily_sleeves.sort(["date", "top_ret", "sleeve_id"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(1)
        .select(["date", pl.col("sleeve_id").alias("best_sleeve")])
    )
    state = (
        state_by_date.join(best_by_date, on="date", how="inner")
        .with_columns(pl.col("best_sleeve").replace_strict(label_map).cast(pl.Int32).alias("label"))
        .sort("date")
    )
    metrics: list[dict[str, Any]] = []
    choice_frames: list[pl.DataFrame] = []
    importance_frames: list[pl.DataFrame] = []
    feature_cols = OLD_STATE_FEATURES

    for fold in default_folds():
        fold_id = str(fold["id"])
        train_state = split_by_year(state, fold["train_start"], fold["train_end"])
        valid_state = split_by_year(state, fold["valid_year"], fold["valid_year"])
        test_state = split_by_year(state, fold["test_year"], fold["test_year"])
        if train_state.is_empty() or valid_state.is_empty() or test_state.is_empty():
            continue

        train_sleeves = split_by_year(daily_sleeves, fold["train_start"], fold["valid_year"])
        train_best = (
            train_sleeves.group_by("sleeve_id")
            .agg(pl.col("top_ret").mean().alias("mean_ret"))
            .sort(["mean_ret", "sleeve_id"], descending=[True, False])
            .row(0, named=True)["sleeve_id"]
        )

        model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                train_state.select(feature_cols).to_numpy().astype(np.float32),
                train_state["label"].to_numpy().astype(np.int32),
                eval_set=[
                    (
                        valid_state.select(feature_cols).to_numpy().astype(np.float32),
                        valid_state["label"].to_numpy().astype(np.int32),
                    )
                ],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

        pred = model.predict(test_state.select(feature_cols).to_numpy().astype(np.float32))
        test_choices = test_state.select(["date", "year"]).with_columns(
            [
                pl.Series("chosen_sleeve", [inverse_label_map[int(idx)] for idx in pred]),
                pl.lit(fold_id).alias("fold_id"),
            ]
        )
        choice_daily = (
            test_choices.join(
                daily_sleeves,
                left_on=["date", "chosen_sleeve"],
                right_on=["date", "sleeve_id"],
                how="inner",
            )
            .select(
                [
                    "date",
                    "year",
                    "fold_id",
                    "chosen_sleeve",
                    "top_ret",
                    "baseline_ret",
                    "edge_ret",
                    "top_hit15",
                    "baseline_hit15",
                    "edge_hit15",
                ]
            )
            .sort("date")
        )
        metrics.append(
            calc_daily_metrics(
                choice_daily,
                strategy_id="state_classifier",
                fold_id=fold_id,
                horizon=horizon,
            )
        )

        train_best_daily = (
            test_state.select(["date", "year"])
            .with_columns(pl.lit(train_best).alias("chosen_sleeve"), pl.lit(fold_id).alias("fold_id"))
            .join(
                daily_sleeves,
                left_on=["date", "chosen_sleeve"],
                right_on=["date", "sleeve_id"],
                how="inner",
            )
            .select(
                [
                    "date",
                    "year",
                    "fold_id",
                    "chosen_sleeve",
                    "top_ret",
                    "baseline_ret",
                    "edge_ret",
                    "top_hit15",
                    "baseline_hit15",
                    "edge_hit15",
                ]
            )
            .sort("date")
        )
        metrics.append(
            calc_daily_metrics(
                train_best_daily,
                strategy_id="train_best_sleeve",
                fold_id=fold_id,
                horizon=horizon,
            )
        )
        choice_frames.extend(
            [
                choice_daily.with_columns(pl.lit("state_classifier").alias("strategy_id")),
                train_best_daily.with_columns(pl.lit("train_best_sleeve").alias("strategy_id")),
            ]
        )
        importance_frames.append(
            pl.DataFrame(
                {
                    "fold_id": [fold_id] * len(feature_cols),
                    "test_year": [fold["test_year"]] * len(feature_cols),
                    "feature": feature_cols,
                    "importance_gain": model.booster_.feature_importance(
                        importance_type="gain"
                    ).tolist(),
                    "importance_split": model.booster_.feature_importance(
                        importance_type="split"
                    ).tolist(),
                }
            )
        )

    choices_df = pl.concat(choice_frames, how="vertical") if choice_frames else pl.DataFrame()
    importance_df = pl.concat(importance_frames, how="vertical") if importance_frames else pl.DataFrame()
    return metrics, choices_df, importance_df


def summarize_metrics(metrics_df: pl.DataFrame) -> dict[str, Any]:
    summary = (
        metrics_df.group_by("strategy_id")
        .agg(
            [
                pl.col("edge_ret").mean().alias("avg_edge_ret"),
                pl.col("mean_ret").mean().alias("avg_mean_ret"),
                pl.col("positive_edge_ratio").mean().alias("avg_positive_edge_ratio"),
                pl.col("edge_nav_end").mean().alias("avg_edge_nav_end"),
                pl.col("test_year").n_unique().alias("test_years"),
            ]
        )
        .sort("avg_edge_ret", descending=True)
    )
    return {
        "by_strategy": summary.to_dicts(),
        "by_year": metrics_df.sort(["test_year", "edge_ret"], descending=[False, True]).to_dicts(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool regime-aware factor sleeve lab")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--label-mode", choices=["next_open_to_close"], default="next_open_to_close")
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--exclude-limit-up-entry", action="store_true")
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--sleeves",
        type=parse_sleeves,
        default=list(SLEEVE_SCORE_COLS),
        help=f"Comma-separated sleeves. Available: {', '.join(SLEEVE_SCORE_COLS)}",
    )
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument("--state-top-n", type=int, default=20)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building executable AMV bull pool dataset...")
    df, _ = build_ltr_dataset(args)
    df = add_sleeve_scores(df)
    print(
        f"Rows={df.height:,}, dates={df['date'].min()} -> {df['date'].max()}, "
        f"codes={df['code'].n_unique():,}, sleeves={', '.join(args.sleeves)}"
    )

    daily_sleeves = build_daily_sleeves(
        df,
        sleeves=args.sleeves,
        horizon=args.horizon,
        top_n=args.top_n,
    )
    state_by_date = build_state_by_date(df)
    metrics: list[dict[str, Any]] = []

    for fold in default_folds():
        fold_id = str(fold["id"])
        test_daily = split_by_year(daily_sleeves, fold["test_year"], fold["test_year"])
        for sleeve_id in args.sleeves:
            sleeve_daily = test_daily.filter(pl.col("sleeve_id") == sleeve_id)
            if not sleeve_daily.is_empty():
                metrics.append(
                    calc_daily_metrics(
                        sleeve_daily,
                        strategy_id=f"static_{sleeve_id}",
                        fold_id=fold_id,
                        horizon=args.horizon,
                    )
                )
        oracle_daily = (
            test_daily.sort(["date", "top_ret", "sleeve_id"], descending=[False, True, False])
            .group_by("date", maintain_order=True)
            .head(1)
            .sort("date")
        )
        if not oracle_daily.is_empty():
            metrics.append(
                calc_daily_metrics(
                    oracle_daily,
                    strategy_id="daily_oracle",
                    fold_id=fold_id,
                    horizon=args.horizon,
                )
            )

    model_metrics, model_choices, model_importance = model_daily_choices(
        daily_sleeves,
        state_by_date,
        sleeve_ids=args.sleeves,
        horizon=args.horizon,
        args=args,
    )
    metrics.extend(model_metrics)
    metrics_df = pl.DataFrame(metrics)

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_sleeves.write_csv(output_dir / "daily_sleeves.csv")
    state_by_date.write_csv(output_dir / "state_by_date.csv")
    metrics_df.write_csv(output_dir / "fold_metrics.csv")
    if not model_choices.is_empty():
        model_choices.write_csv(output_dir / "model_choices.csv")
    if not model_importance.is_empty():
        model_importance.write_csv(output_dir / "model_feature_importance.csv")

    summary = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon": args.horizon,
            "label_mode": args.label_mode,
            "execution_lag_days": args.execution_lag_days,
            "exclude_limit_up_entry": args.exclude_limit_up_entry,
            "top_n": args.top_n,
            "sleeves": args.sleeves,
            "state_features": OLD_STATE_FEATURES,
            "folds": default_folds(),
        },
        "pool": {
            "rows": df.height,
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "unique_codes": df["code"].n_unique(),
            "unique_dates": df["date"].n_unique(),
        },
        "files": {
            "daily_sleeves": "daily_sleeves.csv",
            "state_by_date": "state_by_date.csv",
            "fold_metrics": "fold_metrics.csv",
            "model_choices": "model_choices.csv",
            "model_feature_importance": "model_feature_importance.csv",
        },
        "metrics": summarize_metrics(metrics_df),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_dir / 'summary.json'}")
    print("Average edge by strategy:")
    for row in summary["metrics"]["by_strategy"]:
        print(f"- {row['strategy_id']}: edge={row['avg_edge_ret'] * 100:+.3f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
