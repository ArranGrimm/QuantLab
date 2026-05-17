from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_listwise_ranker_lab import (
    DEFAULT_QMT_DB,
    STATE_FEATURES,
    build_ltr_dataset,
)
from scripts.amv_horizon_aware_oracle_lab import DEFAULT_OUTPUT_ROOT as ORACLE_OUTPUT_ROOT


DEFAULT_OUTPUT_ROOT = ORACLE_OUTPUT_ROOT.parent / "amv_horizon_oracle_explainability"
DEFAULT_ORACLE_DIR = ORACLE_OUTPUT_ROOT / "20260517_130248"
FEATURE_LABELS = {
    "bull_day_scaled": "AMV bull age",
    "bull_phase_code": "AMV bull phase",
    "amv_ret_1d_scaled": "AMV 1d return",
    "amv_ret_2d_scaled": "AMV 2d return",
    "amv_bull_trigger_ret_scaled": "AMV trigger strength",
    "trail_pool_ret_5d": "pool 5d mean return",
    "pool_up_ratio_5d": "pool 5d up ratio",
    "pool_ret_5d_p90": "pool 5d p90 return",
    "pool_topn_ret_5d": "topN 5d momentum",
    "pool_topn_ret_20d": "topN 20d momentum",
    "pool_amount_ma5_vs_20": "pool amount 5v20",
    "pool_candidate_count_scaled": "pool candidate count",
}


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_state_frame(args: argparse.Namespace) -> pl.DataFrame:
    state_args = argparse.Namespace(
        qmt_db=args.qmt_db,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        st_snapshot_date=args.st_snapshot_date,
        mv_min=args.mv_min,
        amount_ma20_min=args.amount_ma20_min,
        horizon=args.state_horizon,
        label_mode="next_open_to_close",
        execution_lag_days=args.execution_lag_days,
        exclude_limit_up_entry=args.exclude_limit_up_entry,
        price_limit_tolerance=args.price_limit_tolerance,
        top_n=args.top_n,
        relevance_top_k=args.relevance_top_k,
        state_top_n=args.state_top_n,
        amv_bull_trigger_pct=args.amv_bull_trigger_pct,
        amv_bull_lookback_days=args.amv_bull_lookback_days,
        amv_bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
        amv_effective_lag_days=args.amv_effective_lag_days,
    )
    df, _ = build_ltr_dataset(state_args)
    return (
        df.group_by("date")
        .agg(
            [
                pl.col("year").first().alias("year"),
                *[pl.col(col).first().alias(col) for col in STATE_FEATURES],
            ]
        )
        .sort("date")
    )


def read_oracle(oracle_dir: Path) -> pl.DataFrame:
    path = oracle_dir / "daily_oracle_choices.csv"
    if not path.exists():
        raise FileNotFoundError(f"daily_oracle_choices.csv not found: {path}")
    return pl.read_csv(path, try_parse_dates=True)


def add_cash_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("candidate_id") == "cash").alias("is_cash"),
            pl.col("candidate_id").str.contains("manual_").alias("is_manual_combo"),
            pl.col("candidate_id").str.contains("ret_").alias("is_momentum"),
            pl.col("candidate_id").str.contains("_2td").alias("is_short_horizon"),
            pl.col("candidate_id").str.contains("_6td").alias("is_6td"),
        ]
    )


def summarize_choice_state(joined: pl.DataFrame) -> pl.DataFrame:
    return (
        joined.group_by(["strategy_id", "candidate_id"])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("top_ret").mean().alias("mean_ret"),
                pl.col("top_ret_dailyized").mean().alias("mean_dailyized_ret"),
                (pl.col("top_ret") > 0).mean().alias("positive_ret_ratio"),
                *[pl.col(col).mean().alias(col) for col in STATE_FEATURES],
            ]
        )
        .sort(["strategy_id", "days"], descending=[False, True])
    )


def summarize_phase_distribution(joined: pl.DataFrame) -> pl.DataFrame:
    counts = (
        joined.group_by(["strategy_id", "candidate_id", "bull_phase_code"])
        .agg(pl.len().alias("days"))
        .sort(["strategy_id", "candidate_id", "bull_phase_code"])
    )
    totals = counts.group_by(["strategy_id", "candidate_id"]).agg(
        pl.col("days").sum().alias("candidate_days")
    )
    return counts.join(totals, on=["strategy_id", "candidate_id"], how="left").with_columns(
        (pl.col("days") / pl.col("candidate_days")).alias("candidate_phase_share")
    )


def feature_separation(joined: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_id in joined["strategy_id"].unique().sort():
        df_s = joined.filter(pl.col("strategy_id") == strategy_id)
        for feature in STATE_FEATURES:
            overall_std = df_s[feature].std()
            if overall_std is None or overall_std == 0:
                continue
            by_candidate = (
                df_s.group_by("candidate_id")
                .agg(
                    [
                        pl.len().alias("days"),
                        pl.col(feature).mean().alias("feature_mean"),
                    ]
                )
                .filter(pl.col("days") >= 10)
                .sort("feature_mean")
            )
            if by_candidate.height < 2:
                continue
            low = by_candidate.row(0, named=True)
            high = by_candidate.row(by_candidate.height - 1, named=True)
            spread = (float(high["feature_mean"]) - float(low["feature_mean"])) / float(overall_std)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "overall_mean": float(df_s[feature].mean()),
                    "overall_std": float(overall_std),
                    "low_candidate": low["candidate_id"],
                    "low_mean": float(low["feature_mean"]),
                    "high_candidate": high["candidate_id"],
                    "high_mean": float(high["feature_mean"]),
                    "spread_std_units": spread,
                }
            )
    return pl.DataFrame(rows).sort(["strategy_id", "spread_std_units"], descending=[False, True])


def candidate_feature_diffs(joined: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_id in joined["strategy_id"].unique().sort():
        df_s = joined.filter(pl.col("strategy_id") == strategy_id)
        for candidate_id in df_s["candidate_id"].unique().sort():
            df_c = df_s.filter(pl.col("candidate_id") == candidate_id)
            df_rest = df_s.filter(pl.col("candidate_id") != candidate_id)
            if df_c.height < 10 or df_rest.height < 10:
                continue
            for feature in STATE_FEATURES:
                std = df_s[feature].std()
                if std is None or std == 0:
                    continue
                mean_c = float(df_c[feature].mean())
                mean_rest = float(df_rest[feature].mean())
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "candidate_id": candidate_id,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS.get(feature, feature),
                        "days": df_c.height,
                        "candidate_mean": mean_c,
                        "rest_mean": mean_rest,
                        "diff_std_units": (mean_c - mean_rest) / float(std),
                    }
                )
    return pl.DataFrame(rows).sort(
        ["strategy_id", "candidate_id", "diff_std_units"],
        descending=[False, False, True],
    )


def top_abs_diffs(diffs: pl.DataFrame, *, top_n: int) -> pl.DataFrame:
    return (
        diffs.with_columns(pl.col("diff_std_units").abs().alias("abs_diff"))
        .sort(["strategy_id", "candidate_id", "abs_diff"], descending=[False, False, True])
        .group_by(["strategy_id", "candidate_id"], maintain_order=True)
        .head(top_n)
        .drop("abs_diff")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain horizon-aware oracle sleeve choices")
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--state-horizon", type=int, default=6)
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--exclude-limit-up-entry", action="store_true")
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument("--state-top-n", type=int, default=20)
    parser.add_argument("--top-diffs", type=int, default=5)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Loading oracle choices...")
    oracle = read_oracle(args.oracle_dir)
    print("Building pre-trade state frame...")
    state = build_state_frame(args)
    joined = add_cash_features(oracle.join(state, on=["date", "year"], how="inner")).sort(
        ["strategy_id", "date"]
    )

    choice_state = summarize_choice_state(joined)
    phase_distribution = summarize_phase_distribution(joined)
    separation = feature_separation(joined)
    diffs = candidate_feature_diffs(joined)
    top_diffs = top_abs_diffs(diffs, top_n=args.top_diffs)

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.write_csv(output_dir / "oracle_choices_with_state.csv")
    choice_state.write_csv(output_dir / "choice_state_summary.csv")
    phase_distribution.write_csv(output_dir / "phase_choice_distribution.csv")
    separation.write_csv(output_dir / "feature_separation.csv")
    diffs.write_csv(output_dir / "candidate_feature_diffs.csv")
    top_diffs.write_csv(output_dir / "candidate_top_feature_diffs.csv")

    summary: dict[str, Any] = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "oracle_dir": str(args.oracle_dir),
            "qmt_db": str(args.qmt_db),
            "state_horizon": args.state_horizon,
            "exclude_limit_up_entry": args.exclude_limit_up_entry,
            "state_features": STATE_FEATURES,
        },
        "panel": {
            "rows": joined.height,
            "dates": joined["date"].n_unique(),
            "date_min": str(joined["date"].min()),
            "date_max": str(joined["date"].max()),
        },
        "metrics": {
            "choice_state_summary": choice_state.to_dicts(),
            "phase_choice_distribution": phase_distribution.to_dicts(),
            "feature_separation": separation.to_dicts(),
            "candidate_top_feature_diffs": top_diffs.to_dicts(),
        },
        "files": {
            "oracle_choices_with_state": "oracle_choices_with_state.csv",
            "choice_state_summary": "choice_state_summary.csv",
            "phase_choice_distribution": "phase_choice_distribution.csv",
            "feature_separation": "feature_separation.csv",
            "candidate_feature_diffs": "candidate_feature_diffs.csv",
            "candidate_top_feature_diffs": "candidate_top_feature_diffs.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved: {output_dir / 'summary.json'}")
    print("Top feature separation:")
    for row in separation.head(12).to_dicts():
        print(
            f"- {row['strategy_id']} {row['feature']}: "
            f"{row['low_candidate']}={row['low_mean']:.4f}, "
            f"{row['high_candidate']}={row['high_mean']:.4f}, "
            f"spread={row['spread_std_units']:.2f}std"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
