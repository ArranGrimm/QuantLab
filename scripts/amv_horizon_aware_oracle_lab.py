from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_listwise_ranker_lab import DEFAULT_QMT_DB, build_ltr_dataset
from scripts.amv_bull_pool_regime_sleeve_lab import (
    DEFAULT_OUTPUT_ROOT as REGIME_OUTPUT_ROOT,
    SLEEVE_SCORE_COLS,
    add_sleeve_scores,
)


DEFAULT_OUTPUT_ROOT = REGIME_OUTPUT_ROOT.parent / "amv_horizon_aware_oracle"
DEFAULT_CANDIDATES = [
    "manual_p2_k0p5_r0:6",
    "manual_p3_k0p5_r0:6",
    "ret_5d:6",
    "ret_20d:2",
    "ret_20d:6",
    "klen:6",
    "kmid2:6",
    "kbar_momentum:6",
]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_candidates(value: str) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"candidate must be sleeve:horizon, got {item}")
        sleeve_id, horizon_raw = item.split(":", 1)
        if sleeve_id not in SLEEVE_SCORE_COLS:
            raise argparse.ArgumentTypeError(f"unknown sleeve: {sleeve_id}")
        try:
            horizon = int(horizon_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid horizon in {item}") from exc
        if horizon < 1:
            raise argparse.ArgumentTypeError(f"horizon must be >= 1 in {item}")
        parsed.append((sleeve_id, horizon))
    if not parsed:
        raise argparse.ArgumentTypeError("at least one candidate is required")
    return parsed


def candidate_id(sleeve_id: str, horizon: int) -> str:
    return f"{sleeve_id}_{horizon}td"


def clone_args(args: argparse.Namespace, *, horizon: int) -> argparse.Namespace:
    values = vars(args).copy()
    values["horizon"] = horizon
    return argparse.Namespace(**values)


def build_candidate_daily(
    args: argparse.Namespace,
    *,
    sleeve_id: str,
    horizon: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    horizon_args = clone_args(args, horizon=horizon)
    df, _ = build_ltr_dataset(horizon_args)
    df = add_sleeve_scores(df)
    score_col = SLEEVE_SCORE_COLS[sleeve_id]
    label_col = f"fwd_ret_{horizon}d"
    mfe_col = f"fwd_mfe_{horizon}d"
    mae_col = f"fwd_mae_{horizon}d"
    cid = candidate_id(sleeve_id, horizon)

    selected = (
        df.filter(pl.col(score_col).is_not_null() & ~pl.col(score_col).is_nan())
        .sort(["date", score_col, "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(args.top_n)
        .with_columns(
            [
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.lit(horizon).alias("horizon"),
                pl.lit(cid).alias("candidate_id"),
                pl.col(score_col).alias("score"),
                pl.col(score_col)
                .rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt32)
                .alias("rank"),
            ]
        )
    )
    daily = (
        selected.group_by("date")
        .agg(
            [
                pl.col("year").first().alias("year"),
                pl.col(label_col).mean().alias("top_ret"),
                pl.col(mfe_col).mean().alias("top_mfe"),
                pl.col(mae_col).mean().alias("top_mae"),
                (pl.col(mfe_col) >= 0.15).mean().alias("top_hit15"),
                pl.len().alias("selected_rows"),
            ]
        )
        .with_columns(
            [
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.lit(horizon).alias("horizon"),
                pl.lit(cid).alias("candidate_id"),
                (((1.0 + pl.col("top_ret")).clip(0.0001, None) ** (1.0 / horizon)) - 1.0).alias(
                    "top_ret_dailyized"
                ),
            ]
        )
        .sort("date")
    )
    selected_keep = selected.select(
        [
            "date",
            "code",
            "year",
            "candidate_id",
            "sleeve_id",
            "horizon",
            "score",
            "rank",
            pl.col(label_col).alias("fwd_exec_ret"),
            pl.col(mfe_col).alias("fwd_exec_mfe"),
            pl.col(mae_col).alias("fwd_exec_mae"),
            "fwd_exec_entry_gap",
        ]
    )
    return daily, selected_keep


def filter_common_dates(daily: pl.DataFrame, candidate_count: int) -> pl.DataFrame:
    common_dates = (
        daily.group_by("date")
        .agg(pl.col("candidate_id").n_unique().alias("candidate_count"))
        .filter(pl.col("candidate_count") == candidate_count)
        .select("date")
    )
    return daily.join(common_dates, on="date", how="inner")


def build_oracle(daily: pl.DataFrame, *, metric_col: str, strategy_id: str) -> pl.DataFrame:
    chosen = (
        daily.sort(["date", metric_col, "candidate_id"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(1)
        .sort("date")
        .with_columns(pl.lit(strategy_id).alias("strategy_id"))
    )
    cash_rows = (
        chosen.filter(pl.col(metric_col) <= 0)
        .with_columns(
            [
                pl.lit("cash").alias("candidate_id"),
                pl.lit("cash").alias("sleeve_id"),
                pl.lit(0).alias("horizon"),
                pl.lit(0.0).alias("top_ret"),
                pl.lit(0.0).alias("top_ret_dailyized"),
                pl.lit(0.0).alias("top_mfe"),
                pl.lit(0.0).alias("top_mae"),
                pl.lit(0.0).alias("top_hit15"),
            ]
        )
        .select(chosen.columns)
    )
    non_cash_rows = chosen.filter(pl.col(metric_col) > 0)
    return pl.concat([non_cash_rows, cash_rows], how="vertical").sort("date")


def summarize_daily(daily: pl.DataFrame, *, id_col: str) -> pl.DataFrame:
    return (
        daily.group_by(id_col)
        .agg(
            [
                pl.len().alias("days"),
                pl.col("top_ret").mean().alias("mean_ret"),
                pl.col("top_ret_dailyized").mean().alias("mean_dailyized_ret"),
                (pl.col("top_ret") > 0).mean().alias("positive_ret_ratio"),
                pl.col("top_hit15").mean().alias("mean_hit15"),
                pl.col("top_mfe").mean().alias("mean_mfe"),
                pl.col("top_mae").mean().alias("mean_mae"),
            ]
        )
        .sort(["mean_ret", id_col], descending=[True, False])
    )


def summarize_by_year(daily: pl.DataFrame, *, id_col: str) -> pl.DataFrame:
    return (
        daily.group_by(["year", id_col])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("top_ret").mean().alias("mean_ret"),
                pl.col("top_ret_dailyized").mean().alias("mean_dailyized_ret"),
                (pl.col("top_ret") > 0).mean().alias("positive_ret_ratio"),
            ]
        )
        .sort(["year", "mean_ret"], descending=[False, True])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV horizon-aware hindsight sleeve oracle lab")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--label-mode", choices=["next_open_to_close"], default="next_open_to_close")
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--exclude-limit-up-entry", action="store_true")
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--candidates", type=parse_candidates, default=parse_candidates(",".join(DEFAULT_CANDIDATES)))
    parser.add_argument("--require-all-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument("--state-top-n", type=int, default=20)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building horizon-aware sleeve candidate panel...")
    daily_frames: list[pl.DataFrame] = []
    selected_frames: list[pl.DataFrame] = []
    for sleeve_id, horizon in args.candidates:
        print(f"- {candidate_id(sleeve_id, horizon)}")
        daily, selected = build_candidate_daily(args, sleeve_id=sleeve_id, horizon=horizon)
        daily_frames.append(daily)
        selected_frames.append(selected)

    daily_candidates = pl.concat(daily_frames, how="vertical")
    selected = pl.concat(selected_frames, how="vertical")
    if args.require_all_candidates:
        daily_candidates = filter_common_dates(daily_candidates, len(args.candidates))

    oracle_raw = build_oracle(
        daily_candidates,
        metric_col="top_ret",
        strategy_id="oracle_raw_return_with_cash",
    )
    oracle_dailyized = build_oracle(
        daily_candidates,
        metric_col="top_ret_dailyized",
        strategy_id="oracle_dailyized_with_cash",
    )
    oracle = pl.concat([oracle_raw, oracle_dailyized], how="vertical")

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_candidates.write_csv(output_dir / "daily_candidate_sleeves.csv")
    selected.write_csv(output_dir / "selected_candidate_signals.csv")
    oracle.write_csv(output_dir / "daily_oracle_choices.csv")
    candidate_summary = summarize_daily(daily_candidates, id_col="candidate_id")
    candidate_year = summarize_by_year(daily_candidates, id_col="candidate_id")
    oracle_summary = summarize_daily(oracle, id_col="strategy_id")
    oracle_year = summarize_by_year(oracle, id_col="strategy_id")
    choice_summary = (
        oracle.group_by(["strategy_id", "candidate_id"])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("top_ret").mean().alias("mean_ret"),
                pl.col("top_ret_dailyized").mean().alias("mean_dailyized_ret"),
            ]
        )
        .sort(["strategy_id", "days"], descending=[False, True])
    )
    for name, frame in [
        ("candidate_summary.csv", candidate_summary),
        ("candidate_year_summary.csv", candidate_year),
        ("oracle_summary.csv", oracle_summary),
        ("oracle_year_summary.csv", oracle_year),
        ("oracle_choice_summary.csv", choice_summary),
    ]:
        frame.write_csv(output_dir / name)

    summary: dict[str, Any] = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label_mode": args.label_mode,
            "execution_lag_days": args.execution_lag_days,
            "exclude_limit_up_entry": args.exclude_limit_up_entry,
            "top_n": args.top_n,
            "require_all_candidates": args.require_all_candidates,
            "candidates": [candidate_id(sleeve, horizon) for sleeve, horizon in args.candidates],
        },
        "panel": {
            "dates": daily_candidates["date"].n_unique(),
            "date_min": str(daily_candidates["date"].min()),
            "date_max": str(daily_candidates["date"].max()),
            "rows": daily_candidates.height,
        },
        "metrics": {
            "candidate_summary": candidate_summary.to_dicts(),
            "oracle_summary": oracle_summary.to_dicts(),
            "choice_summary": choice_summary.to_dicts(),
            "candidate_year_summary": candidate_year.to_dicts(),
            "oracle_year_summary": oracle_year.to_dicts(),
        },
        "files": {
            "daily_candidate_sleeves": "daily_candidate_sleeves.csv",
            "selected_candidate_signals": "selected_candidate_signals.csv",
            "daily_oracle_choices": "daily_oracle_choices.csv",
            "candidate_summary": "candidate_summary.csv",
            "candidate_year_summary": "candidate_year_summary.csv",
            "oracle_summary": "oracle_summary.csv",
            "oracle_year_summary": "oracle_year_summary.csv",
            "oracle_choice_summary": "oracle_choice_summary.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_dir / 'summary.json'}")
    print("Candidate mean returns:")
    for row in candidate_summary.to_dicts():
        print(
            f"- {row['candidate_id']}: mean={row['mean_ret'] * 100:+.2f}% "
            f"dailyized={row['mean_dailyized_ret'] * 100:+.2f}% days={row['days']}"
        )
    print("Oracle choices:")
    for row in choice_summary.to_dicts():
        print(
            f"- {row['strategy_id']} -> {row['candidate_id']}: "
            f"days={row['days']} mean={row['mean_ret'] * 100:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
