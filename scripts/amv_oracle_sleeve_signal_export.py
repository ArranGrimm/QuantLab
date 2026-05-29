from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import (
    DEFAULT_QMT_DB,
    ROOT,
    _git_commit,
    _rel_path,
    build_feature_frame,
)
from scripts.amv_bull_pool_regime_sleeve_lab import (
    SLEEVE_SCORE_COLS,
    add_sleeve_scores,
    parse_sleeves,
)
from scripts.amv_bull_pool_listwise_ranker_lab import build_ltr_dataset


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_oracle_sleeve_signals"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_oracle_selected(args: argparse.Namespace) -> tuple[pl.DataFrame, dict[str, Any]]:
    df, _ = build_ltr_dataset(args)
    df = add_sleeve_scores(df)
    label_col = f"fwd_ret_{args.horizon}d"
    sleeve_frames: list[pl.DataFrame] = []
    selected_frames: list[pl.DataFrame] = []
    for sleeve_id in args.sleeves:
        score_col = SLEEVE_SCORE_COLS[sleeve_id]
        selected = (
            df.filter(pl.col(score_col).is_not_null() & ~pl.col(score_col).is_nan())
            .sort(["date", score_col, "code"], descending=[False, True, False])
            .group_by("date", maintain_order=True)
            .head(args.top_n)
            .with_columns(
                [
                    pl.lit(sleeve_id).alias("sleeve_id"),
                    pl.col(score_col).alias("score"),
                    pl.col(score_col)
                    .rank(method="ordinal", descending=True)
                    .over("date")
                    .cast(pl.UInt32)
                    .alias("rank"),
                ]
            )
        )
        sleeve_daily = selected.group_by("date").agg(
            [
                pl.col(label_col).mean().alias("oracle_choice_ret"),
                pl.col("code").n_unique().alias("selected_codes"),
            ]
        ).with_columns(pl.lit(sleeve_id).alias("sleeve_id"))
        selected_frames.append(selected)
        sleeve_frames.append(sleeve_daily)

    daily_sleeves = pl.concat(sleeve_frames, how="vertical")
    oracle_choice = (
        daily_sleeves.sort(["date", "oracle_choice_ret", "sleeve_id"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(1)
        .select(["date", "sleeve_id", "oracle_choice_ret"])
    )
    all_selected = pl.concat(selected_frames, how="vertical")
    selected = (
        all_selected.join(oracle_choice, on=["date", "sleeve_id"], how="inner")
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                "year",
                "sleeve_id",
                "score",
                "rank",
                label_col,
                f"fwd_mfe_{args.horizon}d",
                f"fwd_mae_{args.horizon}d",
                "fwd_exec_entry_gap",
                "oracle_choice_ret",
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )
    choice_summary = (
        oracle_choice.group_by("sleeve_id")
        .agg(
            [
                pl.len().alias("days"),
                pl.col("oracle_choice_ret").mean().alias("mean_choice_ret"),
            ]
        )
        .sort("days", descending=True)
    )
    summary = {
        "dataset_rows": df.height,
        "dataset_date_min": str(df["date"].min()),
        "dataset_date_max": str(df["date"].max()),
        "oracle_days": oracle_choice.height,
        "selected_rows": selected.height,
        "choice_summary": choice_summary.to_dicts(),
        "label_mean": float(selected[label_col].mean()),
        "entry_gap_mean": float(selected["fwd_exec_entry_gap"].mean()),
    }
    return selected, summary


def build_signal_export(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    market = build_feature_frame(args)
    selected, oracle_summary = build_oracle_selected(args)

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(
        pl.col("date").shift(-1).alias("execution_date")
    ).drop_nulls("execution_date")
    execution_signals = (
        selected.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(
            [
                pl.col("execution_date").alias("date"),
                "code",
                "signal_date",
                "sleeve_id",
                "score",
                "rank",
            ]
        )
    )

    export = (
        market.select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "open_raw",
                "high_raw",
                "low_raw",
                "close_raw",
                "pre_close_raw",
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
                pl.col("sleeve_id").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )
    summary = {
        **oracle_summary,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": selected.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
    }
    return export, selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export hindsight oracle sleeve signals for bt-amv-topn")
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
    parser.add_argument("--sleeves", type=parse_sleeves, default=list(SLEEVE_SCORE_COLS))
    parser.add_argument("--relevance-top-k", type=int, default=100)
    parser.add_argument("--state-top-n", type=int, default=20)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    output_dir = args.output_root / f"{timestamp_token()}_oracle_{args.horizon}td"
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    selected_path = output_dir / "selected_signals.csv"
    meta_path = output_dir / "signal.meta.json"

    print("Building oracle sleeve signal export...")
    export, selected, summary = build_signal_export(args)
    export.write_parquet(signal_path)
    selected.write_csv(selected_path)

    meta: dict[str, Any] = {
        "strategy": "amv_oracle_sleeve_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_oracle_sleeve_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"oracle_sleeve_{args.horizon}td",
        "model_name": "hindsight_oracle",
        "feature_mode": "oracle_sleeve",
        "feature_count": None,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "warning": "This is a hindsight oracle using future returns. It is not tradable.",
        "config": {
            "qmt_db": str(args.qmt_db),
            "horizon": args.horizon,
            "label_mode": args.label_mode,
            "execution_lag_days": args.execution_lag_days,
            "exclude_limit_up_entry": args.exclude_limit_up_entry,
            "top_n": args.top_n,
            "sleeves": args.sleeves,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "summary": summary,
        "files": {
            "signal": _rel_path(signal_path, output_dir),
            "selected_signals": _rel_path(selected_path, output_dir),
        },
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
