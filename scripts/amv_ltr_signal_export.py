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


DEFAULT_LTR_RUN = ROOT / "artifacts/amv_bull_pool_listwise_ranker/20260516_112948"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/amv_ltr_signals"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("variants must not be empty")
    return variants


def load_ltr_selected(args: argparse.Namespace) -> pl.DataFrame:
    selected_path = args.ltr_run / "ltr_topn_selected.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"missing LTR selected file: {selected_path}")

    df = (
        pl.read_csv(selected_path, try_parse_dates=True)
        .filter(pl.col("feature_variant").is_in(args.variants))
        .with_columns(
            [
                pl.col("date").cast(pl.Date).alias("signal_date"),
                pl.col("year").cast(pl.Int64),
                pl.col("ltr_score").cast(pl.Float64),
            ]
        )
    )
    if df.is_empty():
        raise ValueError(f"no LTR selected rows for variants: {args.variants}")
    return df


def apply_repeat_limit(df: pl.DataFrame, max_repeats: int | None) -> pl.DataFrame:
    if max_repeats is None or max_repeats <= 0:
        return df
    return (
        df.sort(["feature_variant", "year", "code", "signal_date"])
        .with_columns(
            pl.col("signal_date")
            .rank(method="ordinal")
            .over(["feature_variant", "year", "code"])
            .alias("_code_pick_order")
        )
        .filter(pl.col("_code_pick_order") <= max_repeats)
        .drop("_code_pick_order")
    )


def combine_variant_signals(df: pl.DataFrame, mode: str, top_n: int) -> pl.DataFrame:
    if mode == "separate":
        return (
            df.sort(["feature_variant", "signal_date", "ltr_score", "code"], descending=[False, False, True, False])
            .with_columns(
                pl.col("ltr_score")
                .rank(method="ordinal", descending=True)
                .over(["feature_variant", "signal_date"])
                .alias("rank")
            )
            .filter(pl.col("rank") <= top_n)
            .rename({"ltr_score": "score"})
        )

    if mode == "union":
        grouped = (
            df.group_by(["signal_date", "code"])
            .agg(
                [
                    pl.col("feature_variant").sort().str.join("+").alias("feature_variant"),
                    pl.col("ltr_score").max().alias("score"),
                    pl.col("fwd_ret_6d").first().alias("fwd_ret_6d"),
                    pl.col("fwd_mfe_6d").first().alias("fwd_mfe_6d"),
                    pl.col("fwd_mae_6d").first().alias("fwd_mae_6d"),
                ]
            )
            .sort(["signal_date", "score", "code"], descending=[False, True, False])
            .with_columns(
                pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("signal_date")
                .alias("rank")
            )
        )
        return grouped.filter(pl.col("rank") <= top_n)

    if mode == "intersection":
        required = len(df["feature_variant"].unique())
        grouped = (
            df.group_by(["signal_date", "code"])
            .agg(
                [
                    pl.col("feature_variant").n_unique().alias("_variant_count"),
                    pl.col("feature_variant").sort().str.join("+").alias("feature_variant"),
                    pl.col("ltr_score").mean().alias("score"),
                    pl.col("fwd_ret_6d").first().alias("fwd_ret_6d"),
                    pl.col("fwd_mfe_6d").first().alias("fwd_mfe_6d"),
                    pl.col("fwd_mae_6d").first().alias("fwd_mae_6d"),
                ]
            )
            .filter(pl.col("_variant_count") == required)
            .sort(["signal_date", "score", "code"], descending=[False, True, False])
            .with_columns(
                pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("signal_date")
                .alias("rank")
            )
        )
        return grouped.filter(pl.col("rank") <= top_n).drop("_variant_count")

    raise ValueError(f"unknown combine mode: {mode}")


def summarize_selected(selected: pl.DataFrame) -> dict[str, Any]:
    if selected.is_empty():
        return {}
    yearly = (
        selected.group_by(["feature_variant", "year"])
        .agg(
            [
                pl.len().alias("selected_rows"),
                pl.col("signal_date").n_unique().alias("days"),
                pl.col("code").n_unique().alias("unique_codes"),
                pl.col("fwd_ret_6d").mean().alias("mean_ret"),
                pl.col("fwd_ret_6d").median().alias("median_ret"),
                (pl.col("fwd_ret_6d") > 0).mean().alias("win_rate"),
                (pl.col("fwd_mfe_6d") >= 0.15).mean().alias("hit15"),
                pl.col("fwd_mfe_6d").mean().alias("mean_mfe"),
                pl.col("fwd_mae_6d").mean().alias("mean_mae"),
            ]
        )
        .sort(["feature_variant", "year"])
    )
    overall = (
        yearly.group_by("feature_variant")
        .agg(
            [
                pl.col("selected_rows").sum().alias("selected_rows"),
                pl.col("days").sum().alias("days"),
                pl.col("mean_ret").mean().alias("avg_year_mean_ret"),
                pl.col("win_rate").mean().alias("avg_year_win_rate"),
                pl.col("hit15").mean().alias("avg_year_hit15"),
                pl.col("mean_mfe").mean().alias("avg_year_mfe"),
                pl.col("mean_mae").mean().alias("avg_year_mae"),
            ]
        )
        .sort("avg_year_mean_ret", descending=True)
    )
    return {
        "overall": overall.to_dicts(),
        "yearly": yearly.to_dicts(),
    }


def build_signal_export(args: argparse.Namespace) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    if args.combine_mode == "separate" and len(args.variants) != 1:
        raise ValueError("--combine-mode separate requires exactly one variant")

    market = build_feature_frame(args)
    selected = apply_repeat_limit(load_ltr_selected(args), args.max_code_repeats)
    combined = combine_variant_signals(selected, args.combine_mode, args.top_n).with_columns(
        pl.col("rank").cast(pl.UInt32)
    )

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(
        pl.col("date").shift(-1).alias("execution_date")
    ).drop_nulls("execution_date")
    execution_signals = (
        combined.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(
            [
                pl.col("execution_date").alias("date"),
                "code",
                "signal_date",
                "feature_variant",
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
                pl.col("feature_variant").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )

    signal_count = int(export["is_signal"].sum())
    signal_days = export.filter(pl.col("is_signal")).select("date").n_unique()
    blocked_by_execution_regime = int(
        export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
    )
    summary = {
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": combined.height,
        "signal_rows_after_shift": signal_count,
        "signal_days_after_shift": signal_days,
        "signals_blocked_by_execution_bear_regime": blocked_by_execution_regime,
        "label_summary_before_shift": summarize_selected(combined),
    }
    return export, combined, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMV LTR OOS signals for bt-amv-topn")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--ltr-run", type=Path, default=DEFAULT_LTR_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=["kbar_momentum_old_state"],
        help="Comma-separated LTR feature variants.",
    )
    parser.add_argument("--combine-mode", choices=["separate", "union", "intersection"], default="separate")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-code-repeats", type=int, default=3)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    variant_token = "+".join(args.variants)
    output_dir = args.output_root / f"{timestamp_token()}_{args.combine_mode}_{variant_token}"
    signal_path = output_dir / "signal.parquet"
    selected_path = output_dir / "selected_signals.csv"
    meta_path = output_dir / "signal.meta.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building AMV LTR signal export...")
    export, selected, summary = build_signal_export(args)
    export.write_parquet(signal_path)
    selected.write_csv(selected_path)

    meta = {
        "strategy": "amv_ltr_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_ltr_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"{args.combine_mode}:{variant_token}",
        "model_name": "lightgbm_lambdarank",
        "feature_mode": variant_token,
        "feature_count": None,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "ltr_run": str(args.ltr_run),
            "variants": args.variants,
            "combine_mode": args.combine_mode,
            "top_n": args.top_n,
            "max_code_repeats": args.max_code_repeats,
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
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Saved signal:   {signal_path}")
    print(f"Saved selected: {selected_path}")
    print(f"Saved meta:     {meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Relative signal: {_rel_path(signal_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
