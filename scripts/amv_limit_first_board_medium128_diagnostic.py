"""Diagnose whether medium128 helps first-board pullback drawdown control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT
from scripts.amv_limit_ecology_drawdown_attribution import (
    BASE_SIGNAL,
    add_feature_buckets,
    feature_args,
    load_enriched_trades,
    load_feature_lookup,
    summarize_trades,
    top_records,
)
from scripts.amv_limit_ecology_signal_export import build_limit_ecology_market
from scripts.amv_medium_trend_quality_diagnostic import add_medium_trend_features


DEFAULT_OUTPUT = ROOT / "reports" / "amv_limit_first_board_medium128_diagnostic.json"
BASE_BACKTEST = "5td_static_strict_top3_no_stop_20260529_111655_limit_first_board_pullback"


def bucket_summary(df: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for row in (
        df.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
        .sort(column)
        .iter_rows(named=True)
    ):
        rows.append(
            {
                column: str(row[column]),
                "trades": int(row["trades"]),
                "pnl": round(float(row["pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
            }
        )
    return rows


def rule_summary(df: pl.DataFrame, name: str, expr: pl.Expr) -> dict[str, Any]:
    skipped = df.filter(expr)
    kept = df.filter(~expr)
    return {
        "rule": name,
        "skipped": summarize_trades(skipped),
        "kept": summarize_trades(kept),
        "trade_level_delta_if_skipped": round(-float(skipped["pnl"].sum()), 2) if skipped.height else 0.0,
        "worst_skipped_trades": top_records(skipped, by="pnl", descending=False, n=8),
    }


def add_medium_buckets(df: pl.DataFrame) -> pl.DataFrame:
    structure = pl.col("structure_score_128d").fill_null(0.5)
    quality = pl.col("trend_quality_score_128d").fill_null(0.5)
    atr = pl.col("atr_14_pct_rank_pct").fill_null(0.5)
    days = pl.col("days_since_prior_limit_up").fill_null(99)
    return df.with_columns(
        [
            (
                pl.when(structure < 0.33)
                .then(pl.lit("structure_low"))
                .when(structure < 0.67)
                .then(pl.lit("structure_mid"))
                .otherwise(pl.lit("structure_high"))
            ).alias("structure128_bucket"),
            (
                pl.when(quality < 0.33)
                .then(pl.lit("quality_low"))
                .when(quality < 0.67)
                .then(pl.lit("quality_mid"))
                .otherwise(pl.lit("quality_high"))
            ).alias("quality128_bucket"),
            (
                pl.when((structure < 0.5) & (quality < 0.5))
                .then(pl.lit("medium_weak"))
                .otherwise(pl.lit("medium_ok"))
            ).alias("medium128_weak_bucket"),
            (
                pl.when(atr > 0.9)
                .then(pl.lit("atr_gt_0p90"))
                .when(atr > 0.85)
                .then(pl.lit("atr_0p85_0p90"))
                .when(atr > 0.8)
                .then(pl.lit("atr_0p80_0p85"))
                .otherwise(pl.lit("atr_le_0p80"))
            ).alias("atr_rank_bucket"),
            (
                pl.when(days >= 7)
                .then(pl.lit("stale_7_10d"))
                .when(days >= 4)
                .then(pl.lit("mid_4_6d"))
                .otherwise(pl.lit("fresh_1_3d"))
            ).alias("limit_age_bucket"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose medium128 on first-board pullback 5td trades")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_args = feature_args()
    market = build_limit_ecology_market(base_args)
    medium = (
        add_medium_trend_features(market)
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                "structure_score_128d",
                "trend_quality_score_128d",
                "ret_128d",
                "pos_128d",
                "trend_eff_128d",
                "ret_vol_128d",
                "structure_score_64d",
                "trend_quality_score_64d",
            ]
        )
        .unique(["signal_date", "code"])
    )
    trades = (
        add_feature_buckets(load_enriched_trades(BASE_SIGNAL, BASE_BACKTEST, load_feature_lookup()))
        .join(medium, on=["signal_date", "code"], how="left")
        .pipe(add_medium_buckets)
    )
    drawdown_trades = trades.filter((pl.col("exit_date") >= pl.date(2023, 8, 3)) & (pl.col("exit_date") <= pl.date(2024, 2, 5)))

    rules = [
        (
            "skip_medium128_weak_t0p50",
            (pl.col("structure_score_128d").fill_null(1.0) < 0.50)
            & (pl.col("trend_quality_score_128d").fill_null(1.0) < 0.50),
        ),
        (
            "skip_quality128_low_t0p33",
            pl.col("trend_quality_score_128d").fill_null(1.0) < 0.33,
        ),
        (
            "skip_structure128_low_t0p33",
            pl.col("structure_score_128d").fill_null(1.0) < 0.33,
        ),
        ("skip_atr_gt_0p90", pl.col("atr_14_pct_rank_pct").fill_null(0.0) > 0.90),
        ("skip_atr_gt_0p85", pl.col("atr_14_pct_rank_pct").fill_null(0.0) > 0.85),
        (
            "skip_stale_7d_and_quality128_not_high",
            (pl.col("days_since_prior_limit_up").fill_null(0) >= 7)
            & (pl.col("trend_quality_score_128d").fill_null(0.0) < 0.67),
        ),
        (
            "skip_atr_gt_0p85_and_medium128_not_high",
            (pl.col("atr_14_pct_rank_pct").fill_null(0.0) > 0.85)
            & (
                (pl.col("structure_score_128d").fill_null(0.0) < 0.67)
                | (pl.col("trend_quality_score_128d").fill_null(0.0) < 0.67)
            ),
        ),
    ]

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "Trade-level diagnostic for medium128 on limit_first_board_pullback 5td",
        "source_backtest": str(BASE_SIGNAL.relative_to(ROOT) / "backtests" / BASE_BACKTEST),
        "total": summarize_trades(trades),
        "drawdown_window": {
            "range": "2023-08-03..2024-02-05 by exit_date",
            "summary": summarize_trades(drawdown_trades),
        },
        "by_structure128_bucket": bucket_summary(trades, "structure128_bucket"),
        "by_quality128_bucket": bucket_summary(trades, "quality128_bucket"),
        "by_medium128_weak_bucket": bucket_summary(trades, "medium128_weak_bucket"),
        "by_atr_rank_bucket": bucket_summary(trades, "atr_rank_bucket"),
        "by_limit_age_bucket": bucket_summary(trades, "limit_age_bucket"),
        "drawdown_by_structure128_bucket": bucket_summary(drawdown_trades, "structure128_bucket"),
        "drawdown_by_quality128_bucket": bucket_summary(drawdown_trades, "quality128_bucket"),
        "drawdown_by_medium128_weak_bucket": bucket_summary(drawdown_trades, "medium128_weak_bucket"),
        "rules": [rule_summary(trades, name, expr) for name, expr in rules],
        "drawdown_rules": [rule_summary(drawdown_trades, name, expr) for name, expr in rules],
        "interpretation": {
            "medium128": "Useful only if weak/old/high-vol interactions isolate losers without deleting the broad first-board pullback edge.",
            "next_step": "Promote only the positive trade-level rules to 5td Rust signal variants.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
