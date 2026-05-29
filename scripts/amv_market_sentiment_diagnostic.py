"""Market sentiment diagnostic for AMV P3/PB3 trades.

Builds signal-date, whole-market A-share context features such as limit-up
counts, failed limit-ups, new-high breadth, and yesterday limit-up premium.
This is a diagnostic step only: it does not export new trading signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT
from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from strategies.amv.factors.market_sentiment import (
    DEFAULT_PRICE_LIMIT_TOLERANCE,
    build_market_sentiment_features,
)


DEFAULT_OUTPUT = ROOT / "reports" / "amv_market_sentiment_diagnostic.json"
LIMIT_TOLERANCE = DEFAULT_PRICE_LIMIT_TOLERANCE

DEFAULT_SLEEVES: dict[str, dict[str, str]] = {
    "p3_static": {
        "label": "P3 static strict",
        "trades": "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092208_801/trades.csv",
        "signals": "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/signal.parquet",
    },
    "pb3_rolling": {
        "label": "PB3 rolling raw",
        "trades": "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/backtests/6td_rolling21_refill_top10_no_stop_20260521_091007_830/trades.csv",
        "signals": "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/signal.parquet",
    },
}


def load_trade_context(trades_path: Path, signals_path: Path) -> pl.DataFrame:
    trades = pl.read_csv(trades_path, try_parse_dates=True).with_row_index("trade_id")
    signals = (
        pl.scan_parquet(signals_path)
        .filter(pl.col("is_signal"))
        .select(
            [
                pl.col("date").alias("entry_date"),
                "code",
                "signal_date",
                "score",
                "rank",
            ]
        )
        .collect()
    )
    joined = trades.join(signals, on=["entry_date", "code"], how="left")
    missing = joined.filter(pl.col("signal_date").is_null()).height
    if missing:
        logger.warning(f"{missing:,}/{joined.height:,} trades could not be matched to signal rows")
    return joined


def summarize_trades(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl_pct": 0.0,
            "win_rate": 0.0,
        }
    return {
        "trades": int(df.height),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 6),
        "win_rate": round(float((df["pnl"] > 0).mean()), 6),
    }


def summarize_by(df: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for row in (
        df.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
        .sort(column)
        .iter_rows(named=True)
    ):
        rows.append(
            {
                column: row[column],
                "trades": int(row["trades"]),
                "total_pnl": round(float(row["total_pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
            }
        )
    return rows


def rule_summary(df: pl.DataFrame, rule: pl.Expr, name: str) -> dict[str, Any]:
    skipped = df.filter(rule)
    kept = df.filter(~rule)
    skipped_pnl = float(skipped["pnl"].sum()) if skipped.height else 0.0
    yearly = (
        skipped.with_columns(pl.col("entry_date").dt.year().alias("year"))
        .group_by("year")
        .agg((-pl.col("pnl").sum()).alias("trade_level_delta"))
        .sort("year")
        if skipped.height
        else pl.DataFrame({"year": [], "trade_level_delta": []})
    )
    return {
        "rule": name,
        "skipped": summarize_trades(skipped),
        "kept": summarize_trades(kept),
        "trade_level_delta": round(-skipped_pnl, 2),
        "skipped_big_winner_gt_20k": int(skipped.filter(pl.col("pnl") > 20_000).height),
        "skipped_big_loser_lt_minus_20k": int(skipped.filter(pl.col("pnl") < -20_000).height),
        "yearly_trade_level_delta": {
            str(row["year"]): round(float(row["trade_level_delta"]), 2) for row in yearly.iter_rows(named=True)
        },
    }


def add_sentiment_buckets(df: pl.DataFrame) -> pl.DataFrame:
    def rank_bucket(col_name: str, label: str, *, high_is_good: bool) -> pl.Expr:
        low_label = f"{label}_low"
        mid_label = f"{label}_mid"
        high_label = f"{label}_high"
        rank = pl.col(col_name).fill_null(0.5)
        if high_is_good:
            return (
                pl.when(rank < 0.33)
                .then(pl.lit(low_label))
                .when(rank < 0.67)
                .then(pl.lit(mid_label))
                .otherwise(pl.lit(high_label))
            )
        return (
            pl.when(rank < 0.33)
            .then(pl.lit(high_label))
            .when(rank < 0.67)
            .then(pl.lit(mid_label))
            .otherwise(pl.lit(low_label))
        )

    return df.with_columns(
        [
            rank_bucket("limit_up_count_rank_pct", "limit_up_count", high_is_good=True).alias(
                "limit_up_count_bucket"
            ),
            rank_bucket("limit_down_count_rank_pct", "limit_down_count", high_is_good=False).alias(
                "limit_down_count_bucket"
            ),
            rank_bucket("failed_limit_up_ratio_rank_pct", "failed_limit_up", high_is_good=False).alias(
                "failed_limit_up_bucket"
            ),
            rank_bucket("new_high_20_ratio_rank_pct", "new_high_20", high_is_good=True).alias(
                "new_high_20_bucket"
            ),
            rank_bucket("market_up_ratio_rank_pct", "market_up_ratio", high_is_good=True).alias(
                "market_up_ratio_bucket"
            ),
            rank_bucket("yday_limit_up_close_premium_rank_pct", "yday_lu_premium", high_is_good=True).alias(
                "yday_lu_premium_bucket"
            ),
            rank_bucket("strong_stock_drawdown_20d_median_rank_pct", "strong_drawdown", high_is_good=True).alias(
                "strong_drawdown_bucket"
            ),
        ]
    )


def analyze_sleeve(
    sleeve_key: str,
    sleeve: dict[str, str],
    features: pl.DataFrame,
    amv_phase: pl.DataFrame,
) -> dict[str, Any]:
    trades = load_trade_context(ROOT / sleeve["trades"], ROOT / sleeve["signals"])
    enriched = (
        trades.join(features, left_on="signal_date", right_on="date", how="left")
        .join(
            amv_phase.select(
                [
                    pl.col("date").alias("signal_date"),
                    "fwd_duration_bucket",
                    "fwd_momentum_bucket",
                    "fwd_phase",
                    "amv_neg_streak",
                    "amplitude_pct",
                ]
            ),
            on="signal_date",
            how="left",
        )
        .pipe(add_sentiment_buckets)
    )

    rules = [
        (
            "skip_low_limit_up_count",
            pl.col("limit_up_count_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_low_new_high_20",
            pl.col("new_high_20_ratio_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_cold_market_breadth",
            pl.col("market_up_ratio_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_high_limit_down_count",
            pl.col("limit_down_count_rank_pct").fill_null(0.0) > 0.67,
        ),
        (
            "skip_high_failed_limit_up",
            pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.0) > 0.67,
        ),
        (
            "skip_poor_yday_limit_up_premium",
            pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_hot_yday_limit_up_premium",
            pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.0) > 0.67,
        ),
        (
            "skip_hot_new_high_20",
            pl.col("new_high_20_ratio_rank_pct").fill_null(0.0) > 0.67,
        ),
        (
            "skip_hot_limit_up_count",
            pl.col("limit_up_count_rank_pct").fill_null(0.0) > 0.67,
        ),
        (
            "skip_hot_yday_premium_and_new_high",
            (pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.0) > 0.67)
            & (pl.col("new_high_20_ratio_rank_pct").fill_null(0.0) > 0.67),
        ),
        (
            "skip_cold_limit_and_new_high",
            (pl.col("limit_up_count_rank_pct").fill_null(0.0) < 0.33)
            & (pl.col("new_high_20_ratio_rank_pct").fill_null(0.0) < 0.33),
        ),
        (
            "skip_bad_board_and_failed_limit",
            (pl.col("limit_down_count_rank_pct").fill_null(0.0) > 0.67)
            & (pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.0) > 0.67),
        ),
    ]

    return {
        "sleeve": sleeve_key,
        "label": sleeve["label"],
        "total": summarize_trades(enriched),
        "missing_feature_trades": int(enriched.filter(pl.col("market_stock_count").is_null()).height),
        "by_limit_up_count_bucket": summarize_by(enriched, "limit_up_count_bucket"),
        "by_limit_down_count_bucket": summarize_by(enriched, "limit_down_count_bucket"),
        "by_failed_limit_up_bucket": summarize_by(enriched, "failed_limit_up_bucket"),
        "by_new_high_20_bucket": summarize_by(enriched, "new_high_20_bucket"),
        "by_market_up_ratio_bucket": summarize_by(enriched, "market_up_ratio_bucket"),
        "by_yday_lu_premium_bucket": summarize_by(enriched, "yday_lu_premium_bucket"),
        "by_strong_drawdown_bucket": summarize_by(enriched, "strong_drawdown_bucket"),
        "rules": [rule_summary(enriched, expr, name) for name, expr in rules],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV market sentiment diagnostic")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=0.04)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-0.023)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    features = build_market_sentiment_features(args)
    logger.info(
        f"Built sentiment features: {features.height:,} dates, "
        f"{features['date'].min()} ~ {features['date'].max()}"
    )
    amv_phase = build_amv_phase_frame()
    sleeves = {
        key: analyze_sleeve(key, sleeve, features, amv_phase)
        for key, sleeve in DEFAULT_SLEEVES.items()
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": (
            "Whole-market non-ST QMT adjusted daily bars. Features are joined on signal_date, "
            "so they are observable before T+1 open execution."
        ),
        "data": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_dates": features.height,
        },
        "feature_definitions": {
            "limit_up_count": "Daily count of close limit-up stocks by board-specific 10%/20% threshold.",
            "limit_down_count": "Daily count of close limit-down stocks by board-specific threshold.",
            "failed_limit_up_ratio": "Touched limit-up intraday but did not close limit-up, divided by touched limit-up count.",
            "new_high_20_ratio": "Share of stocks closing at their 20-day close high.",
            "yday_limit_up_close_premium": "Same-day close return of stocks that closed limit-up on the previous trading day.",
            "strong_stock_drawdown_20d_median": "Median drawdown from 20-day high among stocks in top 20% 20-day return.",
        },
        "sleeves": sleeves,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote {args.output}")
    for sleeve in sleeves.values():
        best_rule = max(sleeve["rules"], key=lambda item: item["trade_level_delta"])
        logger.info(
            f"{sleeve['label']}: total={sleeve['total']['total_pnl']:+,.0f}, "
            f"best_rule={best_rule['rule']} delta={best_rule['trade_level_delta']:+,.0f}, "
            f"skipped={best_rule['skipped']['trades']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
