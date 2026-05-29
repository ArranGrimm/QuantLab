"""Limit-up ecology diagnostic for AMV sleeves.

Stage-3 scope: use currently available daily raw OHLCV to approximate A-share
limit-up ecology features:
- recent limit-up / days since last limit-up
- limit-up streak and first-board context
- failed limit-up / board break acceptance
- pullback, reclaim, and re-board after prior limit-up

Unsupported by current data and intentionally not modeled here: seal amount,
exact seal/open time, intraday open count, auction strength, and Level2 order
book metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT
from scripts.amv_market_sentiment_diagnostic import (
    DEFAULT_SLEEVES,
    load_trade_context,
    rule_summary,
    summarize_by,
    summarize_trades,
)
from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from utils import get_st_blacklist_pl


DEFAULT_OUTPUT = ROOT / "reports" / "amv_limit_ecology_diagnostic.json"
LIMIT_TOLERANCE = 0.001


def price_limit_pct_expr() -> pl.Expr:
    code = pl.col("code")
    is_30pct_board = (
        code.str.starts_with("bj.")
        | code.str.starts_with("4")
        | code.str.starts_with("8")
        | code.str.starts_with("92")
    )
    is_20pct_board = (
        code.str.starts_with("sz.300")
        | code.str.starts_with("sz.301")
        | code.str.starts_with("sh.688")
        | code.str.starts_with("sh.689")
        | code.str.starts_with("300")
        | code.str.starts_with("301")
        | code.str.starts_with("688")
        | code.str.starts_with("689")
    )
    return pl.when(is_30pct_board).then(0.30).when(is_20pct_board).then(0.20).otherwise(0.10)


def _safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def load_raw_daily(args: argparse.Namespace) -> pl.DataFrame:
    conn = duckdb.connect(str(args.qmt_db), read_only=True)
    try:
        daily = conn.execute(
            """
            SELECT code, date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE date >= ? AND date <= ?
            ORDER BY code, date
            """,
            [args.start_date, args.end_date],
        ).pl()
    finally:
        conn.close()

    st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}, schema={"code": pl.Utf8})
    return (
        daily.with_columns(pl.col("date").cast(pl.Date))
        .join(st_blacklist_df, on="code", how="anti")
        .filter(pl.col("volume") > 0)
        .sort(["code", "date"])
    )


def add_limit_ecology_features(raw_daily: pl.DataFrame, *, tolerance: float) -> pl.DataFrame:
    stock = (
        raw_daily.sort(["code", "date"])
        .with_columns(
            [
                pl.col("close").shift(1).over("code").alias("pre_close_raw"),
                pl.int_range(pl.len()).over("code").cast(pl.Int64).alias("_row_idx"),
                price_limit_pct_expr().alias("limit_pct"),
                pl.col("amount").rolling_mean(5).over("code").alias("amount_ma5"),
                pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20"),
            ]
        )
        .with_columns(
            [
                (_safe_div(pl.col("close"), pl.col("pre_close_raw")) - 1.0).alias("ret_1d_raw"),
                (_safe_div(pl.col("open"), pl.col("pre_close_raw")) - 1.0).alias("open_gap_raw"),
                _safe_div(pl.col("amount_ma5"), pl.col("amount_ma20")).alias("amount_ratio_5_20"),
                _safe_div(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")).alias(
                    "close_position_1d"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("ret_1d_raw") >= (pl.col("limit_pct") - tolerance)).alias("is_close_limit_up"),
                (pl.col("ret_1d_raw") <= (-pl.col("limit_pct") + tolerance)).alias("is_close_limit_down"),
                (pl.col("open_gap_raw") >= (pl.col("limit_pct") - tolerance)).alias("is_open_limit_up"),
                ((_safe_div(pl.col("high"), pl.col("pre_close_raw")) - 1.0) >= (pl.col("limit_pct") - tolerance))
                .alias("touched_limit_up"),
                ((_safe_div(pl.col("low"), pl.col("pre_close_raw")) - 1.0) >= (pl.col("limit_pct") - tolerance))
                .alias("low_at_limit_up"),
            ]
        )
        .with_columns(
            [
                (pl.col("touched_limit_up") & ~pl.col("is_close_limit_up")).alias("is_failed_limit_up"),
                (
                    pl.col("is_open_limit_up")
                    & pl.col("low_at_limit_up")
                    & pl.col("is_close_limit_up")
                ).alias("is_one_word_limit_up"),
            ]
        )
        .with_columns((~pl.col("is_close_limit_up")).cum_sum().over("code").alias("_limit_break_group"))
        .with_columns(
            pl.when(pl.col("is_close_limit_up"))
            .then(pl.col("is_close_limit_up").cum_sum().over(["code", "_limit_break_group"]))
            .otherwise(0)
            .cast(pl.UInt32)
            .alias("limit_up_streak")
        )
        .with_columns(
            [
                pl.when(pl.col("is_close_limit_up"))
                .then(pl.col("_row_idx"))
                .otherwise(None)
                .forward_fill()
                .over("code")
                .alias("_last_lu_idx_incl"),
                pl.when(pl.col("is_close_limit_up"))
                .then(pl.col("close"))
                .otherwise(None)
                .forward_fill()
                .over("code")
                .alias("_last_lu_close_incl"),
                pl.when(pl.col("is_close_limit_up"))
                .then(pl.col("high"))
                .otherwise(None)
                .forward_fill()
                .over("code")
                .alias("_last_lu_high_incl"),
                pl.when(pl.col("is_close_limit_up"))
                .then(pl.col("limit_up_streak"))
                .otherwise(None)
                .forward_fill()
                .over("code")
                .alias("_last_lu_streak_incl"),
            ]
        )
        .with_columns(
            [
                pl.col("_last_lu_idx_incl").shift(1).over("code").alias("_last_lu_idx_before"),
                pl.col("_last_lu_close_incl").shift(1).over("code").alias("_last_lu_close_before"),
                pl.col("_last_lu_high_incl").shift(1).over("code").alias("_last_lu_high_before"),
                pl.col("_last_lu_streak_incl").shift(1).over("code").alias("_last_lu_streak_before"),
                pl.col("limit_up_streak").shift(1).over("code").fill_null(0).alias("prior_limit_up_streak"),
                pl.col("is_close_limit_up").shift(1).over("code").fill_null(False).alias("was_limit_up_yesterday"),
                pl.col("is_failed_limit_up").shift(1).over("code").fill_null(False).alias(
                    "was_failed_limit_up_yesterday"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("_row_idx") - pl.col("_last_lu_idx_incl")).alias("days_since_limit_up"),
                (pl.col("_row_idx") - pl.col("_last_lu_idx_before")).alias("days_since_prior_limit_up"),
                (_safe_div(pl.col("close"), pl.col("_last_lu_close_before")) - 1.0).alias(
                    "ret_since_prior_limit_up"
                ),
            ]
        )
        .with_columns(
            [
                pl.col("is_close_limit_up").cast(pl.Int16).rolling_sum(5).over("code").alias("limit_up_count_5d"),
                pl.col("is_close_limit_up").cast(pl.Int16).rolling_sum(10).over("code").alias("limit_up_count_10d"),
                pl.col("is_close_limit_up").cast(pl.Int16).rolling_sum(20).over("code").alias("limit_up_count_20d"),
                pl.col("is_failed_limit_up")
                .cast(pl.Int16)
                .rolling_sum(5)
                .over("code")
                .alias("failed_limit_up_count_5d"),
                pl.col("is_one_word_limit_up")
                .cast(pl.Int16)
                .rolling_sum(10)
                .over("code")
                .alias("one_word_limit_up_count_10d"),
            ]
        )
    )

    after_recent_limit = pl.col("days_since_prior_limit_up").is_between(1, 10)
    after_recent_first_board = after_recent_limit & (pl.col("_last_lu_streak_before") == 1)
    break_after_limit = (pl.col("prior_limit_up_streak") > 0) & ~pl.col("is_close_limit_up")
    good_break_acceptance = (
        break_after_limit
        & (pl.col("ret_1d_raw") > -0.02)
        & (pl.col("close_position_1d").fill_null(0.0) >= 0.55)
    )

    return (
        stock.with_columns(
            [
                (pl.col("limit_up_count_5d").fill_null(0) > 0).alias("has_limit_up_5d"),
                (pl.col("limit_up_count_10d").fill_null(0) > 0).alias("has_limit_up_10d"),
                (pl.col("limit_up_count_20d").fill_null(0) > 0).alias("has_limit_up_20d"),
                (pl.col("failed_limit_up_count_5d").fill_null(0) > 0).alias("has_failed_limit_up_5d"),
                (pl.col("one_word_limit_up_count_10d").fill_null(0) > 0).alias("has_one_word_limit_up_10d"),
                break_after_limit.alias("is_break_after_limit"),
                good_break_acceptance.alias("is_good_break_acceptance"),
                (
                    after_recent_first_board
                    & pl.col("ret_since_prior_limit_up").is_between(-0.10, 0.02)
                    & (pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.10)
                    & ~pl.col("is_close_limit_up")
                ).alias("is_first_board_pullback_setup"),
                (
                    after_recent_limit
                    & ~pl.col("is_close_limit_up")
                    & (pl.col("close") >= pl.col("_last_lu_high_before"))
                ).alias("is_reclaim_after_limit"),
                (
                    pl.col("is_close_limit_up")
                    & pl.col("days_since_prior_limit_up").is_between(1, 10)
                ).alias("is_reboard_after_pullback"),
            ]
        )
        .select(
            [
                "date",
                "code",
                "ret_1d_raw",
                "open_gap_raw",
                "limit_pct",
                "is_close_limit_up",
                "is_close_limit_down",
                "touched_limit_up",
                "is_failed_limit_up",
                "is_open_limit_up",
                "is_one_word_limit_up",
                "limit_up_streak",
                "prior_limit_up_streak",
                "was_limit_up_yesterday",
                "was_failed_limit_up_yesterday",
                "days_since_limit_up",
                "days_since_prior_limit_up",
                "ret_since_prior_limit_up",
                "_last_lu_streak_before",
                "amount_ratio_5_20",
                "limit_up_count_5d",
                "limit_up_count_10d",
                "limit_up_count_20d",
                "failed_limit_up_count_5d",
                "one_word_limit_up_count_10d",
                "has_limit_up_5d",
                "has_limit_up_10d",
                "has_limit_up_20d",
                "has_failed_limit_up_5d",
                "has_one_word_limit_up_10d",
                "is_break_after_limit",
                "is_good_break_acceptance",
                "is_first_board_pullback_setup",
                "is_reclaim_after_limit",
                "is_reboard_after_pullback",
            ]
        )
    )


def add_buckets(df: pl.DataFrame) -> pl.DataFrame:
    days = pl.col("days_since_prior_limit_up")
    last_streak = pl.col("_last_lu_streak_before")
    limit_count_20 = pl.col("limit_up_count_20d").fill_null(0)
    return df.with_columns(
        [
            (
                pl.when(days.is_null())
                .then(pl.lit("no_prior_limit"))
                .when(days <= 3)
                .then(pl.lit("after_lu_1_3d"))
                .when(days <= 10)
                .then(pl.lit("after_lu_4_10d"))
                .when(days <= 20)
                .then(pl.lit("after_lu_11_20d"))
                .otherwise(pl.lit("after_lu_gt20d"))
            ).alias("days_since_lu_bucket"),
            (
                pl.when(last_streak.is_null())
                .then(pl.lit("no_prior_limit"))
                .when(last_streak == 1)
                .then(pl.lit("last_first_board"))
                .when(last_streak == 2)
                .then(pl.lit("last_2_boards"))
                .otherwise(pl.lit("last_3plus_boards"))
            ).alias("last_lu_streak_bucket"),
            (
                pl.when(limit_count_20 == 0)
                .then(pl.lit("no_lu_20d"))
                .when(limit_count_20 == 1)
                .then(pl.lit("one_lu_20d"))
                .when(limit_count_20 <= 3)
                .then(pl.lit("two_three_lu_20d"))
                .otherwise(pl.lit("four_plus_lu_20d"))
            ).alias("limit_up_count_20d_bucket"),
        ]
    )


def event_summary(df: pl.DataFrame, col_name: str) -> dict[str, Any]:
    return {
        "true": summarize_trades(df.filter(pl.col(col_name).fill_null(False))),
        "false": summarize_trades(df.filter(~pl.col(col_name).fill_null(False))),
    }


def analyze_sleeve(
    sleeve_key: str,
    sleeve: dict[str, str],
    features: pl.DataFrame,
    amv_phase: pl.DataFrame,
) -> dict[str, Any]:
    trades = load_trade_context(ROOT / sleeve["trades"], ROOT / sleeve["signals"])
    enriched = (
        trades.join(features, left_on=["signal_date", "code"], right_on=["date", "code"], how="left")
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
        .pipe(add_buckets)
    )

    rules = [
        (
            "skip_no_recent_limit_up_20d",
            ~pl.col("has_limit_up_20d").fill_null(False),
        ),
        (
            "skip_recent_failed_limit_up_5d",
            pl.col("has_failed_limit_up_5d").fill_null(False),
        ),
        (
            "skip_after_3plus_limit_streak",
            pl.col("_last_lu_streak_before").fill_null(0) >= 3,
        ),
        (
            "skip_weak_break_after_limit",
            pl.col("is_break_after_limit").fill_null(False)
            & ~pl.col("is_good_break_acceptance").fill_null(False),
        ),
        (
            "skip_not_first_board_pullback_setup",
            ~pl.col("is_first_board_pullback_setup").fill_null(False),
        ),
        (
            "skip_not_reclaim_or_reboard",
            ~(
                pl.col("is_reclaim_after_limit").fill_null(False)
                | pl.col("is_reboard_after_pullback").fill_null(False)
            ),
        ),
    ]
    event_cols = [
        "has_limit_up_5d",
        "has_limit_up_10d",
        "has_limit_up_20d",
        "has_failed_limit_up_5d",
        "has_one_word_limit_up_10d",
        "was_limit_up_yesterday",
        "was_failed_limit_up_yesterday",
        "is_break_after_limit",
        "is_good_break_acceptance",
        "is_first_board_pullback_setup",
        "is_reclaim_after_limit",
        "is_reboard_after_pullback",
    ]
    return {
        "sleeve": sleeve_key,
        "label": sleeve["label"],
        "total": summarize_trades(enriched),
        "missing_feature_trades": int(enriched.filter(pl.col("limit_up_count_20d").is_null()).height),
        "by_days_since_lu_bucket": summarize_by(enriched, "days_since_lu_bucket"),
        "by_last_lu_streak_bucket": summarize_by(enriched, "last_lu_streak_bucket"),
        "by_limit_up_count_20d_bucket": summarize_by(enriched, "limit_up_count_20d_bucket"),
        "event_summary": {col: event_summary(enriched, col) for col in event_cols},
        "rules": [rule_summary(enriched, expr, name) for name, expr in rules],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV limit-up ecology diagnostic")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    args = parser.parse_args()

    raw_daily = load_raw_daily(args)
    features = add_limit_ecology_features(raw_daily, tolerance=args.price_limit_tolerance)
    logger.info(
        f"Built limit ecology features: {features.height:,} rows, "
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
            "Daily raw OHLCV limit ecology features joined on signal_date + code. "
            "pre_close_raw is approximated by previous trading day's raw close because stock_daily has no official pre_close."
        ),
        "data": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_rows": features.height,
            "price_limit_tolerance": args.price_limit_tolerance,
            "st_snapshot_date": args.st_snapshot_date,
        },
        "supported_features": [
            "near-N-day close limit-up",
            "days since previous close limit-up",
            "close limit-up streak",
            "failed limit-up approximated by intraday high touching limit but close not limit-up",
            "one-word limit-up approximated by open/low/close all at limit-up",
            "first-board pullback setup",
            "reclaim after prior limit-up",
            "re-board after pullback",
            "break-after-limit acceptance",
        ],
        "unsupported_features": [
            "exact seal time",
            "seal amount",
            "number of intraday board opens",
            "auction strength",
            "Level2 order book strength",
            "official ex-right adjusted price-limit reference price",
        ],
        "feature_definitions": {
            "is_close_limit_up": "raw close / previous raw close - 1 >= board limit pct - tolerance",
            "is_failed_limit_up": "raw high touched limit-up but raw close did not close limit-up",
            "limit_up_streak": "consecutive close limit-up count by code",
            "days_since_prior_limit_up": "trading days since the previous close limit-up before signal_date",
            "is_first_board_pullback_setup": "after a prior first board within 10 trading days, price pulls back mildly with non-expanded 5/20 amount",
            "is_reclaim_after_limit": "after a prior limit-up, raw close reclaims the prior limit-up day's high",
            "is_reboard_after_pullback": "close limit-up again within 10 trading days after a prior limit-up",
            "is_good_break_acceptance": "after prior limit-up streak breaks, daily return > -2% and close position >= 0.55",
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
