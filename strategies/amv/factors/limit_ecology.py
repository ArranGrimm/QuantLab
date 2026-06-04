from __future__ import annotations

import argparse

import polars as pl

from utils import get_st_blacklist_pl
from utils.data_source import daily_reader


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


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def load_raw_daily(args: argparse.Namespace) -> pl.DataFrame:
    with daily_reader(args.data_source) as reader:
        daily = reader.load_raw_ohlc(args.start_date, args.end_date)

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
                (safe_div(pl.col("close"), pl.col("pre_close_raw")) - 1.0).alias("ret_1d_raw"),
                (safe_div(pl.col("open"), pl.col("pre_close_raw")) - 1.0).alias("open_gap_raw"),
                safe_div(pl.col("amount_ma5"), pl.col("amount_ma20")).alias("amount_ratio_5_20"),
                safe_div(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")).alias(
                    "close_position_1d"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("ret_1d_raw") >= (pl.col("limit_pct") - tolerance)).alias("is_close_limit_up"),
                (pl.col("ret_1d_raw") <= (-pl.col("limit_pct") + tolerance)).alias("is_close_limit_down"),
                (pl.col("open_gap_raw") >= (pl.col("limit_pct") - tolerance)).alias("is_open_limit_up"),
                ((safe_div(pl.col("high"), pl.col("pre_close_raw")) - 1.0) >= (pl.col("limit_pct") - tolerance))
                .alias("touched_limit_up"),
                ((safe_div(pl.col("low"), pl.col("pre_close_raw")) - 1.0) >= (pl.col("limit_pct") - tolerance))
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
                (safe_div(pl.col("close"), pl.col("_last_lu_close_before")) - 1.0).alias(
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
