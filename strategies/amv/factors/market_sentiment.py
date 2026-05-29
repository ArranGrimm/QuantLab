from __future__ import annotations

import argparse

import polars as pl


DEFAULT_PRICE_LIMIT_TOLERANCE = 0.001


def price_limit_pct_expr() -> pl.Expr:
    is_20pct_board = (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sz.301")
        | pl.col("code").str.starts_with("sh.688")
        | pl.col("code").str.starts_with("sh.689")
        | pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )
    return pl.when(is_20pct_board).then(0.20).otherwise(0.10)


def add_market_sentiment_features(
    market: pl.DataFrame,
    *,
    price_limit_tolerance: float = DEFAULT_PRICE_LIMIT_TOLERANCE,
) -> pl.DataFrame:
    stock = (
        market.sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                (pl.col("open_adj") / pl.col("pre_close_adj") - 1.0).alias("open_gap"),
                price_limit_pct_expr().alias("limit_pct"),
                pl.col("close_adj").rolling_max(20).over("code").alias("high_close_20d"),
            ]
        )
        .with_columns(
            [
                (pl.col("ret_1d") >= (pl.col("limit_pct") - price_limit_tolerance)).alias("is_close_limit_up"),
                (pl.col("ret_1d") <= (-pl.col("limit_pct") + price_limit_tolerance)).alias("is_close_limit_down"),
                (
                    (pl.col("high_adj") / pl.col("pre_close_adj") - 1.0)
                    >= (pl.col("limit_pct") - price_limit_tolerance)
                ).alias("touched_limit_up"),
                (pl.col("close_adj") >= pl.col("high_close_20d")).alias("is_new_high_20"),
                (pl.col("close_adj") / pl.col("high_close_20d") - 1.0).alias("drawdown_from_20d_high"),
                (pl.col("ret_1d") > 0.0).alias("is_up_day"),
                (pl.col("ret_20d").rank("average").over("date") / pl.len().over("date")).alias("ret_20d_rank_pct"),
            ]
        )
        .with_columns(
            [
                (pl.col("touched_limit_up") & ~pl.col("is_close_limit_up")).alias("is_failed_limit_up"),
                (pl.col("ret_20d_rank_pct") >= 0.80).alias("is_strong_20d_stock"),
                pl.col("is_close_limit_up").shift(1).over("code").fill_null(False).alias("was_limit_up_yesterday"),
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
    )

    daily = (
        stock.group_by("date")
        .agg(
            [
                pl.len().alias("market_stock_count"),
                pl.col("is_up_day").sum().alias("up_count"),
                pl.col("is_up_day").mean().alias("market_up_ratio"),
                pl.col("is_close_limit_up").sum().alias("limit_up_count"),
                pl.col("is_close_limit_down").sum().alias("limit_down_count"),
                pl.col("touched_limit_up").sum().alias("touched_limit_up_count"),
                pl.col("is_failed_limit_up").sum().alias("failed_limit_up_count"),
                pl.col("is_new_high_20").sum().alias("new_high_20_count"),
                pl.col("is_new_high_20").mean().alias("new_high_20_ratio"),
                pl.col("limit_up_streak").max().alias("limit_up_streak_max"),
                pl.col("drawdown_from_20d_high")
                .filter(pl.col("is_strong_20d_stock"))
                .median()
                .alias("strong_stock_drawdown_20d_median"),
                pl.col("open_gap")
                .filter(pl.col("was_limit_up_yesterday"))
                .mean()
                .alias("yday_limit_up_open_premium"),
                pl.col("ret_1d")
                .filter(pl.col("was_limit_up_yesterday"))
                .mean()
                .alias("yday_limit_up_close_premium"),
                pl.col("was_limit_up_yesterday").sum().alias("yday_limit_up_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("limit_up_count") - pl.col("limit_down_count")).alias("limit_up_minus_down"),
                (
                    pl.col("failed_limit_up_count")
                    / pl.when(pl.col("touched_limit_up_count") > 0).then(pl.col("touched_limit_up_count")).otherwise(1)
                ).alias("failed_limit_up_ratio"),
            ]
        )
        .sort("date")
    )

    rank_cols = [
        "market_up_ratio",
        "limit_up_count",
        "limit_down_count",
        "limit_up_minus_down",
        "failed_limit_up_ratio",
        "new_high_20_ratio",
        "strong_stock_drawdown_20d_median",
        "yday_limit_up_close_premium",
    ]
    return daily.with_columns(
        [
            (pl.col(col).rank("average") / pl.len()).alias(f"{col}_rank_pct")
            for col in rank_cols
        ]
    )


def build_market_sentiment_features(args: argparse.Namespace) -> pl.DataFrame:
    from strategies.amv.market import build_market_frame

    return add_market_sentiment_features(
        build_market_frame(args),
        price_limit_tolerance=args.price_limit_tolerance,
    )
