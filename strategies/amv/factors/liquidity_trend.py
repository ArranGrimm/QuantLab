from __future__ import annotations

import argparse

import polars as pl

from strategies.amv.factors.medium_trend_quality import safe_div


def finite_or_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def rank_pct(col_name: str, *, higher_is_better: bool = True) -> pl.Expr:
    return (
        pl.col(col_name).rank("average", descending=not higher_is_better).over("date")
        / pl.len().over("date")
    ).alias(f"{col_name}_rank_pct")


def add_liquidity_trend_features(market: pl.DataFrame) -> pl.DataFrame:
    stock = (
        market.sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                pl.int_range(pl.len()).over("code").cast(pl.Float64).alias("time_idx"),
            ]
        )
        .with_columns(
            [
                pl.col("close_adj").rolling_mean(20).over("code").alias("ma20"),
                pl.col("close_adj").rolling_mean(60).over("code").alias("ma60"),
                pl.col("close_adj").rolling_mean(120).over("code").alias("ma120"),
                pl.col("amount").rolling_mean(5).over("code").alias("amount_ma5"),
                pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20_raw"),
                pl.col("amount").rolling_mean(60).over("code").alias("amount_ma60"),
            ]
        )
        .with_columns(
            [
                safe_div(pl.col("amount"), pl.col("amount_ma20_raw")).alias("amount_ratio_1_20"),
                safe_div(pl.col("amount_ma5"), pl.col("amount_ma20_raw")).alias("amount_ratio_5_20"),
                safe_div(pl.col("amount_ma20_raw"), pl.col("amount_ma60")).alias("amount_ratio_20_60"),
                ((pl.col("ma20") > pl.col("ma60")) & (pl.col("ma60") > pl.col("ma120"))).alias(
                    "ma_stack_bullish"
                ),
            ]
        )
        .with_columns(
            pl.col("ma_stack_bullish").cast(pl.Float64).rolling_mean(20).over("code").alias(
                "ma_stack_stability_20d"
            )
        )
    )

    trend_exprs: list[pl.Expr] = []
    for window in (64, 128):
        high = pl.col("close_adj").rolling_max(window).over("code")
        low = pl.col("close_adj").rolling_min(window).over("code")
        drawdown_from_high = pl.col("close_adj") / high - 1.0
        range_pos = safe_div(pl.col("close_adj") - low, high - low)
        corr = pl.rolling_corr(
            pl.col("time_idx"),
            pl.col("close_adj"),
            window_size=window,
            min_samples=window,
        ).over("code")
        trend_exprs.extend(
            [
                drawdown_from_high.alias(f"dd_from_high_{window}d_refine"),
                range_pos.alias(f"range_pos_{window}d_refine"),
                finite_or_null(corr.pow(2)).alias(f"trend_linearity_{window}d"),
            ]
        )
    stock = stock.with_columns(trend_exprs)

    event_exprs = [
        (pl.col("amount_ratio_1_20") > 1.8).alias("is_amount_spike_1d"),
        (pl.col("amount_ratio_5_20") > 1.25).alias("is_amount_expansion_5d"),
        (pl.col("amount_ratio_5_20") < 0.75).alias("is_dry_5d"),
        ((pl.col("ret_5d") < 0.0) & (pl.col("amount_ratio_5_20") < 0.85)).alias("is_dry_pullback_5d"),
        ((pl.col("ret_5d") > 0.03) & (pl.col("price_pos_20d") > 0.80) & (pl.col("amount_ratio_5_20") > 1.10)).alias(
            "is_breakout_volume_confirmed"
        ),
        ((pl.col("amount_ratio_1_20") > 1.8) & (pl.col("ret_1d") < 0.01)).alias("is_volume_without_price_1d"),
        ((pl.col("amount_ratio_5_20") > 1.10) & (pl.col("amount_ratio_20_60") < 0.95)).alias(
            "is_liquidity_recovery"
        ),
    ]
    stock = stock.with_columns(event_exprs)

    rank_cols = [
        ("dd_from_high_64d_refine", True),
        ("dd_from_high_128d_refine", True),
        ("trend_linearity_64d", True),
        ("trend_linearity_128d", True),
        ("ma_stack_stability_20d", True),
        ("amount_ratio_1_20", True),
        ("amount_ratio_5_20", True),
        ("amount_ratio_20_60", True),
    ]
    stock = stock.with_columns([rank_pct(col, higher_is_better=good) for col, good in rank_cols])

    return stock.select(
        [
            "date",
            "code",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "price_pos_20d",
            "amount_ratio_1_20",
            "amount_ratio_5_20",
            "amount_ratio_20_60",
            "dd_from_high_64d_refine",
            "dd_from_high_128d_refine",
            "range_pos_64d_refine",
            "range_pos_128d_refine",
            "trend_linearity_64d",
            "trend_linearity_128d",
            "ma_stack_stability_20d",
            "is_amount_spike_1d",
            "is_amount_expansion_5d",
            "is_dry_5d",
            "is_dry_pullback_5d",
            "is_breakout_volume_confirmed",
            "is_volume_without_price_1d",
            "is_liquidity_recovery",
            "dd_from_high_64d_refine_rank_pct",
            "dd_from_high_128d_refine_rank_pct",
            "trend_linearity_64d_rank_pct",
            "trend_linearity_128d_rank_pct",
            "ma_stack_stability_20d_rank_pct",
            "amount_ratio_1_20_rank_pct",
            "amount_ratio_5_20_rank_pct",
            "amount_ratio_20_60_rank_pct",
        ]
    )


def build_liquidity_trend_features(args: argparse.Namespace) -> pl.DataFrame:
    from strategies.amv.market import build_market_frame

    return add_liquidity_trend_features(build_market_frame(args))
