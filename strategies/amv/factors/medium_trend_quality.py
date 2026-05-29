from __future__ import annotations

import argparse

import polars as pl


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def add_medium_trend_features(market: pl.DataFrame) -> pl.DataFrame:
    market = market.sort(["code", "date"])
    stock = market.with_columns(
        [
            (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
            (pl.col("close_adj") > pl.col("pre_close_adj")).alias("is_up_day"),
            (
                (pl.col("close_adj") - pl.col("open_adj")).abs()
                / (pl.col("high_adj") - pl.col("low_adj")).abs().clip(lower_bound=1e-12)
            ).alias("body_efficiency_1d"),
        ]
    ).with_columns(
        [
            pl.col("ret_1d").abs().alias("abs_ret_1d"),
            pl.col("ret_1d").rolling_std(20).over("code").alias("vol_20d"),
            pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20_raw"),
            pl.col("amount").rolling_mean(60).over("code").alias("amount_ma60_raw"),
        ]
    )

    feature_exprs: list[pl.Expr] = []
    for window in (64, 128):
        high = pl.col("close_adj").rolling_max(window).over("code")
        low = pl.col("close_adj").rolling_min(window).over("code")
        ret = pl.col("close_adj") / pl.col("close_adj").shift(window).over("code") - 1.0
        vol = pl.col("ret_1d").rolling_std(window).over("code")
        path_len = pl.col("abs_ret_1d").rolling_sum(window).over("code")
        ma = pl.col("close_adj").rolling_mean(window).over("code")
        feature_exprs.extend(
            [
                ret.alias(f"ret_{window}d"),
                safe_div(pl.col("close_adj") - low, high - low).alias(f"pos_{window}d"),
                (pl.col("close_adj") / high - 1.0).alias(f"dd_from_high_{window}d"),
                safe_div(ret.abs(), path_len).alias(f"trend_eff_{window}d"),
                pl.col("is_up_day").rolling_mean(window).over("code").alias(f"up_ratio_{window}d"),
                safe_div(ret, vol).alias(f"ret_vol_{window}d"),
                safe_div(pl.col("vol_20d"), vol).alias(f"vol_contraction_20_{window}d"),
                (safe_div(ma, ma.shift(20).over("code")) - 1.0).alias(f"ma_slope_{window}d"),
                pl.col("body_efficiency_1d").rolling_mean(window).over("code").alias(
                    f"body_efficiency_{window}d"
                ),
            ]
        )

    stock = stock.with_columns(feature_exprs)
    for window in (64, 128):
        rank_specs = [
            (f"ret_{window}d", True),
            (f"pos_{window}d", True),
            (f"trend_eff_{window}d", True),
            (f"up_ratio_{window}d", True),
            (f"ret_vol_{window}d", True),
            (f"vol_contraction_20_{window}d", False),
            (f"ma_slope_{window}d", True),
            (f"body_efficiency_{window}d", True),
        ]
        for col_name, higher_is_better in rank_specs:
            rank_expr = pl.col(col_name).rank("average", descending=not higher_is_better).over("date") / pl.len().over(
                "date"
            )
            stock = stock.with_columns(rank_expr.alias(f"{col_name}_rank_pct"))
        structure_score = (
            pl.col(f"ret_{window}d_rank_pct")
            + pl.col(f"pos_{window}d_rank_pct")
            + pl.col(f"ma_slope_{window}d_rank_pct")
        ) / 3.0
        quality_score = (
            pl.col(f"trend_eff_{window}d_rank_pct")
            + pl.col(f"up_ratio_{window}d_rank_pct")
            + pl.col(f"ret_vol_{window}d_rank_pct")
            + pl.col(f"body_efficiency_{window}d_rank_pct")
        ) / 4.0
        stock = stock.with_columns(
            [
                structure_score.alias(f"structure_score_{window}d"),
                quality_score.alias(f"trend_quality_score_{window}d"),
            ]
        )

    return stock.select(
        [
            "date",
            "code",
            "ret_64d",
            "pos_64d",
            "dd_from_high_64d",
            "trend_eff_64d",
            "up_ratio_64d",
            "ret_vol_64d",
            "vol_contraction_20_64d",
            "ma_slope_64d",
            "body_efficiency_64d",
            "structure_score_64d",
            "trend_quality_score_64d",
            "ret_128d",
            "pos_128d",
            "dd_from_high_128d",
            "trend_eff_128d",
            "up_ratio_128d",
            "ret_vol_128d",
            "vol_contraction_20_128d",
            "ma_slope_128d",
            "body_efficiency_128d",
            "structure_score_128d",
            "trend_quality_score_128d",
            *[
                f"{col}_rank_pct"
                for col in [
                    "ret_64d",
                    "pos_64d",
                    "trend_eff_64d",
                    "up_ratio_64d",
                    "ret_vol_64d",
                    "ma_slope_64d",
                    "body_efficiency_64d",
                    "ret_128d",
                    "pos_128d",
                    "trend_eff_128d",
                    "up_ratio_128d",
                    "ret_vol_128d",
                    "ma_slope_128d",
                    "body_efficiency_128d",
                ]
            ],
        ]
    )


def build_medium_trend_features(args: argparse.Namespace) -> pl.DataFrame:
    from strategies.amv.market import build_market_frame

    return add_medium_trend_features(build_market_frame(args))
