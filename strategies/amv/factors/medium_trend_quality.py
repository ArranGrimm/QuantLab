"""Medium trend quality features — 128d structure/quality composite.

Pure lazy, no internal collect. Must be applied BEFORE the single collect
on the full continuous trading day dataset.
"""
from __future__ import annotations

import polars as pl


def _safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def add_medium_trend_features_lazy(lf: pl.LazyFrame, *, window: int = 128) -> pl.LazyFrame:
    """Add 128d medium trend structure/quality features to a LazyFrame.

    PURE LAZY — no internal collect.

    Adds: _structure_score_128d, _quality_score_128d (+ intermediates).
    """
    W = window

    lf = lf.with_columns([
        (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("_ret1d"),
        (pl.col("close_adj") > pl.col("pre_close_adj")).alias("_upday"),
        (
            (pl.col("close_adj") - pl.col("open_adj")).abs()
            / pl.max_horizontal((pl.col("high_adj") - pl.col("low_adj")).abs(), pl.lit(1e-12))
        ).alias("_body_eff"),
    ]).with_columns(pl.col("_ret1d").abs().alias("_abs_ret"))

    lf = lf.with_columns([
        (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).alias(f"_ret_{W}d"),
        _safe_div(
            pl.col("close_adj") - pl.col("close_adj").rolling_min(W).over("code"),
            pl.col("close_adj").rolling_max(W).over("code") - pl.col("close_adj").rolling_min(W).over("code"),
        ).alias(f"_pos_{W}d"),
        _safe_div(
            (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).abs(),
            pl.col("_abs_ret").rolling_sum(W).over("code"),
        ).alias(f"_trend_eff_{W}d"),
        pl.col("_upday").rolling_mean(W).over("code").alias(f"_up_ratio_{W}d"),
        _safe_div(
            pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0,
            pl.col("_ret1d").rolling_std(W).over("code"),
        ).alias(f"_ret_vol_{W}d"),
        (
            _safe_div(
                pl.col("close_adj").rolling_mean(W).over("code"),
                pl.col("close_adj").rolling_mean(W).over("code").shift(20).over("code"),
            ) - 1.0
        ).alias(f"_ma_slope_{W}d"),
        pl.col("_body_eff").rolling_mean(W).over("code").alias(f"_body_eff_{W}d"),
    ])

    rank_specs = [
        (f"_ret_{W}d", True), (f"_pos_{W}d", True), (f"_trend_eff_{W}d", True),
        (f"_up_ratio_{W}d", True), (f"_ret_vol_{W}d", True), (f"_ma_slope_{W}d", True),
        (f"_body_eff_{W}d", True),
    ]
    for col_name, higher_is_better in rank_specs:
        lf = lf.with_columns(
            (
                pl.col(col_name)
                .rank("average", descending=not higher_is_better)
                .over("date")
                / pl.len().over("date")
            ).alias(f"{col_name}_rank_pct")
        )

    structure_score = (
        pl.col(f"_ret_{W}d_rank_pct")
        + pl.col(f"_pos_{W}d_rank_pct")
        + pl.col(f"_ma_slope_{W}d_rank_pct")
    ) / 3.0

    quality_score = (
        pl.col(f"_trend_eff_{W}d_rank_pct")
        + pl.col(f"_up_ratio_{W}d_rank_pct")
        + pl.col(f"_ret_vol_{W}d_rank_pct")
        + pl.col(f"_body_eff_{W}d_rank_pct")
    ) / 4.0

    return lf.with_columns([
        structure_score.alias("_structure_score_128d"),
        quality_score.alias("_quality_score_128d"),
    ])
