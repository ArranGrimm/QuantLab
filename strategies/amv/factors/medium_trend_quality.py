from __future__ import annotations

import argparse
from typing import Any

import polars as pl

_MEDIUM_INPUT_COLS = (
    "date",
    "code",
    "close_adj",
    "pre_close_adj",
    "open_adj",
    "high_adj",
    "low_adj",
    "amount",
)

_MEDIUM_PENALTY_COLS = (
    "date",
    "code",
    "structure_score_128d",
    "trend_quality_score_128d",
)


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def _medium_penalty_lazy(lf: pl.LazyFrame, *, window: int = 128) -> pl.LazyFrame:
    """仅计算 medium penalty 需要的 128d 结构/质量分（lazy，单次 collect）。"""

    high = pl.col("close_adj").rolling_max(window).over("code")
    low = pl.col("close_adj").rolling_min(window).over("code")
    ret = pl.col("close_adj") / pl.col("close_adj").shift(window).over("code") - 1.0
    vol = pl.col("ret_1d").rolling_std(window).over("code")
    path_len = pl.col("abs_ret_1d").rolling_sum(window).over("code")
    ma = pl.col("close_adj").rolling_mean(window).over("code")

    lf = lf.with_columns(
        [
            ret.alias(f"ret_{window}d"),
            safe_div(pl.col("close_adj") - low, high - low).alias(f"pos_{window}d"),
            safe_div(ret.abs(), path_len).alias(f"trend_eff_{window}d"),
            pl.col("is_up_day").rolling_mean(window).over("code").alias(f"up_ratio_{window}d"),
            safe_div(ret, vol).alias(f"ret_vol_{window}d"),
            (safe_div(ma, ma.shift(20).over("code")) - 1.0).alias(f"ma_slope_{window}d"),
            pl.col("body_efficiency_1d").rolling_mean(window).over("code").alias(f"body_efficiency_{window}d"),
        ]
    )

    rank_specs = [
        (f"ret_{window}d", True),
        (f"pos_{window}d", True),
        (f"trend_eff_{window}d", True),
        (f"up_ratio_{window}d", True),
        (f"ret_vol_{window}d", True),
        (f"ma_slope_{window}d", True),
        (f"body_efficiency_{window}d", True),
    ]
    rank_exprs = [
        (
            pl.col(col_name).rank("average", descending=not higher_is_better).over("date")
            / pl.len().over("date")
        ).alias(f"{col_name}_rank_pct")
        for col_name, higher_is_better in rank_specs
    ]
    lf = lf.with_columns(rank_exprs)

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

    return lf.with_columns(
        [
            structure_score.alias(f"structure_score_{window}d"),
            quality_score.alias(f"trend_quality_score_{window}d"),
        ]
    ).select(list(_MEDIUM_PENALTY_COLS))


def compute_medium_penalty(
    market: pl.DataFrame,
    rule_params: dict[str, Any],
) -> pl.DataFrame:
    """自包含：从 market frame 计算 medium 特征 → 生成 penalty 列。

    不 join，不依赖外部数据。返回 penalty 列：(date, code, _medium_penalty, _medium_weak)。
    """
    mode = rule_params.get("medium_penalty_mode", "linear")
    penalty_val = rule_params.get("medium_penalty", 0.03)
    weak_threshold = rule_params.get("medium_weak_threshold", 0.50)
    window = 128

    lf = (
        market.lazy()
        .sort(["code", "date"])
        .with_columns([
            (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
            (pl.col("close_adj") > pl.col("pre_close_adj")).alias("is_up_day"),
            ((pl.col("close_adj") - pl.col("open_adj")).abs()
             / (pl.col("high_adj") - pl.col("low_adj")).abs().clip(lower_bound=1e-12))
            .alias("body_efficiency_1d"),
        ])
        .with_columns(pl.col("ret_1d").abs().alias("abs_ret_1d"))
    )

    high = pl.col("close_adj").rolling_max(window).over("code")
    low = pl.col("close_adj").rolling_min(window).over("code")
    ret = pl.col("close_adj") / pl.col("close_adj").shift(window).over("code") - 1.0
    vol = pl.col("ret_1d").rolling_std(window).over("code")
    path_len = pl.col("abs_ret_1d").rolling_sum(window).over("code")
    ma = pl.col("close_adj").rolling_mean(window).over("code")

    lf = lf.with_columns([
        ret.alias(f"ret_{window}d"),
        safe_div(pl.col("close_adj") - low, high - low).alias(f"pos_{window}d"),
        safe_div(ret.abs(), path_len).alias(f"trend_eff_{window}d"),
        pl.col("is_up_day").rolling_mean(window).over("code").alias(f"up_ratio_{window}d"),
        safe_div(ret, vol).alias(f"ret_vol_{window}d"),
        (safe_div(ma, ma.shift(20).over("code")) - 1.0).alias(f"ma_slope_{window}d"),
        pl.col("body_efficiency_1d").rolling_mean(window).over("code").alias(f"body_efficiency_{window}d"),
    ])

    rank_cols = [
        (f"ret_{window}d", True),
        (f"pos_{window}d", True),
        (f"trend_eff_{window}d", True),
        (f"up_ratio_{window}d", True),
        (f"ret_vol_{window}d", True),
        (f"ma_slope_{window}d", True),
        (f"body_efficiency_{window}d", True),
    ]
    for col_name, higher_is_better in rank_cols:
        lf = lf.with_columns(
            (pl.col(col_name).rank("average", descending=not higher_is_better).over("date")
             / pl.len().over("date")).alias(f"{col_name}_rank_pct")
        )

    structure = (
        pl.col(f"ret_{window}d_rank_pct")
        + pl.col(f"pos_{window}d_rank_pct")
        + pl.col(f"ma_slope_{window}d_rank_pct")
    ) / 3.0
    quality = (
        pl.col(f"trend_eff_{window}d_rank_pct")
        + pl.col(f"up_ratio_{window}d_rank_pct")
        + pl.col(f"ret_vol_{window}d_rank_pct")
        + pl.col(f"body_efficiency_{window}d_rank_pct")
    ) / 4.0

    medium_weak = (structure.fill_null(1.0) < weak_threshold) & (quality.fill_null(1.0) < weak_threshold)
    s_sh = (weak_threshold - structure.fill_null(1.0)) / weak_threshold
    q_sh = (weak_threshold - quality.fill_null(1.0)) / weak_threshold
    strength = pl.when(medium_weak).then(((s_sh + q_sh) / 2.0).clip(0.0, 1.0)).otherwise(0.0)
    penalty = strength * penalty_val

    return (
        lf.select(["date", "code", penalty.alias("_medium_penalty"), medium_weak.alias("_medium_weak")])
        .collect(streaming=True)
    )


def add_medium_trend_penalty_features(market: pl.DataFrame) -> pl.DataFrame:
    """export 用：只产出 penalty 两列，避免 64/128 全特征 + 多次 eager 拷贝。"""
    lf = (
        market.select(list(_MEDIUM_INPUT_COLS))
        .lazy()
        .sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                (pl.col("close_adj") > pl.col("pre_close_adj")).alias("is_up_day"),
                (
                    (pl.col("close_adj") - pl.col("open_adj")).abs()
                    / (pl.col("high_adj") - pl.col("low_adj")).abs().clip(lower_bound=1e-12)
                ).alias("body_efficiency_1d"),
            ]
        )
        .with_columns(pl.col("ret_1d").abs().alias("abs_ret_1d"))
    )
    return _medium_penalty_lazy(lf).collect(streaming=True)


def add_medium_trend_features(market: pl.DataFrame) -> pl.DataFrame:
    """研究/诊断用全量 medium 特征（export 默认请用 add_medium_trend_penalty_features）。"""
    lf = (
        market.select(list(_MEDIUM_INPUT_COLS))
        .lazy()
        .sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                (pl.col("close_adj") > pl.col("pre_close_adj")).alias("is_up_day"),
                (
                    (pl.col("close_adj") - pl.col("open_adj")).abs()
                    / (pl.col("high_adj") - pl.col("low_adj")).abs().clip(lower_bound=1e-12)
                ).alias("body_efficiency_1d"),
            ]
        )
        .with_columns(
            [
                pl.col("ret_1d").abs().alias("abs_ret_1d"),
                pl.col("ret_1d").rolling_std(20).over("code").alias("vol_20d"),
            ]
        )
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
                pl.col("body_efficiency_1d").rolling_mean(window).over("code").alias(f"body_efficiency_{window}d"),
            ]
        )

    lf = lf.with_columns(feature_exprs)

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
        rank_exprs = [
            (
                pl.col(col_name).rank("average", descending=not higher_is_better).over("date")
                / pl.len().over("date")
            ).alias(f"{col_name}_rank_pct")
            for col_name, higher_is_better in rank_specs
        ]
        lf = lf.with_columns(rank_exprs)
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
        lf = lf.with_columns(
            [
                structure_score.alias(f"structure_score_{window}d"),
                quality_score.alias(f"trend_quality_score_{window}d"),
            ]
        )

    return lf.collect(streaming=True).select(
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
