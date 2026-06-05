"""AMV 因子层 —— 所有因子公式的唯一真相源。

每项函数在此定义一次，其余模块禁止内联因子公式。
"""
from __future__ import annotations

import polars as pl

# ── Re-exports from internal modules ──
from strategies.amv.factors.base import (
    AMV_BASE_FACTOR_SPECS,
    calc_amv_core_factors,
    build_amv_base_factors,
)
from strategies.amv.factors.limit_ecology import (
    LIMIT_TOLERANCE,
    add_limit_ecology_features,
    load_raw_daily,
)
from strategies.amv.factors.medium_trend_quality import (
    add_medium_trend_features,
    build_medium_trend_features,
    safe_div,
)
from strategies.amv.factors.sector_tailwind import (
    build_sector_features,
    build_sector_tailwind_features,
    format_stock_code,
    load_daily_with_industry,
    load_sector_map,
    rank_source_token,
    refresh_em_sector_map,
    relative_confirm_expr,
    sector_rank_expr,
    threshold_token,
)
from strategies.amv.specs import (
    FactorSpec,
    ScoreComponent,
    RankerSpec,
    FactorDirection,
    FactorRole,
    RuleType,
    RuleSpec,
    RULES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring primitives
# ═══════════════════════════════════════════════════════════════════════════

def finite_expr(col_name: str) -> pl.Expr:
    """True if column is not null and not NaN."""
    return pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan()


def score_component_expr(component: ScoreComponent) -> pl.Expr:
    """Cross-sectional rank percentile for a single score component."""
    rank_descending = not component.higher_is_better
    return (
        pl.col(component.factor)
        .rank(method="average", descending=rank_descending)
        .over("date")
        / pl.len().over("date")
        * component.weight
    )


def build_score_expr(components: tuple[ScoreComponent, ...]) -> pl.Expr:
    """Weighted composite score from multiple components."""
    total_weight = sum(c.weight for c in components)
    if total_weight <= 0:
        raise ValueError("score components must have positive total weight")
    if not components:
        raise ValueError("score components must not be empty")
    expr = score_component_expr(components[0])
    for c in components[1:]:
        expr = expr + score_component_expr(c)
    return expr / total_weight


def required_factor_names(components: tuple[ScoreComponent, ...]) -> list[str]:
    return [c.factor for c in components]


# ═══════════════════════════════════════════════════════════════════════════
# Ranker helpers
# ═══════════════════════════════════════════════════════════════════════════

def ranker_score_expr(ranker: RankerSpec) -> pl.Expr:
    """Build scoring expression for a RankerSpec."""
    if ranker.components:
        return build_score_expr(ranker.components)
    if ranker.factor is None or ranker.descending is None:
        raise ValueError(f"ranker {ranker.id} must define either components or factor/descending")
    return (
        pl.col(ranker.factor)
        .rank(method="average", descending=not ranker.descending)
        .over("date")
        / pl.len().over("date")
    )


def ranker_required_columns(ranker: RankerSpec) -> list[str]:
    """List factor column names required by a RankerSpec."""
    if ranker.components:
        return required_factor_names(ranker.components)
    if ranker.factor is None:
        raise ValueError(f"ranker {ranker.id} must define factor or components")
    return [ranker.factor]


# ═══════════════════════════════════════════════════════════════════════════
# Medium trend features (lazy — no internal collect)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_div_medium(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def add_medium_trend_features_lazy(lf: pl.LazyFrame, *, window: int = 128) -> pl.LazyFrame:
    """Add 128d medium trend structure/quality features to a LazyFrame.

    PURE LAZY — no internal collect. Must be applied BEFORE the single collect
    on the full continuous trading day dataset.

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
        _safe_div_medium(
            pl.col("close_adj") - pl.col("close_adj").rolling_min(W).over("code"),
            pl.col("close_adj").rolling_max(W).over("code") - pl.col("close_adj").rolling_min(W).over("code"),
        ).alias(f"_pos_{W}d"),
        _safe_div_medium(
            (pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0).abs(),
            pl.col("_abs_ret").rolling_sum(W).over("code"),
        ).alias(f"_trend_eff_{W}d"),
        pl.col("_upday").rolling_mean(W).over("code").alias(f"_up_ratio_{W}d"),
        _safe_div_medium(
            pl.col("close_adj") / pl.col("close_adj").shift(W).over("code") - 1.0,
            pl.col("_ret1d").rolling_std(W).over("code"),
        ).alias(f"_ret_vol_{W}d"),
        (
            _safe_div_medium(
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


__all__ = [
    "AMV_BASE_FACTOR_SPECS",
    "LIMIT_TOLERANCE",
    "add_limit_ecology_features",
    "add_medium_trend_features",
    "add_medium_trend_features_lazy",
    "build_medium_trend_features",
    "build_sector_features",
    "build_sector_tailwind_features",
    "build_amv_base_factors",
    "build_score_expr",
    "calc_amv_core_factors",
    "FactorDirection",
    "FactorRole",
    "FactorSpec",
    "finite_expr",
    "format_stock_code",
    "load_daily_with_industry",
    "load_raw_daily",
    "load_sector_map",
    "RankerSpec",
    "ranker_required_columns",
    "ranker_score_expr",
    "rank_source_token",
    "refresh_em_sector_map",
    "relative_confirm_expr",
    "required_factor_names",
    "RuleSpec",
    "RuleType",
    "RULES",
    "safe_div",
    "ScoreComponent",
    "score_component_expr",
    "sector_rank_expr",
    "threshold_token",
]
