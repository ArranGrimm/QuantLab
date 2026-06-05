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
    add_medium_trend_features_lazy,
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


__all__ = [
    "AMV_BASE_FACTOR_SPECS",
    "LIMIT_TOLERANCE",
    "add_limit_ecology_features",
    "add_medium_trend_features_lazy",
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
    "ScoreComponent",
    "score_component_expr",
    "sector_rank_expr",
    "threshold_token",
]
