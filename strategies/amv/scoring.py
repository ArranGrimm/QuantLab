from __future__ import annotations

import polars as pl

from strategies.amv.specs import ScoreComponent


def finite_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan()


def score_component_expr(component: ScoreComponent) -> pl.Expr:
    rank_descending = not component.higher_is_better
    return (
        pl.col(component.factor).rank(method="average", descending=rank_descending).over("date")
        / pl.len().over("date")
        * component.weight
    )


def build_score_expr(components: tuple[ScoreComponent, ...]) -> pl.Expr:
    total_weight = sum(component.weight for component in components)
    if total_weight <= 0:
        raise ValueError("score components must have positive total weight")
    if not components:
        raise ValueError("score components must not be empty")

    expr = score_component_expr(components[0])
    for component in components[1:]:
        expr = expr + score_component_expr(component)
    return expr / total_weight


def required_factor_names(components: tuple[ScoreComponent, ...]) -> list[str]:
    return [component.factor for component in components]
