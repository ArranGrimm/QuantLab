"""
128 日中期结构 + 趋势质量 penalty 规则。
"""

from __future__ import annotations

from typing import Any

import polars as pl


def apply_medium_trend_penalty(
    scored: pl.DataFrame,
    rule_params: dict[str, Any],
) -> pl.DataFrame:
    mode = rule_params.get("medium_penalty_mode", "linear")
    penalty_val = rule_params.get("medium_penalty", 0.03)
    weak_threshold = rule_params.get("medium_weak_threshold", 0.50)

    structure = pl.col("structure_score_128d").fill_null(1.0)
    quality = pl.col("trend_quality_score_128d").fill_null(1.0)
    medium_weak = (structure < weak_threshold) & (quality < weak_threshold)
    structure_shortfall = (weak_threshold - structure) / weak_threshold
    quality_shortfall = (weak_threshold - quality) / weak_threshold
    medium_strength = (
        pl.when(medium_weak)
        .then(((structure_shortfall + quality_shortfall) / 2.0).clip(0.0, 1.0))
        .otherwise(0.0)
    )
    if mode == "linear":
        penalty = medium_strength * penalty_val
    elif mode == "bucket":
        penalty = pl.when(medium_weak).then(penalty_val).otherwise(0.0)
    else:
        raise ValueError(f"unknown medium penalty mode: {mode}")

    return scored.with_columns(
        [
            medium_weak.alias("_medium_weak"),
            medium_strength.alias("_medium_weak_strength"),
            penalty.alias("_medium_penalty"),
        ]
    )
