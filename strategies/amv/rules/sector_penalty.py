"""
行业顺风 penalty 规则。

对候选池中行业排名底部的股票做 soft penalty——不直接过滤，而是扣分后重新排序。
"""

from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.factors.sector_tailwind import relative_confirm_expr, sector_rank_expr


def apply_sector_tailwind_penalty(
    scored: pl.DataFrame,
    context_args: Any,
    rule_params: dict[str, Any],
) -> pl.DataFrame:
    mode = rule_params.get("sector_penalty_mode", "linear")
    penalty_val = rule_params.get("sector_penalty", 0.02)
    bottom_threshold = rule_params.get("sector_bottom_rank_threshold", 0.40)

    sector_rank = sector_rank_expr(context_args).fill_null(1.0)
    sector_bottom_distance = (bottom_threshold - sector_rank) / bottom_threshold
    sector_bottom_strength = pl.when(sector_bottom_distance > 0.0).then(sector_bottom_distance).otherwise(0.0)
    sector_confirm = relative_confirm_expr(context_args)

    if mode == "linear":
        penalty = pl.when(sector_confirm).then(sector_bottom_strength * penalty_val).otherwise(0.0)
    elif mode == "bucket":
        penalty = pl.when((sector_rank < bottom_threshold) & sector_confirm).then(penalty_val).otherwise(0.0)
    else:
        raise ValueError(f"unknown sector penalty mode: {mode}")

    return scored.with_columns(
        [
            sector_rank.alias("_sector_rank_score"),
            sector_confirm.alias("_sector_relative_confirm"),
            penalty.alias("_sector_penalty"),
        ]
    )
