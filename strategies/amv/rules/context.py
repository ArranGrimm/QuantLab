from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.factors.sector_tailwind import (
    rank_source_token,
    relative_confirm_expr,
    sector_rank_expr,
    threshold_token,
)
from strategies.amv.signals import (
    base_candidate_expr,
    build_backtest_signal_frame,
    ranker_required_columns,
    ranker_score_expr,
)
from strategies.amv.specs import SleeveSpec


def context_sleeve_id(config: Any) -> str:
    sleeve_id = (
        f"p3_ctx_sector{rank_source_token(config.rank_source)}_{config.sector_penalty_mode}"
        f"_b{threshold_token(config.bottom_rank_threshold)}_sp{threshold_token(config.sector_penalty)}"
        f"_medium128_{config.medium_penalty_mode}_t{threshold_token(config.weak_threshold)}"
        f"_mp{threshold_token(config.medium_penalty)}"
    )
    if config.relative_confirm != "none":
        sleeve_id = f"{sleeve_id}_{config.relative_confirm}"
    return sleeve_id


def context_config_payload(config: Any, sleeve_id: str) -> dict[str, Any]:
    return {
        "qmt_db": str(config.qmt_db),
        "sector_map": str(config.sector_map),
        "sector_start_date": config.sector_start_date,
        "top_n": config.top_n,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "sector_rank_source": config.rank_source,
        "sector_penalty_mode": config.sector_penalty_mode,
        "sector_penalty": config.sector_penalty,
        "relative_confirm": config.relative_confirm,
        "bottom_rank_threshold": config.bottom_rank_threshold,
        "medium_penalty_mode": config.medium_penalty_mode,
        "medium_penalty": config.medium_penalty,
        "weak_threshold": config.weak_threshold,
        "sleeve_id": sleeve_id,
    }


def _sector_penalty_expr(config: Any, context_args: Any) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    sector_rank = sector_rank_expr(context_args).fill_null(1.0)
    sector_bottom_distance = (config.bottom_rank_threshold - sector_rank) / config.bottom_rank_threshold
    sector_bottom_strength = pl.when(sector_bottom_distance > 0.0).then(sector_bottom_distance).otherwise(0.0)
    sector_confirm = relative_confirm_expr(context_args)
    if config.sector_penalty_mode == "linear":
        penalty = pl.when(sector_confirm).then(sector_bottom_strength * config.sector_penalty).otherwise(0.0)
    elif config.sector_penalty_mode == "bucket":
        penalty = (
            pl.when((sector_rank < config.bottom_rank_threshold) & sector_confirm)
            .then(config.sector_penalty)
            .otherwise(0.0)
        )
    else:
        raise ValueError(f"unknown sector penalty mode: {config.sector_penalty_mode}")
    return sector_rank, sector_confirm, penalty


def _medium_penalty_expr(config: Any) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    structure = pl.col("structure_score_128d").fill_null(1.0)
    quality = pl.col("trend_quality_score_128d").fill_null(1.0)
    medium_weak = (structure < config.weak_threshold) & (quality < config.weak_threshold)
    structure_shortfall = (config.weak_threshold - structure) / config.weak_threshold
    quality_shortfall = (config.weak_threshold - quality) / config.weak_threshold
    medium_strength = (
        pl.when(medium_weak)
        .then(((structure_shortfall + quality_shortfall) / 2.0).clip(lower_bound=0.0, upper_bound=1.0))
        .otherwise(0.0)
    )
    if config.medium_penalty_mode == "linear":
        penalty = medium_strength * config.medium_penalty
    elif config.medium_penalty_mode == "bucket":
        penalty = pl.when(medium_weak).then(config.medium_penalty).otherwise(0.0)
    else:
        raise ValueError(f"unknown medium penalty mode: {config.medium_penalty_mode}")
    return medium_weak, medium_strength, penalty


def build_context_signal(
    *,
    scored_base: pl.DataFrame,
    market: pl.DataFrame,
    sleeve: SleeveSpec,
    config: Any,
    context_args: Any,
) -> tuple[str, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    if sleeve.ranker is None:
        raise ValueError(f"sleeve {sleeve.id} does not define a base ranker")

    sleeve_id = context_sleeve_id(config)
    required_cols = ranker_required_columns(sleeve.ranker)
    candidate_expr = base_candidate_expr(
        required_cols,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
        require_bull_regime=True,
    )
    sector_rank, sector_confirm, sector_penalty = _sector_penalty_expr(config, context_args)
    medium_weak, medium_strength, medium_penalty = _medium_penalty_expr(config)

    scored = (
        scored_base.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            [
                ranker_score_expr(sleeve.ranker).alias("_base_signal_score"),
                sector_rank.alias("_sector_rank_score"),
                sector_confirm.alias("_relative_confirm"),
                sector_penalty.alias("_sector_penalty"),
                medium_weak.alias("_medium_weak"),
                medium_strength.alias("_medium_weak_strength"),
                medium_penalty.alias("_medium_penalty"),
            ]
        )
        .with_columns((pl.col("_sector_penalty") + pl.col("_medium_penalty")).alias("_context_penalty"))
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score") - pl.col("_context_penalty"))
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank"))
    )
    raw_top3 = (
        scored.filter(pl.col("_is_signal_candidate"))
        .with_columns(pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_rank"))
        .filter(pl.col("_base_rank") <= config.top_n)
        .select(["date", "code"])
    )
    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                "industry",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_base_signal_score").alias("base_score"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("_sector_rank_score").alias("sector_rank_score"),
                pl.col("_relative_confirm").alias("relative_confirm"),
                pl.col("_sector_penalty").alias("sector_penalty"),
                pl.col("stock_rel_sector_ret_20d"),
                pl.col("structure_score_128d"),
                pl.col("trend_quality_score_128d"),
                pl.col("_medium_weak").alias("medium_weak"),
                pl.col("_medium_weak_strength").alias("medium_weak_strength"),
                pl.col("_medium_penalty").alias("medium_penalty"),
                pl.col("_context_penalty").alias("context_penalty"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )
    export = build_backtest_signal_frame(market, signal_rows)

    selected_key = signal_rows.select([pl.col("signal_date").alias("date"), "code"])
    overlap = selected_key.join(raw_top3, on=["date", "code"], how="inner").height
    context_penalized = int((signal_rows["context_penalty"] > 0.0).sum()) if signal_rows.height else 0
    summary = {
        "sleeve_id": sleeve_id,
        "base_sleeve_id": "candidate_p3_k0p5_b0_c0_r0",
        "sector_rank_source": config.rank_source,
        "sector_penalty_mode": config.sector_penalty_mode,
        "sector_penalty": config.sector_penalty,
        "relative_confirm": config.relative_confirm,
        "bottom_rank_threshold": config.bottom_rank_threshold,
        "medium_penalty_mode": config.medium_penalty_mode,
        "medium_penalty": config.medium_penalty,
        "weak_threshold": config.weak_threshold,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "raw_top3_overlap_rows": overlap,
        "raw_top3_overlap_ratio": overlap / signal_rows.height if signal_rows.height else None,
        "selected_sector_penalized_rows": int((signal_rows["sector_penalty"] > 0.0).sum()) if signal_rows.height else 0,
        "selected_medium_penalized_rows": int((signal_rows["medium_penalty"] > 0.0).sum()) if signal_rows.height else 0,
        "selected_context_penalized_rows": context_penalized,
        "selected_context_penalized_ratio": context_penalized / signal_rows.height if signal_rows.height else None,
        "selected_sector_penalty_mean": float(signal_rows["sector_penalty"].mean()) if signal_rows.height else None,
        "selected_medium_penalty_mean": float(signal_rows["medium_penalty"].mean()) if signal_rows.height else None,
        "selected_context_penalty_mean": float(signal_rows["context_penalty"].mean()) if signal_rows.height else None,
        "selected_context_penalty_max": float(signal_rows["context_penalty"].max()) if signal_rows.height else None,
    }
    return sleeve_id, signal_rows, export, summary
