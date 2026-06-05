"""Event strategy export pipeline.

Separate from pipeline.py because event strategies use:
  - first_board_event_score_expr instead of ranker components
  - Limit ecology features (limit up/down, streaks, pullback setups)
  - Weak window gate (optional, 7-dimension context scoring)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from strategies.amv.data import (
    MarketConfig,
    build_market_lazy,
    resolve_end_date,
)
from strategies.amv.export import SignalArtifact, write_signal_artifact
from strategies.amv.factors import calc_amv_core_factors, finite_expr
from strategies.amv.factors.limit_ecology import (
    LIMIT_TOLERANCE,
    add_limit_ecology_features,
)
from strategies.amv.pipeline import PipelineConfig, _shift_to_execution, _build_backtest_signal_frame, EXPORT_COLUMNS
from strategies.amv.registry import Strategy
from strategies.amv.regime import build_amv_phase_frame


# ═══════════════════════════════════════════════════════════════════════════
# Event scoring and weakgate (migrated from deleted rules/event_weakgate.py)
# ═══════════════════════════════════════════════════════════════════════════

def _bool_score(col_name: str, weight: float) -> pl.Expr:
    return pl.when(pl.col(col_name).fill_null(False)).then(weight).otherwise(0.0)


def _first_board_event_score_expr() -> pl.Expr:
    """First-board pullback event scoring formula."""
    base_score = (
        _bool_score("is_reboard_after_pullback", 5.0)
        + _bool_score("was_limit_up_yesterday", 4.0)
        + _bool_score("is_reclaim_after_limit", 3.0)
        + _bool_score("has_limit_up_5d", 2.0)
        + _bool_score("has_limit_up_10d", 1.5)
        + _bool_score("has_limit_up_20d", 1.0)
        + _bool_score("has_one_word_limit_up_10d", 0.5)
        - _bool_score("has_failed_limit_up_5d", 1.0)
    )
    recency_bonus = (
        pl.when(pl.col("days_since_prior_limit_up").is_between(1, 3)).then(2.0)
        .when(pl.col("days_since_prior_limit_up").is_between(4, 10)).then(1.0)
        .otherwise(0.0)
    )
    first_board_bonus = (
        pl.when(pl.col("_last_lu_streak_before") == 1).then(1.0)
        .when(pl.col("_last_lu_streak_before") == 2).then(-1.0)
        .otherwise(0.0)
    )
    liquidity_bonus = (
        pl.when(pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.20).then(0.5).otherwise(0.0)
    )
    risk_penalty = (
        pl.col("atr_14_pct_rank_pct").fill_null(0.5)
        + pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    ) / 2.0
    return base_score + recency_bonus + first_board_bonus + liquidity_bonus - risk_penalty


def _add_weak_window_context(scored: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Build 7-dimension weak window context scores.

    Each sub-score is 0-1, summed as _weak_window_score.
    Gate threshold: _weak_window_score >= 3.0 → skip signal day.
    """
    market_breadth = (
        scored.group_by("date")
        .agg(pl.col("is_close_limit_up").fill_null(False).sum().alias("_weak_limit_up_count"))
        .sort("date")
        .with_columns(
            (pl.col("_weak_limit_up_count").rank("average") / pl.len()).alias("_weak_limit_up_count_rank_pct")
        )
    )

    candidates = scored.filter(pl.col("_is_signal_candidate"))
    top3 = candidates.filter(pl.col("_base_signal_rank") <= config.top_n)
    candidate_health = (
        candidates.group_by("date")
        .agg(
            pl.len().alias("_weak_candidate_count"),
            pl.col("_base_signal_score").mean().alias("_weak_candidate_avg_score"),
        )
        .join(
            top3.group_by("date").agg(
                pl.col("_base_signal_score").mean().alias("_weak_top3_avg_score"),
                pl.col("atr_14_pct_rank_pct").mean().alias("_weak_top3_avg_atr_rank"),
                (pl.col("days_since_prior_limit_up") >= 7).mean().alias("_weak_top3_stale_share"),
                pl.col("is_reclaim_after_limit").mean().alias("_weak_top3_reclaim_share"),
            ),
            on="date", how="left",
        )
    )

    amv_phase = build_amv_phase_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    ).select(
        "date",
        pl.col("amv_slope_5d").alias("_weak_amv_slope_5d"),
        pl.col("amv_dd_from_high").alias("_weak_amv_dd_from_high"),
        pl.col("amv_neg_streak").alias("_weak_amv_neg_streak"),
    )

    enriched = (
        scored.join(market_breadth, on="date", how="left")
        .join(candidate_health, on="date", how="left")
        .join(amv_phase, on="date", how="left")
    )

    limit_low = ((0.55 - pl.col("_weak_limit_up_count_rank_pct").fill_null(0.55)) / 0.55).clip(0, 1)
    pool_thin = ((12.0 - pl.col("_weak_candidate_count").fill_null(12).cast(pl.Float64)) / 8.0).clip(0, 1)
    score_low = ((8.0 - pl.col("_weak_top3_avg_score").fill_null(8.0)) / 3.0).clip(0, 1)
    atr_high = ((pl.col("_weak_top3_avg_atr_rank").fill_null(0.75) - 0.75) / 0.20).clip(0, 1)
    stale_high = pl.col("_weak_top3_stale_share").fill_null(0.0).clip(0, 1)
    reclaim_low = ((0.45 - pl.col("_weak_top3_reclaim_share").fill_null(0.45)) / 0.45).clip(0, 1)
    amv_flat = ((2.0 - pl.col("_weak_amv_slope_5d").fill_null(2.0)) / 4.0).clip(0, 1)

    return enriched.with_columns([
        limit_low.alias("_weak_limit_low_score"),
        pool_thin.alias("_weak_pool_thin_score"),
        score_low.alias("_weak_score_low_score"),
        atr_high.alias("_weak_atr_high_score"),
        stale_high.alias("_weak_stale_high_score"),
        reclaim_low.alias("_weak_reclaim_low_score"),
        amv_flat.alias("_weak_amv_flat_score"),
    ]).with_columns(
        (
            pl.col("_weak_limit_low_score")
            + pl.col("_weak_pool_thin_score")
            + pl.col("_weak_score_low_score")
            + pl.col("_weak_atr_high_score")
            + pl.col("_weak_stale_high_score")
            + pl.col("_weak_reclaim_low_score")
            + pl.col("_weak_amv_flat_score")
        ).alias("_weak_window_score")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def export_event_strategy(
    strategy: Strategy,
    config: PipelineConfig,
    output_dir: Path,
) -> SignalArtifact:
    """Export signal for event-based strategies.

    Supports: event-firstboard, event-firstboard-base.

    Pipeline:
      1. Build market frame (data.py)
      2. Load limit ecology features (separate raw OHLC reader)
      3. Join ecology to market
      4. Compute event candidate filter + event score
      5. If event-weakgate rule: build weak window context scores + gate
      6. Rank → TopN
      7. Build backtest signal frame → write
    """
    rules_set = set(strategy.rules)
    end_date = config.end_date or resolve_end_date(config.data_source)

    market_cfg = MarketConfig(
        data_source=config.data_source,
        start_date=config.start_date,
        end_date=end_date,
        st_snapshot_date=config.st_snapshot_date,
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        effective_lag_days=config.amv_effective_lag_days,
    )

    # ── Phase 1: Market frame + limit ecology ──
    reader, lf = build_market_lazy(market_cfg)
    lf = calc_amv_core_factors(lf)
    market = lf.collect(streaming=True)
    reader.close()

    raw_daily = _load_event_raw_daily(config, end_date)
    ecology = add_limit_ecology_features(raw_daily, tolerance=LIMIT_TOLERANCE)

    market = market.join(ecology, on=["date", "code"], how="left").with_columns(
        (pl.col("atr_14_pct").rank("average").over("date") / pl.len().over("date")).alias("atr_14_pct_rank_pct"),
        (pl.col("panic_vol_ratio_20d").rank("average").over("date") / pl.len().over("date")).alias("panic_vol_ratio_20d_rank_pct"),
    )

    # ── Phase 2: Event scoring ──
    event_candidate = pl.col("is_first_board_pullback_setup").fill_null(False)
    valid_expr = finite_expr("price_pos_20d") & finite_expr("amount_ma20")
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= config.mv_min)
        & (pl.col("amount_ma20") >= config.amount_ma20_min)
        & valid_expr
        & event_candidate
    )

    scored = market.with_columns([
        candidate_expr.alias("_is_signal_candidate"),
        pl.when(candidate_expr).then(_first_board_event_score_expr()).otherwise(None).alias("_base_signal_score"),
    ]).with_columns(
        pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_signal_rank")
    )

    skip_gate = "event-weakgate" not in rules_set
    if skip_gate:
        scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score"))
    else:
        scored = _add_weak_window_context(scored, config)
        scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score"))

    scored = scored.with_columns(
        pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank")
    )

    # ── Phase 3: TopN selection ──
    if skip_gate:
        select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n)
    else:
        select_expr = (
            pl.col("_is_signal_candidate")
            & (pl.col("_signal_rank") <= config.top_n)
            & (pl.col("_weak_window_score").fill_null(0.0) < 3.0)
        )

    signal_rows = (
        scored.filter(select_expr)
        .select(
            signal_date=pl.col("date"),
            code="code",
            sleeve_id=pl.lit(strategy.name),
            score=pl.col("_signal_score"),
            rank=pl.col("_signal_rank").cast(pl.UInt32),
        )
        .sort(["signal_date", "rank", "code"])
    )

    # ── Phase 4: Export ──
    available = [c for c in EXPORT_COLUMNS if c in market.columns]
    market = market.select(available)
    export = _build_backtest_signal_frame(market, signal_rows)
    return write_signal_artifact(export=export, output_dir=output_dir)


def _load_event_raw_daily(config: PipelineConfig, end_date: str) -> pl.DataFrame:
    """Load raw OHLC (not QFQ) for limit ecology feature computation."""
    from utils.data_source import daily_reader
    from utils import get_st_blacklist_pl

    st_list = get_st_blacklist_pl(config.st_snapshot_date)
    with daily_reader(config.data_source) as reader:
        raw = (
            reader.load_raw_ohlc(config.start_date, end_date)
            .filter(~pl.col("code").is_in(st_list))
        )
    return raw
