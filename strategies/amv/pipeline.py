"""Unified AMV ranker export pipeline.

One function for ALL ranker-based strategies (trend-, pullback-).
Driven by strategy JSON config (RankerSpec + rules list).

Pipeline:
  1. build_market_lazy → reader + LazyFrame
  2. Add base factors lazily (only columns needed by ranker)
  3. If medium-trend-quality rule: add lazy medium features
  4. If sector-tailwind rule: pre-build sector features eagerly
  5. ONE collect → market_df
  6. Close reader
  7. Apply sector/medium penalties eagerly on full cross-section
  8. Compute base score, context penalty, signal score — on full cross-section
  9. Rank → TopN → signal rows
  10. If amv-regime-gate: apply gate
  11. Trim market to export columns, build backtest frame, write
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from strategies.amv.data import (
    MarketConfig,
    build_market_lazy,
    resolve_end_date,
)
from strategies.amv.export import SignalArtifact, write_signal_artifact
from strategies.amv.factors import (
    calc_amv_core_factors,
    ranker_score_expr,
    ranker_required_columns,
    finite_expr,
    add_medium_trend_features_lazy,
)
from strategies.amv.registry import Strategy
from strategies.amv.regime import build_amv_regime_gate_frame
from strategies.amv.specs import RankerSpec
from utils.data_source import DataSourceSettings, resolve_data_source


EXPORT_COLUMNS = [
    "date", "code",
    "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    "open_raw", "high_raw", "low_raw", "close_raw", "pre_close_raw",
    "is_bull_regime", "amv_mechanical_regime",
    "market_cap_100m", "amount_ma20",
]


@dataclass(frozen=True)
class PipelineConfig:
    """All parameters needed by the unified ranker export pipeline."""

    data_source: DataSourceSettings = field(default_factory=resolve_data_source)
    start_date: str = "2019-01-01"
    end_date: str = ""
    st_snapshot_date: str = "2026-03-31"
    mv_min: float = 100.0
    amount_ma20_min: float = 5e7
    top_n: int = 3
    amv_bull_trigger_pct: float = 4.0
    amv_bull_lookback_days: int = 2
    amv_bear_trigger_1d_pct: float = -2.3
    amv_effective_lag_days: int = 1
    sector_map: Path = Path("data/sector_map_em.csv")
    sector_start_date: str = "2019-01-01"
    refresh_sector_map: bool = False
    sector_map_request_sleep: float = 0.35


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def export_ranker_strategy(
    strategy: Strategy,
    config: PipelineConfig,
    output_dir: Path,
) -> SignalArtifact:
    """Export signal for any ranker-based AMV strategy.

    Supports: trend-p2, trend-p3, trend-p3-medium, trend-p3-enhanced, pullback-pb3.
    """
    rules_set = set(strategy.rules)
    rule_params = strategy.rule_params
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

    # ── Phase 1: Lazy chain ──
    reader, lf = build_market_lazy(market_cfg)

    required = ranker_required_columns(strategy.ranker)
    lf = calc_amv_core_factors(lf)

    has_medium = "medium-trend-quality" in rules_set
    if has_medium:
        lf = add_medium_trend_features_lazy(lf, window=128)

    # ── Pre-build sector features (separate data pipeline) ──
    sector_df: pl.DataFrame | None = None
    has_sector = "sector-tailwind" in rules_set
    if has_sector:
        sector_df = _build_sector_df(config, rule_params)

    # ── Phase 2: ONE collect ──
    market = lf.collect()
    reader.close()

    # ── Phase 3: Eager post-processing ──
    candidate_expr = _build_candidate_expr(
        required_cols=required,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
    )

    if has_sector and sector_df is not None:
        market = _apply_sector_penalty(market, sector_df, rule_params)
    else:
        market = market.with_columns(pl.lit(0.0).alias("_sector_penalty"))

    if has_medium:
        market = _compute_medium_penalty_cols(market, rule_params)
    else:
        market = market.with_columns(
            pl.lit(0.0).alias("_medium_penalty"),
            pl.lit(False).alias("_medium_weak"),
        )

    # ── Phase 4: Score, rank, TopN ──
    signal_rows = _score_and_select(
        market=market,
        ranker=strategy.ranker,
        candidate_expr=candidate_expr,
        strategy_name=strategy.name,
        top_n=config.top_n,
        has_sector=has_sector,
        has_medium=has_medium,
    )

    # ── Phase 5: Gate (pullback-pb3 only) ──
    if "amv-regime-gate" in rules_set:
        signal_rows = _apply_regime_gate(signal_rows, config)

    # ── Phase 6: Build export frame ──
    available = [c for c in EXPORT_COLUMNS if c in market.columns]
    market = market.select(available)

    export = _build_backtest_signal_frame(market, signal_rows)
    return write_signal_artifact(export=export, output_dir=output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_sector_df(config: PipelineConfig, rule_params: dict[str, Any]) -> pl.DataFrame:
    from strategies.amv.factors.sector_tailwind import (
        load_sector_map,
        load_daily_with_industry,
        build_sector_tailwind_features,
    )

    sector_map = load_sector_map(
        config.sector_map,
        refresh=config.refresh_sector_map,
        request_sleep=config.sector_map_request_sleep,
    )
    daily = load_daily_with_industry(config.data_source, sector_map, config.sector_start_date)
    return build_sector_tailwind_features(daily).select([
        "date", "code", "industry",
        "sector_ret_5d_rank_pct", "sector_ret_10d_rank_pct", "sector_ret_20d_rank_pct",
        "stock_rel_sector_ret_5d", "stock_rel_sector_ret_10d", "stock_rel_sector_ret_20d",
        "sector_breadth_ma20", "sector_amount_ratio_20", "sector_tailwind_ok",
    ])


def _apply_sector_penalty(
    market: pl.DataFrame,
    sector_df: pl.DataFrame,
    rule_params: dict[str, Any],
) -> pl.DataFrame:
    rank_source = rule_params.get("sector_rank_source", "mix_10_20")
    mode = rule_params.get("sector_penalty_mode", "linear")
    penalty_val = rule_params.get("sector_penalty", 0.02)
    bottom_threshold = rule_params.get("sector_bottom_rank_threshold", 0.40)
    relative_confirm_key = rule_params.get("sector_relative_confirm", "rel20_under0")

    if rank_source == "5d":
        sector_rank = pl.col("sector_ret_5d_rank_pct")
    elif rank_source == "10d":
        sector_rank = pl.col("sector_ret_10d_rank_pct")
    elif rank_source == "20d":
        sector_rank = pl.col("sector_ret_20d_rank_pct")
    elif rank_source == "mix_10_20":
        sector_rank = (pl.col("sector_ret_10d_rank_pct") + pl.col("sector_ret_20d_rank_pct")) / 2.0
    else:
        raise ValueError(f"unknown rank source: {rank_source}")

    sector_rank = sector_rank.fill_null(1.0)
    sector_bottom_distance = (bottom_threshold - sector_rank) / bottom_threshold
    sector_bottom_strength = pl.when(sector_bottom_distance > 0.0).then(sector_bottom_distance).otherwise(0.0)

    if relative_confirm_key == "none":
        sector_confirm = pl.lit(True)
    elif relative_confirm_key == "rel5_under0":
        sector_confirm = pl.col("stock_rel_sector_ret_5d").fill_null(0.0) < 0.0
    elif relative_confirm_key == "rel10_under0":
        sector_confirm = pl.col("stock_rel_sector_ret_10d").fill_null(0.0) < 0.0
    elif relative_confirm_key == "rel20_under0":
        sector_confirm = pl.col("stock_rel_sector_ret_20d").fill_null(0.0) < 0.0
    else:
        raise ValueError(f"unknown relative confirm: {relative_confirm_key}")

    if mode == "linear":
        penalty = pl.when(sector_confirm).then(sector_bottom_strength * penalty_val).otherwise(0.0)
    elif mode == "bucket":
        penalty = pl.when((sector_rank < bottom_threshold) & sector_confirm).then(penalty_val).otherwise(0.0)
    else:
        raise ValueError(f"unknown sector penalty mode: {mode}")

    joined = market.join(sector_df, on=["date", "code"], how="left")
    return joined.with_columns([
        sector_rank.alias("_sector_rank_score"),
        sector_confirm.alias("_sector_relative_confirm"),
        penalty.alias("_sector_penalty"),
    ])


def _compute_medium_penalty_cols(market: pl.DataFrame, rule_params: dict[str, Any]) -> pl.DataFrame:
    """Compute medium penalty from pre-computed structure/quality scores."""
    penalty_val = rule_params.get("medium_penalty", 0.03)
    weak_threshold = rule_params.get("medium_weak_threshold", 0.50)

    structure = pl.col("_structure_score_128d").fill_null(1.0)
    quality = pl.col("_quality_score_128d").fill_null(1.0)

    medium_weak = (structure < weak_threshold) & (quality < weak_threshold)
    s_sh = (weak_threshold - structure) / weak_threshold
    q_sh = (weak_threshold - quality) / weak_threshold
    strength = pl.when(medium_weak).then(((s_sh + q_sh) / 2.0).clip(0.0, 1.0)).otherwise(0.0)

    return market.with_columns([
        medium_weak.alias("_medium_weak"),
        strength.alias("_medium_weak_strength"),
        (strength * penalty_val).alias("_medium_penalty"),
    ])


def _build_candidate_expr(
    required_cols: list[str],
    mv_min: float,
    amount_ma20_min: float,
    require_bull_regime: bool = True,
) -> pl.Expr:
    candidate = (pl.col("market_cap_100m") >= mv_min) & (pl.col("amount_ma20") >= amount_ma20_min)
    if require_bull_regime:
        candidate = pl.col("is_bull_regime") & candidate
    for col_name in required_cols:
        candidate = candidate & finite_expr(col_name)
    return candidate


def _score_and_select(
    *,
    market: pl.DataFrame,
    ranker: RankerSpec,
    candidate_expr: pl.Expr,
    strategy_name: str,
    top_n: int,
    has_sector: bool,
    has_medium: bool,
) -> pl.DataFrame:
    """Compute scores, apply penalties, rank on FULL cross-section, select TopN."""
    penalty_terms: list[pl.Expr] = []
    if has_sector:
        penalty_terms.append(pl.col("_sector_penalty"))
    if has_medium:
        penalty_terms.append(pl.col("_medium_penalty"))
    context_penalty = pl.sum_horizontal(penalty_terms) if penalty_terms else pl.lit(0.0)

    base_score = ranker_score_expr(ranker)

    # Score on FULL cross-section for correct rank/pl.len() baseline.
    # Non-candidates get _signal_score = None → ordinal rank pushes them to the end.
    market = market.with_columns([
        candidate_expr.alias("_is_signal_candidate"),
        base_score.alias("_base_signal_score"),
        context_penalty.alias("_context_penalty"),
    ])

    market = market.with_columns(
        pl.when(pl.col("_is_signal_candidate"))
        .then(pl.col("_base_signal_score") - pl.col("_context_penalty"))
        .otherwise(None)
        .alias("_signal_score"),
    )

    market = market.with_columns(
        pl.col("_signal_score")
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("_signal_rank"),
    )

    return (
        market.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= top_n))
        .select([
            pl.col("date").alias("signal_date"),
            pl.col("code"),
            pl.lit(strategy_name).alias("sleeve_id"),
            pl.col("_signal_score").alias("score"),
            pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
        ])
        .sort(["signal_date", "rank", "code"])
    )


def _apply_regime_gate(signal_rows: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Apply AMV regime gate: skip signals on aged+non-accelerating or chaos days."""
    gate = build_amv_regime_gate_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    )
    gated = signal_rows.join(gate, on="signal_date", how="left").with_columns(
        pl.col("gate_skip").fill_null(False),
    )
    kept = gated.filter(~pl.col("gate_skip")).sort(["signal_date", "rank", "code"])
    return kept.select(signal_rows.columns)


def _shift_to_execution(market: pl.DataFrame, signal_rows: pl.DataFrame) -> pl.DataFrame:
    """Shift signal_date to the next trading day (execution date)."""
    trading_dates = (
        market.select("date").unique().sort("date")
        .with_columns(pl.col("date").shift(-1).alias("execution_date"))
        .drop_nulls("execution_date")
    )
    return (
        signal_rows
        .join(trading_dates, left_on="signal_date", right_on="date", how="inner")
        .select(["execution_date", "code", "signal_date", "sleeve_id", "score", "rank"])
        .rename({"execution_date": "date"})
    )


def _build_backtest_signal_frame(market: pl.DataFrame, signal_rows: pl.DataFrame) -> pl.DataFrame:
    """Build the full daily panel with signal columns for backtesting."""
    execution_signals = _shift_to_execution(market, signal_rows)
    return (
        market.join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            is_signal=pl.col("signal_date").is_not_null(),
            score=pl.col("score").fill_null(0.0),
            rank=pl.col("rank").fill_null(9999).cast(pl.UInt32),
            sleeve_id=pl.col("sleeve_id").fill_null(""),
        )
        .sort(["date", "code"])
    )
