"""Unified AMV ranker export pipeline.

Driven by strategy JSON config (RankerSpec + rules list).
Rules are pluggable hooks — pipeline itself has zero if-branches for specific rules.

Pipeline:
  1. build_market_lazy → reader + LazyFrame
  2. compute_required_factors (factor registry, on-demand columns)
  3. For each rule: lazy_features (add columns BEFORE collect)
  4. lf.select(final_cols) — projection pushdown, intermediate columns never materialized
  5. ONE collect → market_df → close reader
  6. ONE with_columns: base_score + sum(penalties) + signal_score + rank
  7. Filter TopN → signal_rows
  8. For each rule: gate (filter tiny signal_rows, ~2000 rows)
  9. Build backtest frame → write
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from strategies.amv.data import MarketConfig, build_market_lazy, resolve_end_date
from strategies.amv.export import SignalArtifact, write_signal_artifact
from strategies.amv.factors import ranker_score_expr, ranker_required_columns, finite_expr
from strategies.amv.factors.registry import compute_required_factors
from strategies.amv.hooks import RuleHook, resolve_hooks
from strategies.amv.registry import Strategy
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
    """All parameters needed by the unified ranker export pipeline.

    Sector-related fields retained for future SectorTailwindHook.
    """

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


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def export_ranker_strategy(
    strategy: Strategy,
    config: PipelineConfig,
    output_dir: Path,
) -> SignalArtifact:
    """Export signal for any ranker-based AMV strategy.

    Rules are resolved from strategy JSON via hook registry — pipeline
    itself has no knowledge of specific rules.
    """
    hooks_with_ids = resolve_hooks(strategy.rules)
    hooks = [h for h, _ in hooks_with_ids]
    hook_params = _build_hook_params(strategy, config)

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

    ranker_cols = ranker_required_columns(strategy.ranker)
    lf = compute_required_factors(lf, ranker_cols)

    for hook, params in zip(hooks, hook_params):
        lf = hook.lazy_features(lf, params)

    # ── Phase 2: Projection pushdown → ONE collect ──
    final_cols = set(EXPORT_COLUMNS)
    final_cols |= set(ranker_cols)
    for hook in hooks:
        final_cols |= set(hook.penalty_columns())
    lf = lf.select(list(final_cols))
    market = lf.collect()
    reader.close()

    # ── Phase 3: ONE with_columns (score + penalties + rank) ──
    candidate_expr = _build_candidate_expr(
        required_cols=ranker_cols,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
    )
    penalty_exprs = [h.penalty(p) for h, p in zip(hooks, hook_params)]
    context_penalty = (
        pl.sum_horizontal(penalty_exprs) if penalty_exprs else pl.lit(0.0)
    )
    base_score = ranker_score_expr(strategy.ranker)

    # _score_and_select enriches market with score/rank columns, then
    # extracts signal_rows. market retains all columns for export join.
    signal_rows, market = _score_and_select(
        market=market,
        base_score_expr=base_score,
        context_penalty_expr=context_penalty,
        candidate_expr=candidate_expr,
        strategy_name=strategy.name,
        top_n=config.top_n,
    )

    # ── Phase 4: Gate (ONLY on signal_rows — tiny, ~2000 rows) ──
    for hook, params in zip(hooks, hook_params):
        signal_rows = hook.gate(signal_rows, params)

    # ── Phase 5: Build export frame ──
    available = [c for c in EXPORT_COLUMNS if c in market.columns]
    market = market.select(available)
    export = _build_backtest_signal_frame(market, signal_rows)
    return write_signal_artifact(export=export, output_dir=output_dir)


def _build_hook_params(
    strategy: Strategy, config: PipelineConfig
) -> list[dict]:
    """Build params dict for each hook, merging rule params with config defaults."""
    amv_defaults = {
        "bull_trigger_pct": config.amv_bull_trigger_pct,
        "bull_lookback_days": config.amv_bull_lookback_days,
        "bear_trigger_1d_pct": config.amv_bear_trigger_1d_pct,
        "effective_lag_days": config.amv_effective_lag_days,
    }
    params_list: list[dict] = []
    for rc in strategy.raw_rules:
        p = dict(rc.get("params", {}))
        if rc["id"] == "amv-regime-gate":
            p.update(amv_defaults)
        params_list.append(p)
    return params_list


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_candidate_expr(
    required_cols: list[str],
    mv_min: float,
    amount_ma20_min: float,
    require_bull_regime: bool = True,
) -> pl.Expr:
    candidate = (
        (pl.col("market_cap_100m") >= mv_min)
        & (pl.col("amount_ma20") >= amount_ma20_min)
    )
    if require_bull_regime:
        candidate = pl.col("is_bull_regime") & candidate
    for col_name in required_cols:
        candidate = finite_expr(col_name) & candidate
    return candidate


def _score_and_select(
    *,
    market: pl.DataFrame,
    base_score_expr: pl.Expr,
    context_penalty_expr: pl.Expr,
    candidate_expr: pl.Expr,
    strategy_name: str,
    top_n: int,
) -> pl.DataFrame:
    """Compute scores, apply penalties, rank on FULL cross-section, select TopN.

    Returns (signal_rows, market_enriched) — market remains available for
    the export join, with score/rank columns added.
    """
    market = market.with_columns([
        candidate_expr.alias("_is_signal_candidate"),
        base_score_expr.alias("_base_signal_score"),
        context_penalty_expr.alias("_context_penalty"),
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

    signal_rows = (
        market.filter(
            pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= top_n)
        )
        .select([
            pl.col("date").alias("signal_date"),
            pl.col("code"),
            pl.lit(strategy_name).alias("sleeve_id"),
            pl.col("_signal_score").alias("score"),
            pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
        ])
        .sort(["signal_date", "rank", "code"])
    )

    return signal_rows, market


def _shift_to_execution(
    market: pl.DataFrame, signal_rows: pl.DataFrame
) -> pl.DataFrame:
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


def _build_backtest_signal_frame(
    market: pl.DataFrame, signal_rows: pl.DataFrame
) -> pl.DataFrame:
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
