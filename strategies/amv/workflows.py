"""
AMV 策略导出编排层。

统一管道: market 构造 → ranker 评分 → penalties 扣分 → TopN 选择 → gates 过滤 → 导出
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from strategies.amv.export import SignalArtifact, write_signal_artifact
from strategies.amv.factors.limit_ecology import LIMIT_TOLERANCE, add_limit_ecology_features, load_raw_daily
from strategies.amv.factors.medium_trend_quality import add_medium_trend_features
from strategies.amv.factors.sector_tailwind import build_sector_features
from strategies.amv.market import build_market_frame
from strategies.amv.registry import Strategy
from strategies.amv.rules.amv_regime import apply_amv_regime_gate
from strategies.amv.rules.event_weakgate import build_event_weakgate_signal
from strategies.amv.rules.medium_penalty import apply_medium_trend_penalty
from strategies.amv.rules.sector_penalty import apply_sector_tailwind_penalty
from strategies.amv.signals import (
    SignalAssemblyConfig,
    assemble_ranker_signal,
    base_candidate_expr,
    build_backtest_signal_frame,
    ranker_required_columns,
    ranker_score_expr,
)


@dataclass(frozen=True)
class WorkflowExportConfig:
    """所有 AMV 策略共用的基础导出配置。"""

    qmt_db: Path
    start_date: str = "2019-01-01"
    end_date: str = ""  # 空 = 自动取数据库最新日期
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
    price_limit_tolerance: float = LIMIT_TOLERANCE


# === Market frame helpers ===


def _resolve_end_date(config: WorkflowExportConfig) -> str:
    if config.end_date:
        return config.end_date
    import duckdb
    with duckdb.connect(str(config.qmt_db), read_only=True) as conn:
        max_date = conn.execute("SELECT MAX(date) FROM v_stock_daily_qfq_qmt").fetchone()[0]
        return str(max_date)


def _as_market_args(config: WorkflowExportConfig) -> argparse.Namespace:
    return argparse.Namespace(
        qmt_db=config.qmt_db,
        start_date=config.start_date,
        end_date=_resolve_end_date(config),
        st_snapshot_date=config.st_snapshot_date,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
        top_n=config.top_n,
        amv_bull_trigger_pct=config.amv_bull_trigger_pct,
        amv_bull_lookback_days=config.amv_bull_lookback_days,
        amv_bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        amv_effective_lag_days=config.amv_effective_lag_days,
        price_limit_tolerance=config.price_limit_tolerance,
    )


def _context_args(config: WorkflowExportConfig, rule_params: dict[str, Any]) -> argparse.Namespace:
    args = _as_market_args(config)
    args.sector_map = config.sector_map
    args.sector_start_date = config.sector_start_date
    args.refresh_sector_map = config.refresh_sector_map
    args.sector_map_request_sleep = config.sector_map_request_sleep
    args.rank_source = rule_params.get("sector_rank_source", "mix_10_20")
    args.sector_penalty_mode = rule_params.get("sector_penalty_mode", "linear")
    args.relative_confirm = rule_params.get("sector_relative_confirm", "rel20_under0")
    args.bottom_rank_threshold = rule_params.get("sector_bottom_rank_threshold", 0.40)
    return args


def _build_event_market(config: WorkflowExportConfig) -> pl.DataFrame:
    market_args = _as_market_args(config)
    market = build_market_frame(market_args)
    raw_daily = load_raw_daily(market_args)
    ecology = add_limit_ecology_features(raw_daily, tolerance=config.price_limit_tolerance)
    return market.join(ecology, on=["date", "code"], how="left").with_columns(
        [
            (pl.col("atr_14_pct").rank("average").over("date") / pl.len().over("date")).alias("atr_14_pct_rank_pct"),
            (pl.col("panic_vol_ratio_20d").rank("average").over("date") / pl.len().over("date")).alias("panic_vol_ratio_20d_rank_pct"),
        ]
    )


# === Unified export pipeline ===


def export_strategy(
    strategy: Strategy,
    config: WorkflowExportConfig,
    output_dir: Path,
) -> SignalArtifact:
    """统一策略导出管道 → artifacts/<strategy>/signal.parquet"""
    if strategy.family == "event":
        return _export_event(strategy, config, output_dir)

    rules_set = set(strategy.rules)
    if rules_set & {"sector-tailwind", "medium-trend-quality"}:
        return _export_context_combo(strategy, config, output_dir)

    return _export_ranker(strategy, config, output_dir)


# === Event family ===


def _export_event(
    strategy: Strategy,
    config: WorkflowExportConfig,
    output_dir: Path,
) -> SignalArtifact:
    market = _build_event_market(config)
    signal_rows = build_event_weakgate_signal(
        market=market,
        sleeve_id=strategy.name,
        config=config,
        skip_gate="event-weakgate" not in strategy.rules,
    )
    export = build_backtest_signal_frame(market, signal_rows)
    return write_signal_artifact(export=export, output_dir=output_dir)


# === Trend / Pullback ranker pipeline ===


def _export_ranker(
    strategy: Strategy,
    config: WorkflowExportConfig,
    output_dir: Path,
) -> SignalArtifact:
    """纯 ranker 管道（trend-p2, trend-p3, pullback-pb3）。"""
    market = build_market_frame(_as_market_args(config))
    export, signal_rows, _ = assemble_ranker_signal(
        market, strategy.ranker,
        SignalAssemblyConfig(sleeve_id=strategy.name, top_n=config.top_n, mv_min=config.mv_min, amount_ma20_min=config.amount_ma20_min),
    )

    if "amv-regime-gate" in set(strategy.rules):
        signal_rows, export, _ = apply_amv_regime_gate(market=market, signal_rows=signal_rows, config=config)

    return write_signal_artifact(export=export, output_dir=output_dir)


# === Context combo (sector + medium penalty rerank) ===


def _export_context_combo(
    strategy: Strategy,
    config: WorkflowExportConfig,
    output_dir: Path,
) -> SignalArtifact:
    """trend-p3-enhanced: sector + medium128 两层 penalty 后 rerank Top3。"""
    rule_params = strategy.rule_params
    ctx_args = _context_args(config, rule_params)

    market = build_market_frame(ctx_args)
    scored_base = market.join(
        build_sector_features(ctx_args), on=["date", "code"], how="left"
    ).join(add_medium_trend_features(market), on=["date", "code"], how="left")

    required_cols = ranker_required_columns(strategy.ranker)
    candidate_expr = base_candidate_expr(required_cols, mv_min=config.mv_min, amount_ma20_min=config.amount_ma20_min)

    scored = (
        scored_base.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .pipe(lambda df: apply_sector_tailwind_penalty(df, ctx_args, rule_params))
        .pipe(lambda df: apply_medium_trend_penalty(df, rule_params))
        .with_columns([
            ranker_score_expr(strategy.ranker).alias("_base_signal_score"),
            (pl.col("_sector_penalty") + pl.col("_medium_penalty")).alias("_context_penalty"),
        ])
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score") - pl.col("_context_penalty"))
            .otherwise(None).alias("_signal_score")
        )
        .with_columns(pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank"))
    )

    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n))
        .select([
            pl.col("date").alias("signal_date"), "code",
            pl.lit(strategy.name).alias("sleeve_id"),
            pl.col("_signal_score").alias("score"),
            pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
        ])
        .sort(["signal_date", "rank", "code"])
    )

    export = build_backtest_signal_frame(market, signal_rows)
    return write_signal_artifact(export=export, output_dir=output_dir)
