from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from strategies.amv.export import SignalArtifact, SignalArtifactConfig, write_signal_artifact
from strategies.amv.factors.limit_ecology import LIMIT_TOLERANCE
from strategies.amv.factors.medium_trend_quality import add_medium_trend_features
from strategies.amv.factors.sector_tailwind import build_sector_features
from strategies.amv.market import build_market_frame
from strategies.amv.rules.context import build_context_signal, context_config_payload
from strategies.amv.rules.limit_weakgate import (
    build_limit_ecology_market,
    build_limit_weakgate_signal,
    limit_config_payload,
)
from strategies.amv.rules.pb3_regime import apply_pb3_regime_gate
from strategies.amv.signals import (
    SignalAssemblyConfig,
    assemble_ranker_signal,
    build_backtest_signal_frame,
)
from strategies.amv.specs import SleeveSpec


@dataclass(frozen=True)
class WorkflowExportConfig:
    qmt_db: Path
    output_root: Path
    start_date: str = "2019-01-01"
    end_date: str = "2026-05-10"
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
    rank_source: str = "mix_10_20"
    sector_penalty_mode: str = "linear"
    relative_confirm: str = "rel20_under0"
    bottom_rank_threshold: float = 0.40
    sector_penalty: float = 0.02
    medium_penalty_mode: str = "linear"
    weak_threshold: float = 0.50
    medium_penalty: float = 0.03
    pb3_regime_gate: str = "aged_non_accel_or_chaos"
    price_limit_tolerance: float = LIMIT_TOLERANCE


def _as_market_args(config: WorkflowExportConfig) -> argparse.Namespace:
    return argparse.Namespace(
        qmt_db=config.qmt_db,
        start_date=config.start_date,
        end_date=config.end_date,
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


def _as_context_args(config: WorkflowExportConfig) -> argparse.Namespace:
    args = _as_market_args(config)
    args.sector_map = config.sector_map
    args.sector_start_date = config.sector_start_date
    args.refresh_sector_map = config.refresh_sector_map
    args.sector_map_request_sleep = config.sector_map_request_sleep
    args.rank_source = config.rank_source
    args.sector_penalty_mode = config.sector_penalty_mode
    args.relative_confirm = config.relative_confirm
    args.bottom_rank_threshold = config.bottom_rank_threshold
    args.medium_penalty_mode = config.medium_penalty_mode
    args.weak_threshold = config.weak_threshold
    return args


def _summary(export: pl.DataFrame, signal_rows: pl.DataFrame, *, sleeve_id: str) -> dict[str, Any]:
    return {
        "sleeve_id": sleeve_id,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
    }


def _config_payload(config: WorkflowExportConfig, sleeve: SleeveSpec) -> dict[str, Any]:
    return {
        "qmt_db": str(config.qmt_db),
        "sleeve_id": sleeve.id,
        "top_n": config.top_n,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "st_snapshot_date": config.st_snapshot_date,
        "mv_min": config.mv_min,
        "amount_ma20_min": config.amount_ma20_min,
        "amv_bull_trigger_pct": config.amv_bull_trigger_pct,
        "amv_bull_lookback_days": config.amv_bull_lookback_days,
        "amv_bear_trigger_1d_pct": config.amv_bear_trigger_1d_pct,
        "amv_effective_lag_days": config.amv_effective_lag_days,
        "ranker_id": sleeve.ranker.id if sleeve.ranker else None,
        "ranker_weights": dict(sleeve.ranker.weights) if sleeve.ranker else {},
        "pb3_regime_gate": config.pb3_regime_gate,
    }


def export_limit_weakgate_sleeve(
    sleeve: SleeveSpec,
    config: WorkflowExportConfig,
    *,
    repo_root: Path | None = None,
    started_at: datetime | None = None,
) -> SignalArtifact:
    sleeve_id = sleeve.id
    if sleeve_id != "limit_first_board_pullback_weakgate":
        raise ValueError(f"unsupported limit ecology native sleeve: {sleeve_id}")

    started = started_at or datetime.now()
    market = build_limit_ecology_market(config, _as_market_args(config))
    signal_rows, weakgate_summary = build_limit_weakgate_signal(market=market, sleeve_id=sleeve_id, config=config)
    export = build_backtest_signal_frame(market, signal_rows)
    summary = {
        "sleeve_id": sleeve_id,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
        **weakgate_summary,
    }
    return write_signal_artifact(
        export=export,
        selected=signal_rows,
        artifact_config=SignalArtifactConfig(
            sleeve_id=sleeve_id,
            model_name="daily_raw_limit_ecology_event",
            output_root=config.output_root,
            strategy="amv_limit_ecology_event_sleeve",
            label=f"limit_ecology:{sleeve_id}",
            feature_mode=sleeve_id,
            config=limit_config_payload(config, sleeve_id),
            summary=summary,
        ),
        started_at=started,
        repo_root=repo_root,
    )


def export_ranker_sleeve(
    sleeve: SleeveSpec,
    config: WorkflowExportConfig,
    *,
    repo_root: Path | None = None,
    started_at: datetime | None = None,
) -> SignalArtifact:
    if sleeve.ranker is None:
        raise ValueError(f"sleeve {sleeve.id} does not define a ranker")

    started = started_at or datetime.now()
    market = build_market_frame(_as_market_args(config))
    export, signal_rows, _ = assemble_ranker_signal(
        market,
        sleeve.ranker,
        SignalAssemblyConfig(
            sleeve_id=sleeve.id,
            top_n=config.top_n,
            mv_min=config.mv_min,
            amount_ma20_min=config.amount_ma20_min,
        ),
    )
    if sleeve.id == "pullback_p0_k0_pb3_cp1_rv0" and config.pb3_regime_gate != "none":
        signal_rows, export, gating_summary = apply_pb3_regime_gate(
            market=market,
            signal_rows=signal_rows,
            config=config,
        )
    else:
        gating_summary = {
            "pb3_regime_gate": config.pb3_regime_gate,
            "pb3_regime_gate_applied": False,
        }

    summary = {**_summary(export, signal_rows, sleeve_id=sleeve.id), **gating_summary}
    return write_signal_artifact(
        export=export,
        selected=signal_rows,
        artifact_config=SignalArtifactConfig(
            sleeve_id=sleeve.id,
            model_name="static_factor_sleeve",
            output_root=config.output_root,
            config=_config_payload(config, sleeve),
            summary=summary,
            extra_meta={"feature_count": None},
        ),
        started_at=started,
        repo_root=repo_root,
    )


def export_context_sleeve(
    sleeve: SleeveSpec,
    config: WorkflowExportConfig,
    *,
    repo_root: Path | None = None,
    started_at: datetime | None = None,
) -> SignalArtifact:
    if sleeve.ranker is None:
        raise ValueError(f"sleeve {sleeve.id} does not define a base ranker")

    started = started_at or datetime.now()
    context_args = _as_context_args(config)

    market = build_market_frame(context_args)
    sector_features = build_sector_features(context_args)
    trend_features = add_medium_trend_features(market)
    scored_base = market.join(sector_features, on=["date", "code"], how="left").join(
        trend_features,
        on=["date", "code"],
        how="left",
    )
    sleeve_id, signal_rows, export, summary = build_context_signal(
        scored_base=scored_base,
        market=market,
        sleeve=sleeve,
        config=config,
        context_args=context_args,
    )

    return write_signal_artifact(
        export=export,
        selected=signal_rows,
        artifact_config=SignalArtifactConfig(
            sleeve_id=sleeve_id,
            model_name="static_factor_sleeve_context_combo",
            output_root=config.output_root,
            config=context_config_payload(config, sleeve_id),
            summary=summary,
        ),
        started_at=started,
        repo_root=repo_root,
    )
