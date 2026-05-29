from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from strategies.amv.export import SignalArtifact, SignalArtifactConfig, write_signal_artifact
from strategies.amv.factors.limit_ecology import LIMIT_TOLERANCE, add_limit_ecology_features, load_raw_daily
from strategies.amv.factors.medium_trend_quality import add_medium_trend_features
from strategies.amv.factors.sector_tailwind import (
    build_sector_features,
    rank_source_token,
    relative_confirm_expr,
    sector_rank_expr,
    threshold_token,
)
from strategies.amv.market import build_market_frame
from strategies.amv.regime import build_amv_phase_frame, build_pb3_regime_gate_frame
from strategies.amv.scoring import finite_expr
from strategies.amv.signals import (
    SignalAssemblyConfig,
    assemble_ranker_signal,
    base_candidate_expr,
    build_backtest_signal_frame,
    ranker_required_columns,
    ranker_score_expr,
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


def _context_sleeve_id(config: WorkflowExportConfig) -> str:
    sleeve_id = (
        f"p3_ctx_sector{rank_source_token(config.rank_source)}_{config.sector_penalty_mode}"
        f"_b{threshold_token(config.bottom_rank_threshold)}_sp{threshold_token(config.sector_penalty)}"
        f"_medium128_{config.medium_penalty_mode}_t{threshold_token(config.weak_threshold)}"
        f"_mp{threshold_token(config.medium_penalty)}"
    )
    if config.relative_confirm != "none":
        sleeve_id = f"{sleeve_id}_{config.relative_confirm}"
    return sleeve_id


def _context_config_payload(config: WorkflowExportConfig, sleeve_id: str) -> dict[str, Any]:
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


def _bool_score(col_name: str, weight: float) -> pl.Expr:
    return pl.when(pl.col(col_name).fill_null(False)).then(weight).otherwise(0.0)


def _limit_first_board_score_expr() -> pl.Expr:
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
        pl.when(pl.col("days_since_prior_limit_up").is_between(1, 3))
        .then(2.0)
        .when(pl.col("days_since_prior_limit_up").is_between(4, 10))
        .then(1.0)
        .otherwise(0.0)
    )
    first_board_bonus = (
        pl.when(pl.col("_last_lu_streak_before") == 1)
        .then(1.0)
        .when(pl.col("_last_lu_streak_before") == 2)
        .then(-1.0)
        .otherwise(0.0)
    )
    liquidity_bonus = pl.when(pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.20).then(0.5).otherwise(0.0)
    risk_penalty = (
        pl.col("atr_14_pct_rank_pct").fill_null(0.5) + pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    ) / 2.0
    return base_score + recency_bonus + first_board_bonus + liquidity_bonus - risk_penalty


def _build_limit_ecology_market(config: WorkflowExportConfig) -> pl.DataFrame:
    args = _as_market_args(config)
    market = build_market_frame(args)
    raw_daily = load_raw_daily(args)
    ecology = add_limit_ecology_features(raw_daily, tolerance=config.price_limit_tolerance)
    return market.join(ecology, on=["date", "code"], how="left").with_columns(
        [
            (pl.col("atr_14_pct").rank("average").over("date") / pl.len().over("date")).alias(
                "atr_14_pct_rank_pct"
            ),
            (pl.col("panic_vol_ratio_20d").rank("average").over("date") / pl.len().over("date")).alias(
                "panic_vol_ratio_20d_rank_pct"
            ),
        ]
    )


def _add_weak_window_context(scored: pl.DataFrame, config: WorkflowExportConfig) -> pl.DataFrame:
    market_breadth = (
        scored.group_by("date")
        .agg(pl.col("is_close_limit_up").fill_null(False).sum().alias("_weak_limit_up_count"))
        .sort("date")
        .with_columns((pl.col("_weak_limit_up_count").rank("average") / pl.len()).alias("_weak_limit_up_count_rank_pct"))
    )
    candidates = scored.filter(pl.col("_is_signal_candidate"))
    top3 = candidates.filter(pl.col("_base_signal_rank") <= config.top_n)
    candidate_health = (
        candidates.group_by("date")
        .agg(
            [
                pl.len().alias("_weak_candidate_count"),
                pl.col("_base_signal_score").mean().alias("_weak_candidate_avg_score"),
            ]
        )
        .join(
            top3.group_by("date").agg(
                [
                    pl.col("_base_signal_score").mean().alias("_weak_top3_avg_score"),
                    pl.col("atr_14_pct_rank_pct").mean().alias("_weak_top3_avg_atr_rank"),
                    (pl.col("days_since_prior_limit_up") >= 7).mean().alias("_weak_top3_stale_share"),
                    pl.col("is_reclaim_after_limit").mean().alias("_weak_top3_reclaim_share"),
                ]
            ),
            on="date",
            how="left",
        )
    )
    amv_phase = build_amv_phase_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    ).select(
        [
            "date",
            pl.col("amv_slope_5d").alias("_weak_amv_slope_5d"),
            pl.col("amv_dd_from_high").alias("_weak_amv_dd_from_high"),
            pl.col("amv_neg_streak").alias("_weak_amv_neg_streak"),
        ]
    )
    enriched = scored.join(market_breadth, on="date", how="left").join(
        candidate_health, on="date", how="left"
    ).join(amv_phase, on="date", how="left")

    limit_low = ((0.55 - pl.col("_weak_limit_up_count_rank_pct").fill_null(0.55)) / 0.55).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    pool_thin = ((12.0 - pl.col("_weak_candidate_count").fill_null(12).cast(pl.Float64)) / 8.0).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    score_low = ((8.0 - pl.col("_weak_top3_avg_score").fill_null(8.0)) / 3.0).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    atr_high = ((pl.col("_weak_top3_avg_atr_rank").fill_null(0.75) - 0.75) / 0.20).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    stale_high = pl.col("_weak_top3_stale_share").fill_null(0.0).clip(lower_bound=0.0, upper_bound=1.0)
    reclaim_low = ((0.45 - pl.col("_weak_top3_reclaim_share").fill_null(0.45)) / 0.45).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    amv_flat = ((2.0 - pl.col("_weak_amv_slope_5d").fill_null(2.0)) / 4.0).clip(
        lower_bound=0.0, upper_bound=1.0
    )
    return enriched.with_columns(
        [
            limit_low.alias("_weak_limit_low_score"),
            pool_thin.alias("_weak_pool_thin_score"),
            score_low.alias("_weak_score_low_score"),
            atr_high.alias("_weak_atr_high_score"),
            stale_high.alias("_weak_stale_high_score"),
            reclaim_low.alias("_weak_reclaim_low_score"),
            amv_flat.alias("_weak_amv_flat_score"),
        ]
    ).with_columns(
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


def _limit_config_payload(config: WorkflowExportConfig, sleeve_id: str) -> dict[str, Any]:
    return {
        "qmt_db": str(config.qmt_db),
        "sleeve_id": sleeve_id,
        "top_n": config.top_n,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "st_snapshot_date": config.st_snapshot_date,
        "mv_min": config.mv_min,
        "amount_ma20_min": config.amount_ma20_min,
        "price_limit_tolerance": config.price_limit_tolerance,
        "amv_bull_trigger_pct": config.amv_bull_trigger_pct,
        "amv_bull_lookback_days": config.amv_bull_lookback_days,
        "amv_bear_trigger_1d_pct": config.amv_bear_trigger_1d_pct,
        "amv_effective_lag_days": config.amv_effective_lag_days,
        "execution_price_note": (
            "Exports raw OHLC/pre-close for Rust execution while keeping adjusted OHLC for factor compatibility."
        ),
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
    market = _build_limit_ecology_market(config)
    event_candidate = pl.col("is_first_board_pullback_setup").fill_null(False)
    valid_expr = finite_expr("price_pos_20d") & finite_expr("amount_ma20")
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= config.mv_min)
        & (pl.col("amount_ma20") >= config.amount_ma20_min)
        & valid_expr
        & event_candidate
    )
    scored = market.with_columns(
        [
            candidate_expr.alias("_is_signal_candidate"),
            pl.when(candidate_expr).then(_limit_first_board_score_expr()).otherwise(None).alias("_base_signal_score"),
        ]
    ).with_columns(
        pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_signal_rank")
    )
    scored = _add_weak_window_context(scored, config)
    scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score")).with_columns(
        pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank")
    )

    weak_score = pl.col("_weak_window_score").fill_null(0.0)
    select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= config.top_n) & (weak_score < 3.0)
    selected_cols: list[pl.Expr | str] = [
        pl.col("date").alias("signal_date"),
        "code",
        pl.lit(sleeve_id).alias("sleeve_id"),
        pl.col("_signal_score").alias("score"),
        pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
        "days_since_prior_limit_up",
        "_last_lu_streak_before",
        "limit_up_count_20d",
        "is_reboard_after_pullback",
        "is_reclaim_after_limit",
        "is_first_board_pullback_setup",
        "has_failed_limit_up_5d",
        "amount_ratio_5_20",
        "atr_14_pct_rank_pct",
        "panic_vol_ratio_20d_rank_pct",
        "_weak_window_score",
        "_weak_limit_up_count_rank_pct",
        "_weak_candidate_count",
        "_weak_top3_avg_score",
        "_weak_top3_avg_atr_rank",
        "_weak_top3_stale_share",
        "_weak_top3_reclaim_share",
        "_weak_amv_slope_5d",
        "_weak_amv_dd_from_high",
    ]
    signal_rows = scored.filter(select_expr).select(selected_cols).sort(["signal_date", "rank", "code"])
    export = build_backtest_signal_frame(market, signal_rows)
    candidate_rows = scored.filter(pl.col("_is_signal_candidate"))
    summary = {
        "sleeve_id": sleeve_id,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "candidate_rows_before_shift": candidate_rows.height,
        "candidate_days_before_shift": candidate_rows.select("date").n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "weak_window_signal_rows": int(signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).height),
        "weak_window_signal_days": int(
            signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).select("signal_date").n_unique()
        ),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
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
            config=_limit_config_payload(config, sleeve_id),
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
    gating_summary: dict[str, Any] = {
        "pb3_regime_gate": config.pb3_regime_gate,
        "pb3_regime_gate_applied": False,
    }
    if sleeve.id == "pullback_p0_k0_pb3_cp1_rv0" and config.pb3_regime_gate != "none":
        if config.pb3_regime_gate != "aged_non_accel_or_chaos":
            raise ValueError(f"unknown PB3 regime gate: {config.pb3_regime_gate}")
        pb3_gate = build_pb3_regime_gate_frame(
            bull_trigger_pct=config.amv_bull_trigger_pct,
            bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
            bull_lookback_days=config.amv_bull_lookback_days,
            effective_lag_days=config.amv_effective_lag_days,
        )
        before_rows = signal_rows.height
        before_days = signal_rows.select("signal_date").n_unique()
        signal_rows = (
            signal_rows.join(pb3_gate, on="signal_date", how="left")
            .with_columns(
                [
                    pl.col("pb3_gate_skip").fill_null(False),
                    pl.col("pb3_gate_aged_non_accel").fill_null(False),
                    pl.col("pb3_gate_chaos").fill_null(False),
                ]
            )
        )
        blocked = signal_rows.filter(pl.col("pb3_gate_skip"))
        signal_rows = signal_rows.filter(~pl.col("pb3_gate_skip")).sort(["signal_date", "rank", "code"])
        export = build_backtest_signal_frame(market, signal_rows)
        gating_summary = {
            "pb3_regime_gate": config.pb3_regime_gate,
            "pb3_regime_gate_applied": True,
            "pb3_gate_timing": "signal_date_close_before_t_plus_1_open",
            "pb3_gate_rows_before": before_rows,
            "pb3_gate_rows_after": signal_rows.height,
            "pb3_gate_rows_blocked": blocked.height,
            "pb3_gate_days_before": before_days,
            "pb3_gate_days_after": signal_rows.select("signal_date").n_unique(),
            "pb3_gate_days_blocked": blocked.select("signal_date").n_unique(),
            "pb3_gate_aged_non_accel_rows": int(blocked["pb3_gate_aged_non_accel"].sum()),
            "pb3_gate_chaos_rows": int(blocked["pb3_gate_chaos"].sum()),
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
    sleeve_id = _context_sleeve_id(config)

    market = build_market_frame(context_args)
    sector_features = build_sector_features(context_args)
    trend_features = add_medium_trend_features(market)
    scored_base = market.join(sector_features, on=["date", "code"], how="left").join(
        trend_features,
        on=["date", "code"],
        how="left",
    )

    required_cols = ranker_required_columns(sleeve.ranker)
    candidate_expr = base_candidate_expr(
        required_cols,
        mv_min=config.mv_min,
        amount_ma20_min=config.amount_ma20_min,
        require_bull_regime=True,
    )

    sector_rank = sector_rank_expr(context_args).fill_null(1.0)
    sector_bottom_distance = (config.bottom_rank_threshold - sector_rank) / config.bottom_rank_threshold
    sector_bottom_strength = pl.when(sector_bottom_distance > 0.0).then(sector_bottom_distance).otherwise(0.0)
    sector_confirm = relative_confirm_expr(context_args)
    if config.sector_penalty_mode == "linear":
        sector_penalty_expr = pl.when(sector_confirm).then(sector_bottom_strength * config.sector_penalty).otherwise(
            0.0
        )
    elif config.sector_penalty_mode == "bucket":
        sector_penalty_expr = (
            pl.when((sector_rank < config.bottom_rank_threshold) & sector_confirm)
            .then(config.sector_penalty)
            .otherwise(0.0)
        )
    else:
        raise ValueError(f"unknown sector penalty mode: {config.sector_penalty_mode}")

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
        medium_penalty_expr = medium_strength * config.medium_penalty
    elif config.medium_penalty_mode == "bucket":
        medium_penalty_expr = pl.when(medium_weak).then(config.medium_penalty).otherwise(0.0)
    else:
        raise ValueError(f"unknown medium penalty mode: {config.medium_penalty_mode}")

    scored = (
        scored_base.with_columns(candidate_expr.alias("_is_signal_candidate"))
        .with_columns(
            [
                ranker_score_expr(sleeve.ranker).alias("_base_signal_score"),
                sector_rank.alias("_sector_rank_score"),
                sector_confirm.alias("_relative_confirm"),
                sector_penalty_expr.alias("_sector_penalty"),
                medium_weak.alias("_medium_weak"),
                medium_strength.alias("_medium_weak_strength"),
                medium_penalty_expr.alias("_medium_penalty"),
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

    return write_signal_artifact(
        export=export,
        selected=signal_rows,
        artifact_config=SignalArtifactConfig(
            sleeve_id=sleeve_id,
            model_name="static_factor_sleeve_context_combo",
            output_root=config.output_root,
            config=_context_config_payload(config, sleeve_id),
            summary=summary,
        ),
        started_at=started,
        repo_root=repo_root,
    )
