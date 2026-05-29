"""Export AMV limit-up ecology event sleeves.

This is the first Stage-3 signal export. It intentionally uses only currently
available daily raw OHLCV ecology features for candidate/event definitions, then
exports both adjusted factor prices and raw execution prices for the Rust engine.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import (
    DEFAULT_QMT_DB,
    ROOT,
    _finite_expr,
    _git_commit,
    _rel_path,
    build_feature_frame,
)
from scripts.amv_limit_ecology_diagnostic import (
    LIMIT_TOLERANCE,
    add_limit_ecology_features,
    load_raw_daily,
)
from scripts.amv_medium_trend_quality_diagnostic import add_medium_trend_features
from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from scripts.amv_static_sleeve_signal_export import timestamp_token


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"
SLEEVE_IDS = [
    "limit_reboard_reclaim",
    "limit_recent_lu_ranked",
    "limit_first_board_pullback",
    "limit_first_board_pullback_dry",
    "limit_first_board_pullback_lowvol",
    "limit_first_board_pullback_no_failed",
    "limit_first_board_pullback_quality",
    "limit_first_board_pullback_atrpen1",
    "limit_first_board_pullback_atrpen2",
    "limit_first_board_pullback_medium128pen",
    "limit_first_board_pullback_staleqpen",
    "limit_first_board_pullback_riskmix",
    "limit_first_board_pullback_weakgate",
    "limit_first_board_pullback_weaktop1",
    "limit_first_board_pullback_weaktier",
    "limit_first_board_pullback_weakscorepen",
]

WEAK_WINDOW_SLEEVES = {
    "limit_first_board_pullback_weakgate",
    "limit_first_board_pullback_weaktop1",
    "limit_first_board_pullback_weaktier",
    "limit_first_board_pullback_weakscorepen",
}


def parse_sleeves(value: str) -> list[str]:
    sleeves = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(sleeves) - set(SLEEVE_IDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown sleeves: {', '.join(unknown)}")
    if not sleeves:
        raise argparse.ArgumentTypeError("sleeves must not be empty")
    return sleeves


def bool_score(col_name: str, weight: float) -> pl.Expr:
    return pl.when(pl.col(col_name).fill_null(False)).then(weight).otherwise(0.0)


def needs_medium128_features(sleeves: list[str]) -> bool:
    return any(
        sleeve_id
        in {
            "limit_first_board_pullback_medium128pen",
            "limit_first_board_pullback_staleqpen",
            "limit_first_board_pullback_riskmix",
        }
        for sleeve_id in sleeves
    )


def needs_weak_window_context(sleeves: list[str]) -> bool:
    return any(sleeve_id in WEAK_WINDOW_SLEEVES for sleeve_id in sleeves)


def sleeve_candidate_and_score(sleeve_id: str) -> tuple[pl.Expr, pl.Expr]:
    recent_event = pl.col("has_limit_up_20d").fill_null(False)
    reboard_or_reclaim = (
        pl.col("is_reboard_after_pullback").fill_null(False)
        | pl.col("is_reclaim_after_limit").fill_null(False)
    )
    first_board_pullback = pl.col("is_first_board_pullback_setup").fill_null(False)
    dry_pullback = first_board_pullback & (pl.col("amount_ratio_5_20").fill_null(1.0) <= 0.85)
    low_vol_pullback = (
        first_board_pullback
        & (pl.col("atr_14_pct_rank_pct").fill_null(1.0) <= 0.67)
        & (pl.col("panic_vol_ratio_20d_rank_pct").fill_null(1.0) <= 0.67)
    )
    no_failed_pullback = first_board_pullback & ~pl.col("has_failed_limit_up_5d").fill_null(False)
    quality_pullback = (
        dry_pullback
        & low_vol_pullback
        & ~pl.col("has_failed_limit_up_5d").fill_null(False)
    )

    base_score = (
        bool_score("is_reboard_after_pullback", 5.0)
        + bool_score("was_limit_up_yesterday", 4.0)
        + bool_score("is_reclaim_after_limit", 3.0)
        + bool_score("has_limit_up_5d", 2.0)
        + bool_score("has_limit_up_10d", 1.5)
        + bool_score("has_limit_up_20d", 1.0)
        + bool_score("has_one_word_limit_up_10d", 0.5)
        - bool_score("has_failed_limit_up_5d", 1.0)
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
    liquidity_bonus = (
        pl.when(pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.20)
        .then(0.5)
        .otherwise(0.0)
    )
    risk_penalty = (
        pl.col("atr_14_pct_rank_pct").fill_null(0.5)
        + pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    ) / 2.0
    score = base_score + recency_bonus + first_board_bonus + liquidity_bonus - risk_penalty

    atr_rank = pl.col("atr_14_pct_rank_pct").fill_null(0.5)
    high_atr_strength = ((atr_rank - 0.80) / 0.20).clip(lower_bound=0.0, upper_bound=1.0)
    structure128 = pl.col("structure_score_128d").fill_null(1.0)
    quality128 = pl.col("trend_quality_score_128d").fill_null(1.0)
    medium_weak = (structure128 < 0.50) & (quality128 < 0.50)
    medium_strength = (
        pl.when(medium_weak)
        .then((((0.50 - structure128) / 0.50) + ((0.50 - quality128) / 0.50)) / 2.0)
        .otherwise(0.0)
        .clip(lower_bound=0.0, upper_bound=1.0)
    )
    stale_strength = (
        pl.when(pl.col("days_since_prior_limit_up").fill_null(0) >= 7)
        .then(((pl.col("days_since_prior_limit_up").fill_null(7) - 6) / 4.0).clip(0.0, 1.0))
        .otherwise(0.0)
    )
    stale_quality_strength = (
        stale_strength * ((0.67 - quality128) / 0.67).clip(lower_bound=0.0, upper_bound=1.0)
    )

    if sleeve_id == "limit_reboard_reclaim":
        candidate = recent_event & reboard_or_reclaim
        return candidate, score
    if sleeve_id == "limit_recent_lu_ranked":
        candidate = recent_event
        return candidate, score
    if sleeve_id == "limit_first_board_pullback":
        candidate = first_board_pullback
        return candidate, score
    if sleeve_id == "limit_first_board_pullback_dry":
        return dry_pullback, score
    if sleeve_id == "limit_first_board_pullback_lowvol":
        return low_vol_pullback, score
    if sleeve_id == "limit_first_board_pullback_no_failed":
        return no_failed_pullback, score
    if sleeve_id == "limit_first_board_pullback_quality":
        return quality_pullback, score
    if sleeve_id == "limit_first_board_pullback_atrpen1":
        return first_board_pullback, score - high_atr_strength
    if sleeve_id == "limit_first_board_pullback_atrpen2":
        return first_board_pullback, score - high_atr_strength * 2.0
    if sleeve_id == "limit_first_board_pullback_medium128pen":
        return first_board_pullback, score - medium_strength * 2.0
    if sleeve_id == "limit_first_board_pullback_staleqpen":
        return first_board_pullback, score - stale_quality_strength * 2.0
    if sleeve_id == "limit_first_board_pullback_riskmix":
        return (
            first_board_pullback,
            score - high_atr_strength * 1.5 - stale_quality_strength - medium_strength * 0.5,
        )
    if sleeve_id in WEAK_WINDOW_SLEEVES:
        return first_board_pullback, score
    raise ValueError(f"unknown sleeve_id: {sleeve_id}")


def build_limit_ecology_market(args: argparse.Namespace) -> pl.DataFrame:
    market = build_feature_frame(args)
    raw_daily = load_raw_daily(args)
    ecology = add_limit_ecology_features(raw_daily, tolerance=args.price_limit_tolerance)
    return market.join(ecology, on=["date", "code"], how="left").with_columns(
        [
            (pl.col("atr_14_pct").rank("average").over("date") / pl.len().over("date")).alias(
                "atr_14_pct_rank_pct"
            ),
            (
                pl.col("panic_vol_ratio_20d").rank("average").over("date")
                / pl.len().over("date")
            ).alias("panic_vol_ratio_20d_rank_pct"),
        ]
    )


def add_medium128_if_needed(market: pl.DataFrame, sleeves: list[str]) -> pl.DataFrame:
    if not needs_medium128_features(sleeves):
        return market
    medium = add_medium_trend_features(market).select(
        [
            "date",
            "code",
            "structure_score_64d",
            "trend_quality_score_64d",
            "structure_score_128d",
            "trend_quality_score_128d",
            "ret_128d",
            "pos_128d",
            "trend_eff_128d",
            "ret_vol_128d",
        ]
    )
    return market.join(medium, on=["date", "code"], how="left")


def add_weak_window_context(
    scored: pl.DataFrame,
    *,
    args: argparse.Namespace,
) -> pl.DataFrame:
    market_breadth = (
        scored.group_by("date")
        .agg(pl.col("is_close_limit_up").fill_null(False).sum().alias("_weak_limit_up_count"))
        .sort("date")
        .with_columns(
            (pl.col("_weak_limit_up_count").rank("average") / pl.len()).alias("_weak_limit_up_count_rank_pct")
        )
    )
    candidates = scored.filter(pl.col("_is_signal_candidate"))
    top3 = candidates.filter(pl.col("_base_signal_rank") <= args.top_n)
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
        bull_trigger_pct=args.amv_bull_trigger_pct,
        bear_trigger_1d_pct=args.amv_bear_trigger_1d_pct,
        bull_lookback_days=args.amv_bull_lookback_days,
        effective_lag_days=args.amv_effective_lag_days,
    ).select(
        [
            "date",
            pl.col("amv_slope_5d").alias("_weak_amv_slope_5d"),
            pl.col("amv_dd_from_high").alias("_weak_amv_dd_from_high"),
            pl.col("amv_neg_streak").alias("_weak_amv_neg_streak"),
        ]
    )
    enriched = (
        scored.join(market_breadth, on="date", how="left")
        .join(candidate_health, on="date", how="left")
        .join(amv_phase, on="date", how="left")
    )
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


def build_one_signal(
    market: pl.DataFrame,
    *,
    sleeve_id: str,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    event_candidate, score_expr = sleeve_candidate_and_score(sleeve_id)
    valid_expr = _finite_expr("price_pos_20d") & _finite_expr("amount_ma20")
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= args.mv_min)
        & (pl.col("amount_ma20") >= args.amount_ma20_min)
        & valid_expr
        & event_candidate
    )
    scored = market.with_columns(
        [
            candidate_expr.alias("_is_signal_candidate"),
            pl.when(candidate_expr).then(score_expr).otherwise(None).alias("_base_signal_score"),
        ]
    ).with_columns(
        pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_signal_rank")
    )
    if sleeve_id in WEAK_WINDOW_SLEEVES:
        scored = add_weak_window_context(scored, args=args)
    if sleeve_id == "limit_first_board_pullback_weakscorepen":
        candidate_atr_strength = (
            (pl.col("atr_14_pct_rank_pct").fill_null(0.75) - 0.80) / 0.20
        ).clip(lower_bound=0.0, upper_bound=1.0)
        candidate_stale_strength = (
            pl.when(pl.col("days_since_prior_limit_up").fill_null(0) >= 7)
            .then(((pl.col("days_since_prior_limit_up").fill_null(7) - 6) / 4.0).clip(0.0, 1.0))
            .otherwise(0.0)
        )
        weak_candidate_penalty = (
            pl.when(pl.col("_weak_window_score").fill_null(0.0) >= 2.0)
            .then(
                candidate_atr_strength * 1.5
                + candidate_stale_strength * 0.75
                + pl.when(pl.col("is_reclaim_after_limit").fill_null(False)).then(0.0).otherwise(0.5)
            )
            .otherwise(0.0)
        )
        scored = scored.with_columns((pl.col("_base_signal_score") - weak_candidate_penalty).alias("_signal_score"))
    else:
        scored = scored.with_columns(pl.col("_base_signal_score").alias("_signal_score"))
    scored = scored.with_columns(
        pl.col("_signal_score").rank(method="ordinal", descending=True).over("date").alias("_signal_rank")
    )
    weak_score = pl.col("_weak_window_score").fill_null(0.0)
    if sleeve_id == "limit_first_board_pullback_weakgate":
        select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n) & (weak_score < 3.0)
    elif sleeve_id == "limit_first_board_pullback_weaktop1":
        select_expr = pl.col("_is_signal_candidate") & (
            ((weak_score >= 3.0) & (pl.col("_signal_rank") <= 1))
            | ((weak_score < 3.0) & (pl.col("_signal_rank") <= args.top_n))
        )
    elif sleeve_id == "limit_first_board_pullback_weaktier":
        select_expr = pl.col("_is_signal_candidate") & (
            ((weak_score >= 3.0) & (pl.col("_signal_rank") <= 1))
            | ((weak_score >= 2.4) & (weak_score < 3.0) & (pl.col("_signal_rank") <= 2))
            | ((weak_score < 2.4) & (pl.col("_signal_rank") <= args.top_n))
        )
    else:
        select_expr = pl.col("_is_signal_candidate") & (pl.col("_signal_rank") <= args.top_n)
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
    ]
    selected_cols.extend(
        col
        for col in [
            "structure_score_64d",
            "trend_quality_score_64d",
            "structure_score_128d",
            "trend_quality_score_128d",
            "ret_128d",
            "pos_128d",
            "trend_eff_128d",
            "ret_vol_128d",
        ]
        if col in scored.columns
    )
    selected_cols.extend(
        col
        for col in [
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
        if col in scored.columns
    )
    signal_rows = (
        scored.filter(select_expr)
        .select(selected_cols)
        .sort(["signal_date", "rank", "code"])
    )

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls(
        "execution_date"
    )
    execution_signals = (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(
            [
                pl.col("execution_date").alias("date"),
                "code",
                "signal_date",
                "sleeve_id",
                "score",
                "rank",
            ]
        )
    )
    export = (
        market.select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "open_raw",
                "high_raw",
                "low_raw",
                "close_raw",
                "pre_close_raw",
                "is_bull_regime",
                "amv_mechanical_regime",
                "market_cap_100m",
                "amount_ma20",
            ]
        )
        .join(execution_signals, on=["date", "code"], how="left")
        .with_columns(
            [
                pl.col("signal_date").is_not_null().alias("is_signal"),
                pl.col("score").fill_null(0.0),
                pl.col("rank").fill_null(9999).cast(pl.UInt32),
                pl.col("sleeve_id").fill_null(""),
            ]
        )
        .sort(["date", "code"])
    )
    summary = {
        "sleeve_id": sleeve_id,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "unique_codes": export["code"].n_unique(),
        "candidate_rows_before_shift": scored.filter(pl.col("_is_signal_candidate")).height,
        "candidate_days_before_shift": scored.filter(pl.col("_is_signal_candidate")).select("date").n_unique(),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "weak_window_signal_rows": (
            int(signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).height)
            if "_weak_window_score" in signal_rows.columns
            else 0
        ),
        "weak_window_signal_days": (
            int(signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).select("signal_date").n_unique())
            if "_weak_window_score" in signal_rows.columns
            else 0
        ),
        "signals_blocked_by_execution_bear_regime": int(
            export.filter(pl.col("is_signal") & ~pl.col("is_bull_regime")).height
        ),
        "score_min": float(signal_rows["score"].min()) if signal_rows.height else None,
        "score_max": float(signal_rows["score"].max()) if signal_rows.height else None,
    }
    return export, signal_rows, summary


def write_one_signal(
    output_root: Path,
    *,
    sleeve_id: str,
    export: pl.DataFrame,
    selected: pl.DataFrame,
    summary: dict[str, Any],
    args: argparse.Namespace,
    started_at: datetime,
) -> Path:
    output_dir = output_root / f"{timestamp_token()}_{sleeve_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    selected_path = output_dir / "selected_signals.csv"
    meta_path = output_dir / "signal.meta.json"
    export.write_parquet(signal_path)
    selected.write_csv(selected_path)

    meta = {
        "strategy": "amv_limit_ecology_event_sleeve",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_limit_ecology_event_sleeve_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"limit_ecology:{sleeve_id}",
        "model_name": "daily_raw_limit_ecology_event",
        "feature_mode": sleeve_id,
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "sleeve_id": sleeve_id,
            "top_n": args.top_n,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "price_limit_tolerance": args.price_limit_tolerance,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "execution_price_note": "Exports raw OHLC/pre-close for Rust execution while keeping adjusted OHLC for factor compatibility.",
        },
        "summary": summary,
        "files": {
            "signal": _rel_path(signal_path, output_dir),
            "selected_signals": _rel_path(selected_path, output_dir),
        },
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMV limit ecology event sleeve signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sleeves", type=parse_sleeves, default=["limit_reboard_reclaim"])
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building limit ecology event sleeve exports...")
    market = add_medium128_if_needed(build_limit_ecology_market(args), args.sleeves)
    output_paths: list[str] = []
    for sleeve_id in args.sleeves:
        print(f"Exporting sleeve: {sleeve_id}")
        export, selected, summary = build_one_signal(market, sleeve_id=sleeve_id, args=args)
        meta_path = write_one_signal(
            args.output_root,
            sleeve_id=sleeve_id,
            export=export,
            selected=selected,
            summary=summary,
            args=args,
            started_at=started_at,
        )
        output_paths.append(str(meta_path))
        print(
            f"  -> {meta_path} | signal rows={summary['signal_rows_after_shift']:,}, "
            f"days={summary['signal_days_after_shift']:,}, candidates={summary['candidate_rows_before_shift']:,}"
        )
    print(json.dumps({"outputs": output_paths}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
