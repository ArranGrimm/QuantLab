from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.factors.limit_ecology import add_limit_ecology_features, load_raw_daily
from strategies.amv.market import build_market_frame
from strategies.amv.regime import build_amv_phase_frame
from strategies.amv.scoring import finite_expr


def bool_score(col_name: str, weight: float) -> pl.Expr:
    return pl.when(pl.col(col_name).fill_null(False)).then(weight).otherwise(0.0)


def limit_first_board_score_expr() -> pl.Expr:
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
    liquidity_bonus = pl.when(pl.col("amount_ratio_5_20").fill_null(1.0) <= 1.20).then(0.5).otherwise(0.0)
    risk_penalty = (
        pl.col("atr_14_pct_rank_pct").fill_null(0.5) + pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    ) / 2.0
    return base_score + recency_bonus + first_board_bonus + liquidity_bonus - risk_penalty


def build_limit_ecology_market(config: Any, market_args: Any) -> pl.DataFrame:
    market = build_market_frame(market_args)
    raw_daily = load_raw_daily(market_args)
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


def add_weak_window_context(scored: pl.DataFrame, config: Any) -> pl.DataFrame:
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


def limit_config_payload(config: Any, sleeve_id: str) -> dict[str, Any]:
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


def build_limit_weakgate_signal(
    *,
    market: pl.DataFrame,
    sleeve_id: str,
    config: Any,
) -> tuple[pl.DataFrame, dict[str, Any]]:
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
            pl.when(candidate_expr).then(limit_first_board_score_expr()).otherwise(None).alias("_base_signal_score"),
        ]
    ).with_columns(
        pl.col("_base_signal_score").rank(method="ordinal", descending=True).over("date").alias("_base_signal_rank")
    )
    scored = add_weak_window_context(scored, config)
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
    candidate_rows = scored.filter(pl.col("_is_signal_candidate"))
    summary = {
        "sleeve_id": sleeve_id,
        "candidate_rows_before_shift": candidate_rows.height,
        "candidate_days_before_shift": candidate_rows.select("date").n_unique(),
        "weak_window_signal_rows": int(signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).height),
        "weak_window_signal_days": int(
            signal_rows.filter(pl.col("_weak_window_score").fill_null(0.0) >= 3.0).select("signal_date").n_unique()
        ),
    }
    return signal_rows, summary
