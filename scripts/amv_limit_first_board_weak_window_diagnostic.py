"""Diagnose observable weak-window context for first-board pullback sleeve."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT, _finite_expr
from scripts.amv_limit_ecology_drawdown_attribution import (
    BASE_SIGNAL,
    add_feature_buckets,
    feature_args,
    load_enriched_trades,
    load_feature_lookup,
    summarize_trades,
    top_records,
)
from scripts.amv_limit_ecology_signal_export import build_limit_ecology_market, sleeve_candidate_and_score
from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from strategies.amv.factors.market_sentiment import build_market_sentiment_features


DEFAULT_OUTPUT = ROOT / "reports" / "amv_limit_first_board_weak_window_diagnostic.json"
BASE_BACKTEST = "5td_static_strict_top3_no_stop_20260529_111655_limit_first_board_pullback"
DRAWDOWN_START = "2023-08-03"
DRAWDOWN_END = "2024-02-05"


def safe_mean(df: pl.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.height == 0:
        return None
    value = df[column].mean()
    return None if value is None else round(float(value), 6)


def safe_median(df: pl.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.height == 0:
        return None
    value = df[column].median()
    return None if value is None else round(float(value), 6)


def group_profile(df: pl.DataFrame, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "summary": summarize_trades(df),
        "context_means": {
            "amv_ret_ma5": safe_mean(df, "amv_ret_ma5"),
            "amv_slope_5d": safe_mean(df, "amv_slope_5d"),
            "amv_dd_from_high": safe_mean(df, "amv_dd_from_high"),
            "amplitude_ma3": safe_mean(df, "amplitude_ma3"),
            "amv_neg_streak": safe_mean(df, "amv_neg_streak"),
            "market_up_ratio_rank_pct": safe_mean(df, "market_up_ratio_rank_pct"),
            "limit_up_count_rank_pct": safe_mean(df, "limit_up_count_rank_pct"),
            "limit_down_count_rank_pct": safe_mean(df, "limit_down_count_rank_pct"),
            "failed_limit_up_ratio_rank_pct": safe_mean(df, "failed_limit_up_ratio_rank_pct"),
            "new_high_20_ratio_rank_pct": safe_mean(df, "new_high_20_ratio_rank_pct"),
            "yday_limit_up_close_premium_rank_pct": safe_mean(df, "yday_limit_up_close_premium_rank_pct"),
            "strong_stock_drawdown_20d_median_rank_pct": safe_mean(df, "strong_stock_drawdown_20d_median_rank_pct"),
            "candidate_count": safe_mean(df, "candidate_count"),
            "candidate_top3_avg_score": safe_mean(df, "candidate_top3_avg_score"),
            "candidate_top3_avg_atr_rank": safe_mean(df, "candidate_top3_avg_atr_rank"),
            "candidate_top3_stale_share": safe_mean(df, "candidate_top3_stale_share"),
            "candidate_top3_reclaim_share": safe_mean(df, "candidate_top3_reclaim_share"),
            "candidate_top3_failed_share": safe_mean(df, "candidate_top3_failed_share"),
        },
        "context_medians": {
            "amv_ret_ma5": safe_median(df, "amv_ret_ma5"),
            "amv_slope_5d": safe_median(df, "amv_slope_5d"),
            "amv_dd_from_high": safe_median(df, "amv_dd_from_high"),
            "market_up_ratio_rank_pct": safe_median(df, "market_up_ratio_rank_pct"),
            "yday_limit_up_close_premium_rank_pct": safe_median(df, "yday_limit_up_close_premium_rank_pct"),
            "candidate_top3_avg_atr_rank": safe_median(df, "candidate_top3_avg_atr_rank"),
            "candidate_top3_stale_share": safe_median(df, "candidate_top3_stale_share"),
        },
    }


def summarize_by(df: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    rows = []
    for row in (
        df.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
                pl.col("candidate_top3_avg_atr_rank").mean().alias("avg_top3_atr"),
            ]
        )
        .sort("pnl")
        .iter_rows(named=True)
    ):
        rows.append(
            {
                column: str(row[column]),
                "trades": int(row["trades"]),
                "pnl": round(float(row["pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
                "avg_top3_atr": round(float(row["avg_top3_atr"]), 6) if row["avg_top3_atr"] is not None else None,
            }
        )
    return rows


def rule_summary(df: pl.DataFrame, name: str, expr: pl.Expr) -> dict[str, Any]:
    flagged = df.filter(expr)
    kept = df.filter(~expr)
    return {
        "rule": name,
        "flagged": summarize_trades(flagged),
        "kept": summarize_trades(kept),
        "trade_level_delta_if_gated": round(-float(flagged["pnl"].sum()), 2) if flagged.height else 0.0,
        "flagged_big_winners_gt_20k": int(flagged.filter(pl.col("pnl") > 20_000).height),
        "flagged_big_losers_lt_minus_20k": int(flagged.filter(pl.col("pnl") < -20_000).height),
        "drawdown_flagged_trades": int(flagged.filter(pl.col("is_drawdown_window")).height),
        "worst_flagged_trades": top_records(flagged, by="pnl", descending=False, n=8),
    }


def build_candidate_health(market: pl.DataFrame) -> pl.DataFrame:
    event_candidate, score_expr = sleeve_candidate_and_score("limit_first_board_pullback")
    valid_expr = _finite_expr("price_pos_20d") & _finite_expr("amount_ma20")
    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= 100.0)
        & (pl.col("amount_ma20") >= 5e7)
        & valid_expr
        & event_candidate
    )
    scored = (
        market.with_columns(
            [
                candidate_expr.alias("_is_candidate"),
                pl.when(candidate_expr).then(score_expr).otherwise(None).alias("_score"),
            ]
        )
        .with_columns(pl.col("_score").rank(method="ordinal", descending=True).over("date").alias("_rank"))
    )
    candidates = scored.filter(pl.col("_is_candidate"))
    top3 = candidates.filter(pl.col("_rank") <= 3)
    return (
        candidates.group_by("date")
        .agg(
            [
                pl.len().alias("candidate_count"),
                pl.col("_score").mean().alias("candidate_avg_score"),
                pl.col("atr_14_pct_rank_pct").mean().alias("candidate_avg_atr_rank"),
                (pl.col("days_since_prior_limit_up") >= 7).mean().alias("candidate_stale_share"),
            ]
        )
        .join(
            top3.group_by("date").agg(
                [
                    pl.col("_score").mean().alias("candidate_top3_avg_score"),
                    pl.col("_score").min().alias("candidate_top3_min_score"),
                    pl.col("atr_14_pct_rank_pct").mean().alias("candidate_top3_avg_atr_rank"),
                    (pl.col("atr_14_pct_rank_pct") > 0.85).mean().alias("candidate_top3_high_atr_share"),
                    (pl.col("days_since_prior_limit_up") >= 7).mean().alias("candidate_top3_stale_share"),
                    pl.col("is_reclaim_after_limit").mean().alias("candidate_top3_reclaim_share"),
                    pl.col("is_reboard_after_pullback").mean().alias("candidate_top3_reboard_share"),
                    pl.col("has_failed_limit_up_5d").mean().alias("candidate_top3_failed_share"),
                    pl.col("amount_ratio_5_20").mean().alias("candidate_top3_amount_ratio"),
                ]
            ),
            on="date",
            how="left",
        )
        .sort("date")
    )


def add_context_buckets(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.when(pl.col("market_up_ratio_rank_pct").fill_null(0.5) < 0.33)
                .then(pl.lit("breadth_low"))
                .when(pl.col("market_up_ratio_rank_pct").fill_null(0.5) < 0.67)
                .then(pl.lit("breadth_mid"))
                .otherwise(pl.lit("breadth_high"))
            ).alias("market_breadth_bucket"),
            (
                pl.when(pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.33)
                .then(pl.lit("yday_premium_low"))
                .when(pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.67)
                .then(pl.lit("yday_premium_mid"))
                .otherwise(pl.lit("yday_premium_high"))
            ).alias("yday_lu_premium_bucket"),
            (
                pl.when(pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.5) > 0.67)
                .then(pl.lit("failed_lu_high"))
                .when(pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.5) > 0.33)
                .then(pl.lit("failed_lu_mid"))
                .otherwise(pl.lit("failed_lu_low"))
            ).alias("failed_lu_bucket"),
            (
                pl.when(pl.col("candidate_top3_avg_atr_rank").fill_null(0.5) > 0.85)
                .then(pl.lit("top3_atr_high"))
                .when(pl.col("candidate_top3_avg_atr_rank").fill_null(0.5) > 0.75)
                .then(pl.lit("top3_atr_mid"))
                .otherwise(pl.lit("top3_atr_low"))
            ).alias("candidate_top3_atr_bucket"),
            (
                pl.when(pl.col("fwd_duration_bucket").fill_null("unknown").is_in(["aged", "old"]))
                .then(pl.lit("amv_aged_old"))
                .otherwise(pl.lit("amv_fresh_young"))
            ).alias("amv_age_bucket"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose first-board pullback weak market windows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_args = feature_args()
    trades = add_feature_buckets(load_enriched_trades(BASE_SIGNAL, BASE_BACKTEST, load_feature_lookup()))
    market = build_limit_ecology_market(base_args)
    candidate_health = build_candidate_health(market)
    sentiment = build_market_sentiment_features(base_args)
    amv_phase = build_amv_phase_frame(
        bull_trigger_pct=base_args.amv_bull_trigger_pct,
        bear_trigger_1d_pct=base_args.amv_bear_trigger_1d_pct,
        bull_lookback_days=base_args.amv_bull_lookback_days,
        effective_lag_days=base_args.amv_effective_lag_days,
    ).select(
        [
            pl.col("date").alias("signal_date"),
            "fwd_duration_bucket",
            "fwd_momentum_bucket",
            "fwd_phase",
            "amv_ret_ma5",
            "amv_slope_5d",
            "amv_slope_20d",
            "amv_acceleration",
            "amv_neg_streak",
            "amv_dd_from_high",
            "amplitude_pct",
            "amplitude_ma3",
            "regime_duration_days",
            "regime_maturity",
        ]
    )
    enriched = (
        trades.join(sentiment, left_on="signal_date", right_on="date", how="left")
        .join(candidate_health.rename({"date": "signal_date"}), on="signal_date", how="left")
        .join(amv_phase, on="signal_date", how="left")
        .with_columns(
            [
                ((pl.col("exit_date") >= pl.date(2023, 8, 3)) & (pl.col("exit_date") <= pl.date(2024, 2, 5))).alias(
                    "is_drawdown_window"
                ),
                (pl.col("pnl") < -20_000).alias("is_big_loser"),
                (pl.col("pnl") > 20_000).alias("is_big_winner"),
            ]
        )
        .pipe(add_context_buckets)
    )

    drawdown = enriched.filter(pl.col("is_drawdown_window"))
    non_drawdown = enriched.filter(~pl.col("is_drawdown_window"))
    big_losers = enriched.filter(pl.col("is_big_loser"))
    big_winners = enriched.filter(pl.col("is_big_winner"))

    rules = [
        (
            "weak_amv_retreating_or_stalling",
            pl.col("fwd_momentum_bucket").fill_null("") .is_in(["retreating", "stalling"]),
        ),
        (
            "weak_amv_aged_retreating",
            (pl.col("fwd_duration_bucket").fill_null("").is_in(["aged", "old"]))
            & (pl.col("fwd_momentum_bucket").fill_null("").is_in(["retreating", "stalling"])),
        ),
        (
            "weak_poor_yday_lu_premium",
            pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.33,
        ),
        (
            "weak_bad_board_ecology",
            (pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.5) > 0.67)
            | (pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.33),
        ),
        (
            "weak_candidate_fragile",
            (pl.col("candidate_top3_avg_atr_rank").fill_null(0.0) > 0.85)
            & (pl.col("candidate_top3_stale_share").fill_null(0.0) >= 0.5),
        ),
        (
            "weak_ecology_and_candidate_fragile",
            (
                (pl.col("failed_limit_up_ratio_rank_pct").fill_null(0.5) > 0.67)
                | (pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.33)
            )
            & (pl.col("candidate_top3_avg_atr_rank").fill_null(0.0) > 0.80),
        ),
        (
            "weak_amv_or_ecology_plus_candidate",
            (
                pl.col("fwd_momentum_bucket").fill_null("").is_in(["retreating", "stalling"])
                | (pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.5) < 0.33)
            )
            & (pl.col("candidate_top3_avg_atr_rank").fill_null(0.0) > 0.80),
        ),
        (
            "weak_drawdown_like_candidate_pool",
            (pl.col("candidate_count").fill_null(99) <= 8)
            & (pl.col("candidate_top3_avg_score").fill_null(99.0) <= 6.5)
            & (pl.col("candidate_top3_avg_atr_rank").fill_null(0.0) >= 0.85),
        ),
        (
            "weak_amv_flat_and_pool_thin",
            (pl.col("amv_slope_5d").fill_null(99.0) <= 1.0)
            & (pl.col("candidate_count").fill_null(99) <= 8)
            & (pl.col("candidate_top3_stale_share").fill_null(0.0) >= 0.5),
        ),
        (
            "weak_lu_count_low_and_pool_bad",
            (pl.col("limit_up_count_rank_pct").fill_null(1.0) < 0.45)
            & (pl.col("candidate_top3_avg_score").fill_null(99.0) <= 6.5)
            & (pl.col("candidate_top3_avg_atr_rank").fill_null(0.0) > 0.80),
        ),
        (
            "weak_pool_stale_no_reclaim",
            (pl.col("candidate_count").fill_null(99) <= 8)
            & (pl.col("candidate_top3_stale_share").fill_null(0.0) >= 0.5)
            & (pl.col("candidate_top3_reclaim_share").fill_null(1.0) <= 0.34),
        ),
    ]

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "Forward-observable weak-window diagnostic for limit_first_board_pullback base 5td",
        "drawdown_window": f"{DRAWDOWN_START}..{DRAWDOWN_END} by exit_date",
        "profiles": [
            group_profile(enriched, "all_trades"),
            group_profile(drawdown, "drawdown_window"),
            group_profile(non_drawdown, "non_drawdown"),
            group_profile(big_losers, "big_losers_lt_minus_20k"),
            group_profile(big_winners, "big_winners_gt_20k"),
        ],
        "by_fwd_phase": summarize_by(enriched, "fwd_phase"),
        "by_market_breadth_bucket": summarize_by(enriched, "market_breadth_bucket"),
        "by_yday_lu_premium_bucket": summarize_by(enriched, "yday_lu_premium_bucket"),
        "by_failed_lu_bucket": summarize_by(enriched, "failed_lu_bucket"),
        "by_candidate_top3_atr_bucket": summarize_by(enriched, "candidate_top3_atr_bucket"),
        "by_amv_age_bucket": summarize_by(enriched, "amv_age_bucket"),
        "drawdown_feature_dates": (
            drawdown.select(
                [
                    "signal_date",
                    "entry_date",
                    "code",
                    "pnl",
                    "fwd_phase",
                    "amv_ret_ma5",
                    "amv_dd_from_high",
                    "yday_limit_up_close_premium_rank_pct",
                    "failed_limit_up_ratio_rank_pct",
                    "candidate_count",
                    "candidate_top3_avg_score",
                    "candidate_top3_avg_atr_rank",
                    "candidate_top3_stale_share",
                    "candidate_top3_reclaim_share",
                ]
            )
            .sort("pnl")
            .head(20)
            .to_dicts()
        ),
        "candidate_rules": [rule_summary(enriched, name, expr) for name, expr in rules],
        "interpretation": {
            "diagnostic_first": "Rules are not promoted unless forward-observable features distinguish the drawdown window or repeated loser clusters.",
            "next_step": "If a candidate rule has positive full-sample trade-level delta and captures drawdown losers without many big winners, promote it to a Rust gate/top1/downshift test.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
