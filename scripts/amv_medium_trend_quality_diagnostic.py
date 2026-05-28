"""Medium-term structure and trend-quality diagnostic for AMV sleeves.

Second-stage AMV context factors:
- 64/128 day medium-term structure
- trend quality, not just short-term proximity to new highs

This script is diagnostic only. It joins signal-date stock-level features to
executed trades and evaluates buckets / what-if skip rules.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT
from scripts.amv_market_sentiment_diagnostic import (
    DEFAULT_SLEEVES,
    load_trade_context,
    rule_summary,
    summarize_by,
    summarize_trades,
)
from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from scripts.amv_sector_tailwind_signal_export import build_market_frame


DEFAULT_OUTPUT = ROOT / "reports" / "amv_medium_trend_quality_diagnostic.json"


def _safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)


def add_medium_trend_features(market: pl.DataFrame) -> pl.DataFrame:
    market = market.sort(["code", "date"])
    stock = market.with_columns(
        [
            (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
            (pl.col("close_adj") > pl.col("pre_close_adj")).alias("is_up_day"),
            (
                (pl.col("close_adj") - pl.col("open_adj")).abs()
                / (pl.col("high_adj") - pl.col("low_adj")).abs().clip(lower_bound=1e-12)
            ).alias("body_efficiency_1d"),
        ]
    ).with_columns(
        [
            pl.col("ret_1d").abs().alias("abs_ret_1d"),
            pl.col("ret_1d").rolling_std(20).over("code").alias("vol_20d"),
            pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20_raw"),
            pl.col("amount").rolling_mean(60).over("code").alias("amount_ma60_raw"),
        ]
    )

    feature_exprs: list[pl.Expr] = []
    for window in (64, 128):
        high = pl.col("close_adj").rolling_max(window).over("code")
        low = pl.col("close_adj").rolling_min(window).over("code")
        ret = pl.col("close_adj") / pl.col("close_adj").shift(window).over("code") - 1.0
        vol = pl.col("ret_1d").rolling_std(window).over("code")
        path_len = pl.col("abs_ret_1d").rolling_sum(window).over("code")
        ma = pl.col("close_adj").rolling_mean(window).over("code")
        feature_exprs.extend(
            [
                ret.alias(f"ret_{window}d"),
                _safe_div(pl.col("close_adj") - low, high - low).alias(f"pos_{window}d"),
                (pl.col("close_adj") / high - 1.0).alias(f"dd_from_high_{window}d"),
                _safe_div(ret.abs(), path_len).alias(f"trend_eff_{window}d"),
                pl.col("is_up_day").rolling_mean(window).over("code").alias(f"up_ratio_{window}d"),
                _safe_div(ret, vol).alias(f"ret_vol_{window}d"),
                _safe_div(pl.col("vol_20d"), vol).alias(f"vol_contraction_20_{window}d"),
                (_safe_div(ma, ma.shift(20).over("code")) - 1.0).alias(f"ma_slope_{window}d"),
                pl.col("body_efficiency_1d").rolling_mean(window).over("code").alias(
                    f"body_efficiency_{window}d"
                ),
            ]
        )

    stock = stock.with_columns(feature_exprs)
    score_cols = []
    for window in (64, 128):
        rank_specs = [
            (f"ret_{window}d", True),
            (f"pos_{window}d", True),
            (f"trend_eff_{window}d", True),
            (f"up_ratio_{window}d", True),
            (f"ret_vol_{window}d", True),
            (f"vol_contraction_20_{window}d", False),
            (f"ma_slope_{window}d", True),
            (f"body_efficiency_{window}d", True),
        ]
        for col_name, higher_is_better in rank_specs:
            rank_expr = pl.col(col_name).rank("average", descending=not higher_is_better).over("date") / pl.len().over(
                "date"
            )
            stock = stock.with_columns(rank_expr.alias(f"{col_name}_rank_pct"))
        structure_score = (
            pl.col(f"ret_{window}d_rank_pct")
            + pl.col(f"pos_{window}d_rank_pct")
            + pl.col(f"ma_slope_{window}d_rank_pct")
        ) / 3.0
        quality_score = (
            pl.col(f"trend_eff_{window}d_rank_pct")
            + pl.col(f"up_ratio_{window}d_rank_pct")
            + pl.col(f"ret_vol_{window}d_rank_pct")
            + pl.col(f"body_efficiency_{window}d_rank_pct")
        ) / 4.0
        stock = stock.with_columns(
            [
                structure_score.alias(f"structure_score_{window}d"),
                quality_score.alias(f"trend_quality_score_{window}d"),
            ]
        )
        score_cols.extend([f"structure_score_{window}d", f"trend_quality_score_{window}d"])

    return stock.select(
        [
            "date",
            "code",
            "ret_64d",
            "pos_64d",
            "dd_from_high_64d",
            "trend_eff_64d",
            "up_ratio_64d",
            "ret_vol_64d",
            "vol_contraction_20_64d",
            "ma_slope_64d",
            "body_efficiency_64d",
            "structure_score_64d",
            "trend_quality_score_64d",
            "ret_128d",
            "pos_128d",
            "dd_from_high_128d",
            "trend_eff_128d",
            "up_ratio_128d",
            "ret_vol_128d",
            "vol_contraction_20_128d",
            "ma_slope_128d",
            "body_efficiency_128d",
            "structure_score_128d",
            "trend_quality_score_128d",
            *[f"{col}_rank_pct" for col in [
                "ret_64d",
                "pos_64d",
                "trend_eff_64d",
                "up_ratio_64d",
                "ret_vol_64d",
                "ma_slope_64d",
                "body_efficiency_64d",
                "ret_128d",
                "pos_128d",
                "trend_eff_128d",
                "up_ratio_128d",
                "ret_vol_128d",
                "ma_slope_128d",
                "body_efficiency_128d",
            ]],
        ]
    )


def build_medium_trend_features(args: argparse.Namespace) -> pl.DataFrame:
    return add_medium_trend_features(build_market_frame(args))


def add_buckets(df: pl.DataFrame) -> pl.DataFrame:
    def bucket(col_name: str, label: str) -> pl.Expr:
        value = pl.col(col_name).fill_null(0.5)
        return (
            pl.when(value < 0.33)
            .then(pl.lit(f"{label}_low"))
            .when(value < 0.67)
            .then(pl.lit(f"{label}_mid"))
            .otherwise(pl.lit(f"{label}_high"))
        )

    return df.with_columns(
        [
            bucket("structure_score_64d", "structure64").alias("structure64_bucket"),
            bucket("trend_quality_score_64d", "quality64").alias("quality64_bucket"),
            bucket("structure_score_128d", "structure128").alias("structure128_bucket"),
            bucket("trend_quality_score_128d", "quality128").alias("quality128_bucket"),
            bucket("pos_128d_rank_pct", "pos128").alias("pos128_bucket"),
            bucket("trend_eff_128d_rank_pct", "eff128").alias("eff128_bucket"),
            bucket("ret_vol_128d_rank_pct", "retvol128").alias("retvol128_bucket"),
        ]
    )


def analyze_sleeve(
    sleeve_key: str,
    sleeve: dict[str, str],
    features: pl.DataFrame,
    amv_phase: pl.DataFrame,
) -> dict[str, Any]:
    trades = load_trade_context(ROOT / sleeve["trades"], ROOT / sleeve["signals"])
    enriched = (
        trades.join(features, left_on=["signal_date", "code"], right_on=["date", "code"], how="left")
        .join(
            amv_phase.select(
                [
                    pl.col("date").alias("signal_date"),
                    "fwd_duration_bucket",
                    "fwd_momentum_bucket",
                    "fwd_phase",
                    "amv_neg_streak",
                    "amplitude_pct",
                ]
            ),
            on="signal_date",
            how="left",
        )
        .pipe(add_buckets)
    )

    rules = [
        ("skip_low_structure_128", pl.col("structure_score_128d").fill_null(0.0) < 0.33),
        ("skip_low_quality_128", pl.col("trend_quality_score_128d").fill_null(0.0) < 0.33),
        (
            "skip_high_pos_low_quality_128",
            (pl.col("pos_128d_rank_pct").fill_null(0.0) > 0.67)
            & (pl.col("trend_quality_score_128d").fill_null(0.0) < 0.50),
        ),
        (
            "skip_low_efficiency_128",
            pl.col("trend_eff_128d_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_low_retvol_128",
            pl.col("ret_vol_128d_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_medium_structure_and_quality_weak",
            (pl.col("structure_score_128d").fill_null(0.0) < 0.50)
            & (pl.col("trend_quality_score_128d").fill_null(0.0) < 0.50),
        ),
        (
            "skip_64_128_quality_both_low",
            (pl.col("trend_quality_score_64d").fill_null(0.0) < 0.33)
            & (pl.col("trend_quality_score_128d").fill_null(0.0) < 0.33),
        ),
    ]

    return {
        "sleeve": sleeve_key,
        "label": sleeve["label"],
        "total": summarize_trades(enriched),
        "missing_feature_trades": int(enriched.filter(pl.col("structure_score_128d").is_null()).height),
        "by_structure64_bucket": summarize_by(enriched, "structure64_bucket"),
        "by_quality64_bucket": summarize_by(enriched, "quality64_bucket"),
        "by_structure128_bucket": summarize_by(enriched, "structure128_bucket"),
        "by_quality128_bucket": summarize_by(enriched, "quality128_bucket"),
        "by_pos128_bucket": summarize_by(enriched, "pos128_bucket"),
        "by_eff128_bucket": summarize_by(enriched, "eff128_bucket"),
        "by_retvol128_bucket": summarize_by(enriched, "retvol128_bucket"),
        "rules": [rule_summary(enriched, expr, name) for name, expr in rules],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV medium-term trend quality diagnostic")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    features = build_medium_trend_features(args)
    logger.info(
        f"Built medium trend features: {features.height:,} rows, "
        f"{features['date'].min()} ~ {features['date'].max()}"
    )
    amv_phase = build_amv_phase_frame()
    sleeves = {
        key: analyze_sleeve(key, sleeve, features, amv_phase)
        for key, sleeve in DEFAULT_SLEEVES.items()
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": (
            "Stock-level 64/128 day structure and trend-quality features joined on signal_date. "
            "Diagnostic only; no signal export."
        ),
        "data": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_rows": features.height,
        },
        "feature_definitions": {
            "structure_score": "Average of medium-term return rank, range-position rank, and long MA slope rank.",
            "trend_quality_score": "Average of trend efficiency, up-day ratio, return/volatility, and body efficiency ranks.",
            "trend_eff": "Absolute medium-term return divided by cumulative absolute daily returns in the same window.",
        },
        "sleeves": sleeves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote {args.output}")
    for sleeve in sleeves.values():
        best_rule = max(sleeve["rules"], key=lambda item: item["trade_level_delta"])
        logger.info(
            f"{sleeve['label']}: total={sleeve['total']['total_pnl']:+,.0f}, "
            f"best_rule={best_rule['rule']} delta={best_rule['trade_level_delta']:+,.0f}, "
            f"skipped={best_rule['skipped']['trades']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
