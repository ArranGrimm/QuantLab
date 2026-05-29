"""Second-stage closing diagnostic for AMV trend refinements and liquidity shocks.

This covers the remaining second-stage factor ideas not fully closed by the
128-day structure / trend-quality rerank:
- explicit trend refinements: drawdown depth, trend linearity, MA stack stability
- liquidity / amount shocks: amount expansion, dry pullback, volume-without-price,
  breakout volume confirmation, liquidity recovery
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
from strategies.amv.factors.medium_trend_quality import safe_div
from strategies.amv.market import build_market_frame


DEFAULT_OUTPUT = ROOT / "reports" / "amv_liquidity_trend_refinement_diagnostic.json"


def finite_or_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def rank_pct(col_name: str, *, higher_is_better: bool = True) -> pl.Expr:
    return (
        pl.col(col_name).rank("average", descending=not higher_is_better).over("date")
        / pl.len().over("date")
    ).alias(f"{col_name}_rank_pct")


def add_refinement_features(market: pl.DataFrame) -> pl.DataFrame:
    stock = (
        market.sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                pl.int_range(pl.len()).over("code").cast(pl.Float64).alias("time_idx"),
            ]
        )
        .with_columns(
            [
                pl.col("close_adj").rolling_mean(20).over("code").alias("ma20"),
                pl.col("close_adj").rolling_mean(60).over("code").alias("ma60"),
                pl.col("close_adj").rolling_mean(120).over("code").alias("ma120"),
                pl.col("amount").rolling_mean(5).over("code").alias("amount_ma5"),
                pl.col("amount").rolling_mean(20).over("code").alias("amount_ma20_raw"),
                pl.col("amount").rolling_mean(60).over("code").alias("amount_ma60"),
            ]
        )
        .with_columns(
            [
                safe_div(pl.col("amount"), pl.col("amount_ma20_raw")).alias("amount_ratio_1_20"),
                safe_div(pl.col("amount_ma5"), pl.col("amount_ma20_raw")).alias("amount_ratio_5_20"),
                safe_div(pl.col("amount_ma20_raw"), pl.col("amount_ma60")).alias("amount_ratio_20_60"),
                ((pl.col("ma20") > pl.col("ma60")) & (pl.col("ma60") > pl.col("ma120"))).alias(
                    "ma_stack_bullish"
                ),
            ]
        )
        .with_columns(
            pl.col("ma_stack_bullish").cast(pl.Float64).rolling_mean(20).over("code").alias(
                "ma_stack_stability_20d"
            )
        )
    )

    exprs: list[pl.Expr] = []
    for window in (64, 128):
        high = pl.col("close_adj").rolling_max(window).over("code")
        low = pl.col("close_adj").rolling_min(window).over("code")
        dd = pl.col("close_adj") / high - 1.0
        pos = safe_div(pl.col("close_adj") - low, high - low)
        corr = pl.rolling_corr(
            pl.col("time_idx"),
            pl.col("close_adj"),
            window_size=window,
            min_samples=window,
        ).over("code")
        exprs.extend(
            [
                dd.alias(f"dd_from_high_{window}d_refine"),
                pos.alias(f"range_pos_{window}d_refine"),
                finite_or_null(corr.pow(2)).alias(f"trend_linearity_{window}d"),
            ]
        )
    stock = stock.with_columns(exprs)

    event_exprs = [
        (pl.col("amount_ratio_1_20") > 1.8).alias("is_amount_spike_1d"),
        (pl.col("amount_ratio_5_20") > 1.25).alias("is_amount_expansion_5d"),
        (pl.col("amount_ratio_5_20") < 0.75).alias("is_dry_5d"),
        ((pl.col("ret_5d") < 0.0) & (pl.col("amount_ratio_5_20") < 0.85)).alias("is_dry_pullback_5d"),
        ((pl.col("ret_5d") > 0.03) & (pl.col("price_pos_20d") > 0.80) & (pl.col("amount_ratio_5_20") > 1.10)).alias(
            "is_breakout_volume_confirmed"
        ),
        ((pl.col("amount_ratio_1_20") > 1.8) & (pl.col("ret_1d") < 0.01)).alias("is_volume_without_price_1d"),
        ((pl.col("amount_ratio_5_20") > 1.10) & (pl.col("amount_ratio_20_60") < 0.95)).alias(
            "is_liquidity_recovery"
        ),
    ]
    stock = stock.with_columns(event_exprs)

    rank_cols = [
        ("dd_from_high_64d_refine", True),
        ("dd_from_high_128d_refine", True),
        ("trend_linearity_64d", True),
        ("trend_linearity_128d", True),
        ("ma_stack_stability_20d", True),
        ("amount_ratio_1_20", True),
        ("amount_ratio_5_20", True),
        ("amount_ratio_20_60", True),
    ]
    stock = stock.with_columns([rank_pct(col, higher_is_better=good) for col, good in rank_cols])

    return stock.select(
        [
            "date",
            "code",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "price_pos_20d",
            "amount_ratio_1_20",
            "amount_ratio_5_20",
            "amount_ratio_20_60",
            "dd_from_high_64d_refine",
            "dd_from_high_128d_refine",
            "range_pos_64d_refine",
            "range_pos_128d_refine",
            "trend_linearity_64d",
            "trend_linearity_128d",
            "ma_stack_stability_20d",
            "is_amount_spike_1d",
            "is_amount_expansion_5d",
            "is_dry_5d",
            "is_dry_pullback_5d",
            "is_breakout_volume_confirmed",
            "is_volume_without_price_1d",
            "is_liquidity_recovery",
            "dd_from_high_64d_refine_rank_pct",
            "dd_from_high_128d_refine_rank_pct",
            "trend_linearity_64d_rank_pct",
            "trend_linearity_128d_rank_pct",
            "ma_stack_stability_20d_rank_pct",
            "amount_ratio_1_20_rank_pct",
            "amount_ratio_5_20_rank_pct",
            "amount_ratio_20_60_rank_pct",
        ]
    )


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
            bucket("dd_from_high_128d_refine_rank_pct", "drawdown128_quality").alias(
                "drawdown128_quality_bucket"
            ),
            bucket("trend_linearity_128d_rank_pct", "linearity128").alias("linearity128_bucket"),
            bucket("ma_stack_stability_20d_rank_pct", "ma_stack").alias("ma_stack_bucket"),
            bucket("amount_ratio_1_20_rank_pct", "amount1v20").alias("amount1v20_bucket"),
            bucket("amount_ratio_5_20_rank_pct", "amount5v20").alias("amount5v20_bucket"),
            bucket("amount_ratio_20_60_rank_pct", "amount20v60").alias("amount20v60_bucket"),
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
        ("skip_low_linearity_128", pl.col("trend_linearity_128d_rank_pct").fill_null(0.0) < 0.33),
        ("skip_unstable_ma_stack", pl.col("ma_stack_stability_20d_rank_pct").fill_null(0.0) < 0.33),
        (
            "skip_bad_linearity_and_stack",
            (pl.col("trend_linearity_128d_rank_pct").fill_null(0.0) < 0.50)
            & (pl.col("ma_stack_stability_20d_rank_pct").fill_null(0.0) < 0.50),
        ),
        (
            "skip_deep_drawdown_128",
            pl.col("dd_from_high_128d_refine_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_breakout_without_volume",
            (pl.col("price_pos_20d").fill_null(0.0) > 0.80)
            & (pl.col("amount_ratio_5_20_rank_pct").fill_null(0.0) < 0.33),
        ),
        ("skip_amount_spike_no_price", pl.col("is_volume_without_price_1d").fill_null(False)),
        (
            "skip_low_medium_liquidity",
            pl.col("amount_ratio_20_60_rank_pct").fill_null(0.0) < 0.33,
        ),
        (
            "skip_liquidity_recovery",
            pl.col("is_liquidity_recovery").fill_null(False),
        ),
        (
            "skip_not_dry_pullback",
            ~pl.col("is_dry_pullback_5d").fill_null(False),
        ),
        (
            "skip_unconfirmed_breakout",
            (pl.col("price_pos_20d").fill_null(0.0) > 0.80)
            & ~pl.col("is_breakout_volume_confirmed").fill_null(False),
        ),
    ]

    event_summary = {}
    for col_name in [
        "is_amount_spike_1d",
        "is_amount_expansion_5d",
        "is_dry_5d",
        "is_dry_pullback_5d",
        "is_breakout_volume_confirmed",
        "is_volume_without_price_1d",
        "is_liquidity_recovery",
    ]:
        event_summary[col_name] = {
            "true": summarize_trades(enriched.filter(pl.col(col_name).fill_null(False))),
            "false": summarize_trades(enriched.filter(~pl.col(col_name).fill_null(False))),
        }

    return {
        "sleeve": sleeve_key,
        "label": sleeve["label"],
        "total": summarize_trades(enriched),
        "missing_feature_trades": int(enriched.filter(pl.col("amount_ratio_5_20").is_null()).height),
        "by_drawdown128_quality_bucket": summarize_by(enriched, "drawdown128_quality_bucket"),
        "by_linearity128_bucket": summarize_by(enriched, "linearity128_bucket"),
        "by_ma_stack_bucket": summarize_by(enriched, "ma_stack_bucket"),
        "by_amount1v20_bucket": summarize_by(enriched, "amount1v20_bucket"),
        "by_amount5v20_bucket": summarize_by(enriched, "amount5v20_bucket"),
        "by_amount20v60_bucket": summarize_by(enriched, "amount20v60_bucket"),
        "event_summary": event_summary,
        "rules": [rule_summary(enriched, expr, name) for name, expr in rules],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV liquidity and trend refinement diagnostic")
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

    market = build_market_frame(args)
    features = add_refinement_features(market)
    logger.info(
        f"Built liquidity/trend refinement features: {features.height:,} rows, "
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
            "Stock-level trend refinement and liquidity/amount features joined on signal_date. "
            "Diagnostic only; no signal export."
        ),
        "data": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_rows": features.height,
        },
        "feature_definitions": {
            "trend_linearity": "Rolling R^2 of adjusted close against time index within 64/128 day windows.",
            "ma_stack_stability_20d": "20-day share of days with MA20 > MA60 > MA120.",
            "amount_ratio_1_20": "Daily amount divided by 20-day average amount.",
            "amount_ratio_5_20": "5-day average amount divided by 20-day average amount.",
            "amount_ratio_20_60": "20-day average amount divided by 60-day average amount.",
            "dry_pullback_5d": "5-day return below zero and 5-day amount below 85% of 20-day amount.",
            "breakout_volume_confirmed": "5-day return > 3%, 20-day position > 80%, and 5-day amount expansion > 1.10x.",
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
