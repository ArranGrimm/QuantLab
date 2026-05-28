"""Export P3 market-sentiment gated signals for bt-amv-topn.

The first market sentiment candidate is a date-level "overheated/crowded"
gate inside AMV bull:

- yesterday limit-up stocks have high next-day close premium
- and whole-market 20-day new-high breadth is high

Because this is a date-level feature, subtracting the same penalty from every
candidate on the date would not change Top3 order. The first executable test is
therefore a strict no-new-entry gate on crowded signal dates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT, _finite_expr, _git_commit, _rel_path
from scripts.amv_sector_tailwind_signal_export import build_market_frame
from scripts.amv_static_sleeve_signal_export import pullback_combo_score_expr


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "amv_static_sleeve_signals"


def threshold_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def price_limit_pct_expr() -> pl.Expr:
    is_20pct_board = (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sz.301")
        | pl.col("code").str.starts_with("sh.688")
        | pl.col("code").str.starts_with("sh.689")
        | pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )
    return pl.when(is_20pct_board).then(0.20).otherwise(0.10)


def build_sentiment_features(market: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    stock = (
        market.sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0).alias("ret_1d"),
                (pl.col("open_adj") / pl.col("pre_close_adj") - 1.0).alias("open_gap"),
                price_limit_pct_expr().alias("limit_pct"),
                pl.col("close_adj").rolling_max(20).over("code").alias("high_close_20d"),
            ]
        )
        .with_columns(
            [
                ((pl.col("ret_1d") >= (pl.col("limit_pct") - args.price_limit_tolerance))).alias(
                    "is_close_limit_up"
                ),
                ((pl.col("ret_1d") <= (-pl.col("limit_pct") + args.price_limit_tolerance))).alias(
                    "is_close_limit_down"
                ),
                (
                    (pl.col("high_adj") / pl.col("pre_close_adj") - 1.0)
                    >= (pl.col("limit_pct") - args.price_limit_tolerance)
                ).alias("touched_limit_up"),
                (pl.col("close_adj") >= pl.col("high_close_20d")).alias("is_new_high_20"),
                (pl.col("close_adj") / pl.col("high_close_20d") - 1.0).alias("drawdown_from_20d_high"),
                (pl.col("ret_20d").rank("average").over("date") / pl.len().over("date")).alias("ret_20d_rank_pct"),
            ]
        )
        .with_columns(
            [
                (pl.col("touched_limit_up") & ~pl.col("is_close_limit_up")).alias("is_failed_limit_up"),
                (pl.col("ret_20d_rank_pct") >= 0.80).alias("is_strong_20d_stock"),
                pl.col("is_close_limit_up").shift(1).over("code").fill_null(False).alias("was_limit_up_yesterday"),
            ]
        )
    )

    daily = (
        stock.group_by("date")
        .agg(
            [
                pl.len().alias("market_stock_count"),
                pl.col("is_close_limit_up").sum().alias("limit_up_count"),
                pl.col("is_close_limit_down").sum().alias("limit_down_count"),
                pl.col("touched_limit_up").sum().alias("touched_limit_up_count"),
                pl.col("is_failed_limit_up").sum().alias("failed_limit_up_count"),
                pl.col("is_new_high_20").mean().alias("new_high_20_ratio"),
                pl.col("drawdown_from_20d_high")
                .filter(pl.col("is_strong_20d_stock"))
                .median()
                .alias("strong_stock_drawdown_20d_median"),
                pl.col("ret_1d")
                .filter(pl.col("was_limit_up_yesterday"))
                .mean()
                .alias("yday_limit_up_close_premium"),
            ]
        )
        .with_columns(
            (
                pl.col("failed_limit_up_count")
                / pl.when(pl.col("touched_limit_up_count") > 0).then(pl.col("touched_limit_up_count")).otherwise(1)
            ).alias("failed_limit_up_ratio")
        )
        .sort("date")
    )

    return daily.with_columns(
        [
            (pl.col("new_high_20_ratio").rank("average") / pl.len()).alias("new_high_20_ratio_rank_pct"),
            (pl.col("yday_limit_up_close_premium").rank("average") / pl.len()).alias(
                "yday_limit_up_close_premium_rank_pct"
            ),
        ]
    )


def build_signal(
    market: pl.DataFrame,
    sentiment: pl.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    sleeve_id = (
        "p3_sentiment_hot_yday_premium_newhigh_gate"
        f"_yp{threshold_token(args.yday_premium_rank_threshold)}"
        f"_nh{threshold_token(args.new_high_rank_threshold)}"
    )

    base_score_expr, required_cols = pullback_combo_score_expr(
        price_weight=3.0,
        kbar_weight=0.5,
        bias_weight=0.0,
        close_pullback_weight=0.0,
        risk_weight=0.0,
    )
    valid_expr = _finite_expr(required_cols[0])
    for col_name in required_cols[1:]:
        valid_expr = valid_expr & _finite_expr(col_name)

    candidate_expr = (
        pl.col("is_bull_regime")
        & (pl.col("market_cap_100m") >= args.mv_min)
        & (pl.col("amount_ma20") >= args.amount_ma20_min)
        & valid_expr
    )
    hot_expr = (
        (pl.col("yday_limit_up_close_premium_rank_pct").fill_null(0.0) >= args.yday_premium_rank_threshold)
        & (pl.col("new_high_20_ratio_rank_pct").fill_null(0.0) >= args.new_high_rank_threshold)
    )

    scored = (
        market.join(sentiment, on="date", how="left")
        .with_columns(
            [
                candidate_expr.alias("_is_signal_candidate"),
                hot_expr.alias("_sentiment_hot_gate"),
                base_score_expr.alias("_base_signal_score"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_is_signal_candidate"))
            .then(pl.col("_base_signal_score"))
            .otherwise(None)
            .alias("_base_candidate_score")
        )
        .with_columns(
            pl.when(pl.col("_is_signal_candidate") & ~pl.col("_sentiment_hot_gate"))
            .then(pl.col("_base_signal_score"))
            .otherwise(None)
            .alias("_signal_score")
        )
        .with_columns(
            [
                pl.col("_base_candidate_score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .alias("_base_rank"),
                pl.col("_signal_score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .alias("_signal_rank"),
            ]
        )
    )

    raw_top3 = (
        scored.filter(pl.col("_is_signal_candidate"))
        .filter(pl.col("_base_rank") <= args.top_n)
        .select(["date", "code"])
    )
    gated_raw_top3 = (
        scored.filter(pl.col("_is_signal_candidate") & pl.col("_sentiment_hot_gate"))
        .filter(pl.col("_base_rank") <= args.top_n)
        .select(["date", "code"])
    )

    signal_rows = (
        scored.filter(pl.col("_is_signal_candidate") & ~pl.col("_sentiment_hot_gate") & (pl.col("_signal_rank") <= args.top_n))
        .select(
            [
                pl.col("date").alias("signal_date"),
                "code",
                pl.lit(sleeve_id).alias("sleeve_id"),
                pl.col("_base_signal_score").alias("base_score"),
                pl.col("_signal_score").alias("score"),
                pl.col("_signal_rank").cast(pl.UInt32).alias("rank"),
                pl.col("_sentiment_hot_gate").alias("sentiment_hot_gate"),
                pl.col("limit_up_count"),
                pl.col("limit_down_count"),
                pl.col("failed_limit_up_ratio"),
                pl.col("new_high_20_ratio"),
                pl.col("yday_limit_up_close_premium"),
                pl.col("strong_stock_drawdown_20d_median"),
                pl.col("new_high_20_ratio_rank_pct"),
                pl.col("yday_limit_up_close_premium_rank_pct"),
            ]
        )
        .sort(["signal_date", "rank", "code"])
    )

    gated_dates = (
        scored.filter(pl.col("_sentiment_hot_gate"))
        .select(pl.col("date").alias("signal_date"))
        .unique()
        .sort("signal_date")
    )

    trading_dates = market.select("date").unique().sort("date")
    next_dates = trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date")).drop_nulls(
        "execution_date"
    )
    execution_signals = (
        signal_rows.join(next_dates, left_on="signal_date", right_on="date", how="inner")
        .select(["execution_date", "code", "signal_date", "sleeve_id", "score", "rank"])
        .rename({"execution_date": "date"})
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

    selected_key = signal_rows.select([pl.col("signal_date").alias("date"), "code"])
    overlap = selected_key.join(raw_top3, on=["date", "code"], how="inner").height
    summary = {
        "sleeve_id": sleeve_id,
        "base_sleeve_id": "candidate_p3_k0p5_b0_c0_r0",
        "gate": "hot_yday_limit_up_premium_and_high_new_high_20",
        "yday_premium_rank_threshold": args.yday_premium_rank_threshold,
        "new_high_rank_threshold": args.new_high_rank_threshold,
        "rows": export.height,
        "date_min": str(export["date"].min()),
        "date_max": str(export["date"].max()),
        "signal_rows_before_shift": signal_rows.height,
        "signal_rows_after_shift": int(export["is_signal"].sum()),
        "signal_days_after_shift": export.filter(pl.col("is_signal")).select("date").n_unique(),
        "gated_signal_dates": gated_dates.height,
        "gated_raw_top3_rows": gated_raw_top3.height,
        "raw_top3_overlap_rows": overlap,
        "raw_top3_overlap_ratio": overlap / signal_rows.height if signal_rows.height else None,
    }
    return export, signal_rows, summary


def write_signal_artifact(
    *,
    output_root: Path,
    export: pl.DataFrame,
    selected: pl.DataFrame,
    summary: dict[str, Any],
    args: argparse.Namespace,
    started_at: datetime,
) -> Path:
    output_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{summary['sleeve_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / "signal.parquet"
    selected_path = output_dir / "selected_signals.csv"
    meta_path = output_dir / "signal.meta.json"
    export.write_parquet(signal_path)
    selected.write_csv(selected_path)

    meta = {
        "strategy": "amv_static_sleeve_topn",
        "signal_id": output_dir.name,
        "signal_run_id": f"amv_static_sleeve_topn_{output_dir.name}",
        "signal_timestamp": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "label": f"static_sleeve:{summary['sleeve_id']}",
        "model_name": "static_factor_sleeve_market_sentiment_gate",
        "feature_mode": summary["sleeve_id"],
        "signal_path": "signal.parquet",
        "canonical_signal_path": "signal.parquet",
        "signal_meta_path": "signal.meta.json",
        "backtest_index_path": "backtest.jsonl",
        "backtests_dir": "backtests",
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "top_n": args.top_n,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "price_limit_tolerance": args.price_limit_tolerance,
            "yday_premium_rank_threshold": args.yday_premium_rank_threshold,
            "new_high_rank_threshold": args.new_high_rank_threshold,
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
    parser = argparse.ArgumentParser(description="Export P3 market-sentiment gated signals")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--price-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--yday-premium-rank-threshold", type=float, default=0.67)
    parser.add_argument("--new-high-rank-threshold", type=float, default=0.67)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building P3 market feature frame ...")
    market = build_market_frame(args)
    print("Building market sentiment features ...")
    sentiment = build_sentiment_features(market, args)
    export, selected, summary = build_signal(market, sentiment, args=args)
    meta_path = write_signal_artifact(
        output_root=args.output_root,
        export=export,
        selected=selected,
        summary=summary,
        args=args,
        started_at=started_at,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Saved signal meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
