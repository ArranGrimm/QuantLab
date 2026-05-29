"""Sector tailwind diagnostic for AMV sleeves.

Prototype scope:
- Use a static East Money industry map for stock -> industry.
- Use QMT daily adjusted OHLCV to derive industry breadth, trend, and liquidity factors.
- Join factors on signal_date to avoid using entry-day close for T+1 open decisions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame
from strategies.amv.factors.sector_tailwind import (
    build_sector_tailwind_features,
    format_stock_code,
    load_daily_with_industry,
    load_sector_map,
    refresh_em_sector_map,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_SECTOR_MAP = ROOT / "data" / "sector_map_em.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "amv_sector_tailwind_diagnostic.json"

DEFAULT_SLEEVES: dict[str, dict[str, str]] = {
    "p3_static": {
        "label": "P3 static strict",
        "trades": "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092208_801/trades.csv",
        "signals": "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/signal.parquet",
    },
    "pb3_rolling": {
        "label": "PB3 rolling raw",
        "trades": "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/backtests/6td_rolling21_refill_top10_no_stop_20260521_091007_830/trades.csv",
        "signals": "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/signal.parquet",
    },
}


def load_trade_context(trades_path: Path, signals_path: Path) -> pl.DataFrame:
    trades = pl.read_csv(trades_path, try_parse_dates=True).with_row_index("trade_id")
    signals = (
        pl.scan_parquet(signals_path)
        .filter(pl.col("is_signal"))
        .select(
            [
                pl.col("date").alias("entry_date"),
                "code",
                "signal_date",
                "score",
                "rank",
            ]
        )
        .collect()
    )
    joined = trades.join(signals, on=["entry_date", "code"], how="left")
    missing = joined.filter(pl.col("signal_date").is_null()).height
    if missing:
        logger.warning(f"{missing:,}/{joined.height:,} trades could not be matched to signal rows")
    return joined


def summarize_trades(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl_pct": 0.0,
            "win_rate": 0.0,
        }
    return {
        "trades": int(df.height),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 6),
        "win_rate": round(float((df["pnl"] > 0).mean()), 6),
    }


def summarize_by(df: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for row in (
        df.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
        .sort("total_pnl", descending=True)
        .iter_rows(named=True)
    ):
        rows.append(
            {
                column: row[column],
                "trades": int(row["trades"]),
                "total_pnl": round(float(row["total_pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
            }
        )
    return rows


def rule_summary(df: pl.DataFrame, rule: pl.Expr, name: str) -> dict[str, Any]:
    skipped = df.filter(rule)
    kept = df.filter(~rule)
    skipped_pnl = float(skipped["pnl"].sum()) if skipped.height else 0.0
    yearly = (
        skipped.with_columns(pl.col("entry_date").dt.year().alias("year"))
        .group_by("year")
        .agg((-pl.col("pnl").sum()).alias("trade_level_delta"))
        .sort("year")
        if skipped.height
        else pl.DataFrame({"year": [], "trade_level_delta": []})
    )
    return {
        "rule": name,
        "skipped": summarize_trades(skipped),
        "kept": summarize_trades(kept),
        "trade_level_delta": round(-skipped_pnl, 2),
        "skipped_big_winner_gt_20k": int(skipped.filter(pl.col("pnl") > 20_000).height),
        "skipped_big_loser_lt_minus_20k": int(skipped.filter(pl.col("pnl") < -20_000).height),
        "yearly_trade_level_delta": {
            str(row["year"]): round(float(row["trade_level_delta"]), 2) for row in yearly.iter_rows(named=True)
        },
    }


def analyze_sleeve(
    sleeve_key: str,
    sleeve: dict[str, str],
    features: pl.DataFrame,
    amv_phase: pl.DataFrame,
) -> dict[str, Any]:
    trades = load_trade_context(ROOT / sleeve["trades"], ROOT / sleeve["signals"])
    enriched = (
        trades.join(
            features,
            left_on=["signal_date", "code"],
            right_on=["date", "code"],
            how="left",
        )
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
        .with_columns(
            [
                pl.when(pl.col("sector_tailwind_ok").fill_null(False))
                .then(pl.lit("tailwind_ok"))
                .otherwise(pl.lit("tailwind_weak"))
                .alias("sector_tailwind_bucket"),
                pl.when(pl.col("sector_ret_10d_rank_pct").is_null())
                .then(pl.lit("no_data"))
                .when(pl.col("sector_ret_10d_rank_pct") >= 0.70)
                .then(pl.lit("top_30pct"))
                .when(pl.col("sector_ret_10d_rank_pct") >= 0.40)
                .then(pl.lit("mid_30pct"))
                .otherwise(pl.lit("bottom_40pct"))
                .alias("sector_rank_bucket"),
            ]
        )
    )

    rules = [
        (
            "skip_sector_tailwind_weak",
            ~pl.col("sector_tailwind_ok").fill_null(False),
        ),
        (
            "skip_sector_bottom_40pct",
            pl.col("sector_ret_10d_rank_pct").fill_null(0.0) < 0.40,
        ),
        (
            "skip_sector_cold_breadth",
            pl.col("sector_breadth_ma20").fill_null(0.0) < 0.35,
        ),
        (
            "skip_aged_or_old_and_sector_weak",
            pl.col("fwd_duration_bucket").is_in(["aged", "old"]) & ~pl.col("sector_tailwind_ok").fill_null(False),
        ),
        (
            "skip_aged_or_old_and_sector_bottom",
            pl.col("fwd_duration_bucket").is_in(["aged", "old"])
            & (pl.col("sector_ret_10d_rank_pct").fill_null(0.0) < 0.40),
        ),
    ]

    return {
        "sleeve": sleeve_key,
        "label": sleeve["label"],
        "total": summarize_trades(enriched),
        "missing_feature_trades": int(enriched.filter(pl.col("industry").is_null()).height),
        "by_tailwind_bucket": summarize_by(enriched, "sector_tailwind_bucket"),
        "by_sector_rank_bucket": summarize_by(enriched, "sector_rank_bucket"),
        "top_industries_by_pnl": summarize_by(enriched.filter(pl.col("industry").is_not_null()), "industry")[:12],
        "bottom_industries_by_pnl": summarize_by(enriched.filter(pl.col("industry").is_not_null()), "industry")[-12:],
        "rules": [rule_summary(enriched, expr, name) for name, expr in rules],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV sector tailwind factor diagnostic")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--refresh-sector-map", action="store_true")
    parser.add_argument("--sector-map-request-sleep", type=float, default=0.35)
    args = parser.parse_args()

    sector_map = load_sector_map(
        args.sector_map,
        refresh=args.refresh_sector_map,
        request_sleep=args.sector_map_request_sleep,
    )
    logger.info(f"Loaded {sector_map.height:,} sector mappings across {sector_map['industry'].n_unique()} industries")

    daily = load_daily_with_industry(args.db, sector_map, args.start_date)
    logger.info(
        f"Loaded daily frame: {daily.height:,} rows, {daily['code'].n_unique():,} stocks, "
        f"{daily['date'].min()} ~ {daily['date'].max()}"
    )

    features = build_sector_tailwind_features(daily)
    logger.info(f"Built feature frame: {features.height:,} rows")

    amv_phase = build_amv_phase_frame()
    sleeves = {
        key: analyze_sleeve(key, sleeve, features, amv_phase)
        for key, sleeve in DEFAULT_SLEEVES.items()
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": (
            "Static East Money industry map plus QMT daily adjusted prices. "
            "Sector factors are joined on signal_date to avoid entry-day close lookahead."
        ),
        "data": {
            "qmt_db": str(args.db),
            "sector_map": str(args.sector_map),
            "sector_map_rows": sector_map.height,
            "industries": sector_map["industry"].n_unique(),
            "start_date": args.start_date,
            "daily_rows": daily.height,
        },
        "tailwind_definition": {
            "sector_ret_10d_rank_pct": ">= 0.65",
            "sector_breadth_ma20": ">= 0.45",
            "sector_amount_ratio_20": ">= 0.90",
        },
        "sleeves": sleeves,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps({"output": str(args.output), "data": payload["data"]}, ensure_ascii=False, indent=2))
    for sleeve in sleeves.values():
        best_rule = max(sleeve["rules"], key=lambda item: item["trade_level_delta"])
        print(
            f"{sleeve['label']}: total={sleeve['total']['total_pnl']:+,.0f}, "
            f"best_rule={best_rule['rule']} delta={best_rule['trade_level_delta']:+,.0f}, "
            f"skipped={best_rule['skipped']['trades']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
