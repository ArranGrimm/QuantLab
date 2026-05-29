"""Drawdown attribution for AMV limit ecology event sleeves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT
from scripts.amv_limit_ecology_signal_export import build_limit_ecology_market
from strategies.amv.factors.limit_ecology import LIMIT_TOLERANCE


BASE_SIGNAL = ROOT / "artifacts" / "amv_static_sleeve_signals" / "20260529_111655_limit_first_board_pullback"
QUALITY_SIGNAL = ROOT / "artifacts" / "amv_static_sleeve_signals" / "20260529_112355_limit_first_board_pullback_quality"
DEFAULT_OUTPUT = ROOT / "reports" / "amv_limit_first_board_pullback_drawdown_attribution.json"


def feature_args() -> argparse.Namespace:
    return argparse.Namespace(
        qmt_db=DEFAULT_QMT_DB,
        start_date="2019-01-01",
        end_date="2026-05-10",
        st_snapshot_date="2026-03-31",
        mv_min=100.0,
        amount_ma20_min=5e7,
        top_n=3,
        price_limit_tolerance=LIMIT_TOLERANCE,
        amv_bull_trigger_pct=4.0,
        amv_bull_lookback_days=2,
        amv_bear_trigger_1d_pct=-2.3,
        amv_effective_lag_days=1,
    )


def summarize_trades(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl_pct": 0.0,
            "win_rate": 0.0,
            "big_winners_gt_20k": 0,
            "big_losers_lt_minus_20k": 0,
        }
    return {
        "trades": int(df.height),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 6),
        "win_rate": round(float((df["pnl"] > 0).mean()), 6),
        "big_winners_gt_20k": int(df.filter(pl.col("pnl") > 20_000).height),
        "big_losers_lt_minus_20k": int(df.filter(pl.col("pnl") < -20_000).height),
    }


def risk_profile(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {}
    cols = set(df.columns)

    def optional_mean(column: str) -> float | None:
        if column not in cols:
            return None
        value = df[column].mean()
        return None if value is None else round(float(value), 6)

    def optional_pct(expr: pl.Expr) -> float | None:
        try:
            value = df.select(expr.mean()).item()
        except pl.exceptions.ColumnNotFoundError:
            return None
        return None if value is None else round(float(value), 6)

    return {
        "trades": int(df.height),
        "avg_score": optional_mean("score"),
        "avg_amount_ratio_5_20": optional_mean("amount_ratio_5_20"),
        "avg_atr_14_pct_rank_pct": optional_mean("atr_14_pct_rank_pct"),
        "avg_panic_vol_ratio_20d_rank_pct": optional_mean("panic_vol_ratio_20d_rank_pct"),
        "pct_atr_gt_0p8": optional_pct(pl.col("atr_14_pct_rank_pct") > 0.8),
        "pct_panic_gt_0p8": optional_pct(pl.col("panic_vol_ratio_20d_rank_pct") > 0.8),
        "pct_amount_le_0p85": optional_pct(pl.col("amount_ratio_5_20") <= 0.85),
        "pct_days_since_lu_le_3": optional_pct(pl.col("days_since_prior_limit_up") <= 3),
        "pct_days_since_lu_ge_7": optional_pct(pl.col("days_since_prior_limit_up") >= 7),
        "pct_failed_limit_5d": optional_pct(pl.col("has_failed_limit_up_5d")),
    }


def top_records(df: pl.DataFrame, *, by: str, descending: bool, n: int = 10) -> list[dict[str, Any]]:
    cols = [
        "code",
        "entry_date",
        "exit_date",
        "pnl",
        "pnl_pct",
        "hold_trading_days",
        "score",
        "rank",
        "days_since_prior_limit_up",
        "amount_ratio_5_20",
        "atr_14_pct_rank_pct",
        "panic_vol_ratio_20d_rank_pct",
        "has_failed_limit_up_5d",
    ]
    available = [col for col in cols if col in df.columns]
    records = []
    for row in df.sort(by, descending=descending).head(n).select(available).iter_rows(named=True):
        item = dict(row)
        for key in ("entry_date", "exit_date"):
            if key in item and item[key] is not None:
                item[key] = str(item[key])
        if "pnl" in item and item["pnl"] is not None:
            item["pnl"] = round(float(item["pnl"]), 2)
        if "pnl_pct" in item and item["pnl_pct"] is not None:
            item["pnl_pct"] = round(float(item["pnl_pct"]), 6)
        records.append(item)
    return records


def load_feature_lookup() -> pl.DataFrame:
    return build_limit_ecology_market(feature_args()).select(
        [
            pl.col("date").alias("signal_date"),
            "code",
            "amount_ratio_5_20",
            "atr_14_pct_rank_pct",
            "panic_vol_ratio_20d_rank_pct",
        ]
    )


def load_enriched_trades(
    signal_dir: Path,
    backtest_name: str,
    feature_lookup: pl.DataFrame,
) -> pl.DataFrame:
    trades = pl.read_csv(signal_dir / "backtests" / backtest_name / "trades.csv", try_parse_dates=True)
    signal_rows = (
        pl.scan_parquet(signal_dir / "signal.parquet")
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
    selected = pl.read_csv(signal_dir / "selected_signals.csv", try_parse_dates=True)
    enriched = (
        trades.join(signal_rows, on=["entry_date", "code"], how="left")
        .join(
            selected.drop(["sleeve_id", "score", "rank"], strict=False),
            on=["signal_date", "code"],
            how="left",
        )
    )
    missing_feature_cols = [
        col
        for col in ["amount_ratio_5_20", "atr_14_pct_rank_pct", "panic_vol_ratio_20d_rank_pct"]
        if col not in enriched.columns
    ]
    if missing_feature_cols:
        enriched = enriched.join(feature_lookup, on=["signal_date", "code"], how="left")
    return (
        enriched
        .with_columns(
            [
                pl.col("entry_date").dt.strftime("%Y-%m").alias("entry_month"),
                pl.col("exit_date").dt.strftime("%Y-%m").alias("exit_month"),
                pl.col("entry_date").dt.year().alias("entry_year"),
                pl.col("exit_date").dt.year().alias("exit_year"),
            ]
        )
    )


def load_equity(signal_dir: Path, backtest_name: str) -> pl.DataFrame:
    return pl.read_csv(signal_dir / "backtests" / backtest_name / "daily_equity.csv", try_parse_dates=True)


def drawdown_summary(equity: pl.DataFrame) -> dict[str, Any]:
    dd = (
        equity.with_columns(pl.col("total_value").cum_max().alias("peak_value"))
        .with_columns((pl.col("total_value") / pl.col("peak_value") - 1.0).alias("drawdown"))
        .with_row_index("row_idx")
    )
    trough = dd.sort("drawdown").row(0, named=True)
    trough_idx = int(trough["row_idx"])
    peak = dd.filter(pl.col("row_idx") <= trough_idx).sort("total_value", descending=True).row(0, named=True)
    after = dd.filter(pl.col("row_idx") > trough_idx).filter(pl.col("total_value") >= float(peak["total_value"]))
    recovery_date = str(after["date"][0]) if after.height else None
    return {
        "peak_date": str(peak["date"]),
        "trough_date": str(trough["date"]),
        "recovery_date": recovery_date,
        "peak_value": round(float(peak["total_value"]), 2),
        "trough_value": round(float(trough["total_value"]), 2),
        "max_drawdown_pct": round(float(-trough["drawdown"] * 100.0), 2),
    }


def monthly_pnl(trades: pl.DataFrame, column: str = "exit_month") -> list[dict[str, Any]]:
    rows = []
    for row in (
        trades.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
        .sort("pnl")
        .iter_rows(named=True)
    ):
        rows.append(
            {
                "month": row[column],
                "trades": int(row["trades"]),
                "pnl": round(float(row["pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
            }
        )
    return rows


def bucket_summary(df: pl.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for row in (
        df.group_by(column)
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
        .sort(column)
        .iter_rows(named=True)
    ):
        rows.append(
            {
                str(column): str(row[column]),
                "trades": int(row["trades"]),
                "pnl": round(float(row["pnl"]), 2),
                "avg_pnl_pct": round(float(row["avg_pnl_pct"]), 6),
                "win_rate": round(float(row["win_rate"]), 6),
            }
        )
    return rows


def add_feature_buckets(trades: pl.DataFrame) -> pl.DataFrame:
    amount = pl.col("amount_ratio_5_20").fill_null(1.0)
    atr = pl.col("atr_14_pct_rank_pct").fill_null(0.5)
    panic = pl.col("panic_vol_ratio_20d_rank_pct").fill_null(0.5)
    days = pl.col("days_since_prior_limit_up")
    return trades.with_columns(
        [
            (
                pl.when(days <= 3)
                .then(pl.lit("d1_3"))
                .when(days <= 6)
                .then(pl.lit("d4_6"))
                .when(days <= 10)
                .then(pl.lit("d7_10"))
                .otherwise(pl.lit("other"))
            ).alias("days_bucket"),
            (
                pl.when(amount <= 0.85)
                .then(pl.lit("dry_le_0p85"))
                .when(amount <= 1.10)
                .then(pl.lit("normal_0p85_1p10"))
                .otherwise(pl.lit("expanded_gt_1p10"))
            ).alias("amount_bucket"),
            (
                pl.when((atr <= 0.67) & (panic <= 0.67))
                .then(pl.lit("lowvol_ok"))
                .otherwise(pl.lit("high_risk_vol"))
            ).alias("vol_risk_bucket"),
        ]
    )


def compare_trade_sets(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, Any]:
    left_keyed = left.with_columns((pl.col("entry_date").cast(pl.Utf8) + "|" + pl.col("code")).alias("_key"))
    right_keys = right.with_columns((pl.col("entry_date").cast(pl.Utf8) + "|" + pl.col("code")).alias("_key")).select(
        "_key"
    )
    left_only = left_keyed.join(right_keys, on="_key", how="anti")
    common = left_keyed.join(right_keys, on="_key", how="inner")
    return {
        "common": summarize_trades(common),
        "left_only": summarize_trades(left_only),
        "left_only_worst_trades": top_records(left_only, by="pnl", descending=False, n=10),
        "left_only_by_amount_bucket": bucket_summary(add_feature_buckets(left_only), "amount_bucket"),
        "left_only_by_vol_risk_bucket": bucket_summary(add_feature_buckets(left_only), "vol_risk_bucket"),
    }


def analyze_case(
    name: str,
    signal_dir: Path,
    backtest_name: str,
    feature_lookup: pl.DataFrame,
) -> dict[str, Any]:
    trades = add_feature_buckets(load_enriched_trades(signal_dir, backtest_name, feature_lookup))
    equity = load_equity(signal_dir, backtest_name)
    dd = drawdown_summary(equity)
    peak_date = datetime.fromisoformat(dd["peak_date"]).date()
    trough_date = datetime.fromisoformat(dd["trough_date"]).date()
    dd_trades = trades.filter((pl.col("exit_date") >= peak_date) & (pl.col("exit_date") <= trough_date))
    worst_trades = trades.sort("pnl").head(12)
    dd_worst_trades = dd_trades.sort("pnl").head(12)
    return {
        "name": name,
        "signal_dir": str(signal_dir.relative_to(ROOT)),
        "backtest_name": backtest_name,
        "total": summarize_trades(trades),
        "drawdown": dd,
        "drawdown_trades": summarize_trades(dd_trades),
        "total_risk_profile": risk_profile(trades),
        "drawdown_risk_profile": risk_profile(dd_trades),
        "worst_12_risk_profile": risk_profile(worst_trades),
        "drawdown_worst_12_risk_profile": risk_profile(dd_worst_trades),
        "worst_months": monthly_pnl(trades)[:8],
        "drawdown_worst_months": monthly_pnl(dd_trades)[:8],
        "worst_trades": top_records(worst_trades, by="pnl", descending=False, n=12),
        "drawdown_worst_trades": top_records(dd_worst_trades, by="pnl", descending=False, n=12),
        "by_days_bucket": bucket_summary(trades, "days_bucket"),
        "by_amount_bucket": bucket_summary(trades, "amount_bucket"),
        "by_vol_risk_bucket": bucket_summary(trades, "vol_risk_bucket"),
        "by_failed_limit_flag": bucket_summary(trades, "has_failed_limit_up_5d"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV limit ecology drawdown attribution")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_backtest = "5td_static_strict_top3_no_stop_20260529_111655_limit_first_board_pullback"
    quality_backtest = "6td_static_strict_top3_no_stop_20260529_112355_limit_first_board_pullback_quality"
    feature_lookup = load_feature_lookup()
    base_trades = add_feature_buckets(load_enriched_trades(BASE_SIGNAL, base_backtest, feature_lookup))
    quality_trades = add_feature_buckets(load_enriched_trades(QUALITY_SIGNAL, quality_backtest, feature_lookup))

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "Drawdown attribution for limit_first_board_pullback base 5td and quality 6td",
        "cases": [
            analyze_case("base_5td", BASE_SIGNAL, base_backtest, feature_lookup),
            analyze_case("quality_6td", QUALITY_SIGNAL, quality_backtest, feature_lookup),
        ],
        "base_vs_quality_trade_set": compare_trade_sets(base_trades, quality_trades),
        "interpretation": {
            "base_5td": "High total return comes from broad participation, but the same breadth keeps high-risk volatility and expanded-volume pullbacks in the book.",
            "quality_6td": "Quality filter removes most high-risk trades and lowers MaxDD, but also cuts too much gross edge and leaves 2023 weak.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
