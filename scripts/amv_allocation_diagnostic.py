from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "amv_p3_pb3_gated_allocation.json"
TRADING_DAYS_PER_YEAR = 252.0

P3_EQUITY = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0"
    / "backtests/6td_static_strict_top3_no_stop_20260520_092208_801/daily_equity.csv"
)
PB3_RAW_EQUITY = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0"
    / "backtests/6td_rolling21_refill_top10_no_stop_20260521_091007_830/daily_equity.csv"
)
PB3_GATED_EQUITY = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260526_184023_pullback_p0_k0_pb3_cp1_rv0"
    / "backtests/6td_rolling21_refill_top10_no_stop_20260526_184038_485/daily_equity.csv"
)


def load_returns(path: Path, name: str, *, initial_capital: float) -> pl.DataFrame:
    df = (
        pl.read_csv(path, try_parse_dates=True)
        .sort("date")
        .select(["date", pl.col("total_value").cast(pl.Float64)])
        .with_columns(
            pl.when(pl.col("total_value").shift(1).is_null())
            .then(pl.col("total_value") / initial_capital - 1.0)
            .otherwise(pl.col("total_value") / pl.col("total_value").shift(1) - 1.0)
            .alias(f"{name}_ret")
        )
        .select(["date", f"{name}_ret"])
    )
    return df


def max_drawdown(nav: pl.Series) -> tuple[float, str | None, str | None, str | None]:
    values = nav.to_list()
    if not values:
        return 0.0, None, None, None
    dates = nav.struct.field("date").to_list() if nav.dtype == pl.Struct else None
    peak = values[0]
    peak_idx = 0
    max_dd = 0.0
    trough_idx = 0
    peak_at_trough_idx = 0
    for idx, value in enumerate(values):
        if value > peak:
            peak = value
            peak_idx = idx
        dd = value / peak - 1.0 if peak else 0.0
        if dd < max_dd:
            max_dd = dd
            trough_idx = idx
            peak_at_trough_idx = peak_idx
    if dates is None:
        return max_dd, None, None, None
    recovery = None
    peak_value = values[peak_at_trough_idx]
    for idx in range(trough_idx + 1, len(values)):
        if values[idx] >= peak_value:
            recovery = str(dates[idx])
            break
    return max_dd, str(dates[peak_at_trough_idx]), str(dates[trough_idx]), recovery


def drawdown_stats(df: pl.DataFrame) -> dict[str, Any]:
    rows = df.select(["date", "nav"]).to_dicts()
    if not rows:
        return {
            "max_drawdown": 0.0,
            "drawdown_peak_date": None,
            "drawdown_trough_date": None,
            "drawdown_recovery_date": None,
        }
    peak = rows[0]["nav"]
    peak_date = rows[0]["date"]
    max_dd = 0.0
    trough_date = rows[0]["date"]
    peak_at_trough = rows[0]["date"]
    peak_value_at_trough = peak
    for row in rows:
        nav = row["nav"]
        if nav > peak:
            peak = nav
            peak_date = row["date"]
        dd = nav / peak - 1.0 if peak else 0.0
        if dd < max_dd:
            max_dd = dd
            trough_date = row["date"]
            peak_at_trough = peak_date
            peak_value_at_trough = peak
    recovery_date = None
    after_trough = False
    for row in rows:
        if row["date"] == trough_date:
            after_trough = True
        if after_trough and row["nav"] >= peak_value_at_trough:
            recovery_date = row["date"]
            break
    return {
        "max_drawdown": max_dd,
        "drawdown_peak_date": str(peak_at_trough),
        "drawdown_trough_date": str(trough_date),
        "drawdown_recovery_date": None if recovery_date is None else str(recovery_date),
    }


def yearly_returns(df: pl.DataFrame) -> dict[str, float]:
    yearly = (
        df.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year", maintain_order=True)
        .agg(((pl.col("daily_ret") + 1.0).product() - 1.0).alias("return"))
        .sort("year")
    )
    return {str(row["year"]): row["return"] for row in yearly.iter_rows(named=True)}


def monthly_returns(df: pl.DataFrame) -> dict[str, float]:
    monthly = (
        df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(((pl.col("daily_ret") + 1.0).product() - 1.0).alias("return"))
        .sort("month")
    )
    return {str(row["month"]): row["return"] for row in monthly.iter_rows(named=True)}


def summarize_series(df: pl.DataFrame, label: str) -> dict[str, Any]:
    dd = drawdown_stats(df)
    monthly = monthly_returns(df)
    worst_months = sorted(monthly.items(), key=lambda item: item[1])[:8]
    best_months = sorted(monthly.items(), key=lambda item: item[1], reverse=True)[:8]
    start_date = df["date"].min()
    end_date = df["date"].max()
    years = (end_date - start_date).days / 365.25 if start_date and end_date else 0.0
    total_return = float(df["nav"][-1] - 1.0)
    daily_mean = float(df["daily_ret"].mean())
    daily_std = float(df["daily_ret"].std())
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0.0 else 0.0
    annualized_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR) if daily_std else 0.0
    sharpe = daily_mean / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR) if daily_std else 0.0
    calmar = annualized_return / abs(dd["max_drawdown"]) if dd["max_drawdown"] else 0.0
    return {
        "label": label,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "annualization_years": years,
        "max_drawdown": float(dd["max_drawdown"]),
        "drawdown_peak_date": dd["drawdown_peak_date"],
        "drawdown_trough_date": dd["drawdown_trough_date"],
        "drawdown_recovery_date": dd["drawdown_recovery_date"],
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "daily_win_rate": float((df["daily_ret"] > 0).mean()),
        "yearly_returns": yearly_returns(df),
        "worst_months": [{"month": month, "return": value} for month, value in worst_months],
        "best_months": [{"month": month, "return": value} for month, value in best_months],
    }


def build_nav(df: pl.DataFrame, ret_col: str) -> pl.DataFrame:
    return (
        df.select(["date", pl.col(ret_col).alias("daily_ret")])
        .with_columns((pl.col("daily_ret") + 1.0).cum_prod().alias("nav"))
        .sort("date")
    )


def combine_returns(
    returns: pl.DataFrame,
    *,
    p3_weight: float,
    pb3_col: str,
    label: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    pb3_weight = 1.0 - p3_weight
    df = (
        returns.select(["date", "p3_ret", pb3_col])
        .with_columns((pl.col("p3_ret") * p3_weight + pl.col(pb3_col) * pb3_weight).alias("daily_ret"))
        .with_columns((pl.col("daily_ret") + 1.0).cum_prod().alias("nav"))
        .sort("date")
    )
    summary = summarize_series(df, label)
    summary["p3_weight"] = p3_weight
    summary["pb3_weight"] = pb3_weight
    return df, summary


def nav_sample(df: pl.DataFrame, label: str) -> list[dict[str, Any]]:
    sampled = (
        df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month", maintain_order=True)
        .agg([pl.col("date").last(), pl.col("nav").last()])
        .sort("date")
    )
    return [
        {"date": str(row["date"]), "label": label, "nav": round(float(row["nav"]), 6)}
        for row in sampled.iter_rows(named=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="P3 + PB3 allocation diagnostic")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-capital", type=float, default=500_000.0)
    parser.add_argument("--weights", default="1.0,0.9,0.8,0.7,0.6,0.5")
    args = parser.parse_args()

    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    if not weights:
        raise ValueError("--weights must not be empty")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("--weights values must be between 0 and 1")

    returns = (
        load_returns(P3_EQUITY, "p3", initial_capital=args.initial_capital)
        .join(load_returns(PB3_RAW_EQUITY, "pb3_raw", initial_capital=args.initial_capital), on="date")
        .join(load_returns(PB3_GATED_EQUITY, "pb3_gated", initial_capital=args.initial_capital), on="date")
        .sort("date")
    )

    p3_nav = build_nav(returns, "p3_ret")
    pb3_raw_nav = build_nav(returns, "pb3_raw_ret")
    pb3_gated_nav = build_nav(returns, "pb3_gated_ret")

    standalone = {
        "p3_static": summarize_series(p3_nav, "P3 static strict"),
        "pb3_raw": summarize_series(pb3_raw_nav, "PB3 rolling raw"),
        "pb3_gated": summarize_series(pb3_gated_nav, "PB3 rolling gated"),
    }

    correlations = {
        "p3_vs_pb3_raw": float(returns.select(pl.corr("p3_ret", "pb3_raw_ret")).item()),
        "p3_vs_pb3_gated": float(returns.select(pl.corr("p3_ret", "pb3_gated_ret")).item()),
        "pb3_raw_vs_pb3_gated": float(returns.select(pl.corr("pb3_raw_ret", "pb3_gated_ret")).item()),
    }

    allocation_rows: list[dict[str, Any]] = []
    nav_samples: dict[str, list[dict[str, Any]]] = {
        "P3 static": nav_sample(p3_nav, "P3 static"),
        "PB3 raw": nav_sample(pb3_raw_nav, "PB3 raw"),
        "PB3 gated": nav_sample(pb3_gated_nav, "PB3 gated"),
    }
    selected_combo_navs: dict[str, pl.DataFrame] = {}

    for pb3_col, pb3_label in [("pb3_raw_ret", "PB3 raw"), ("pb3_gated_ret", "PB3 gated")]:
        for p3_weight in weights:
            label = f"P3 {int(round(p3_weight * 100))}% / {pb3_label} {int(round((1 - p3_weight) * 100))}%"
            combo_nav, summary = combine_returns(
                returns,
                p3_weight=p3_weight,
                pb3_col=pb3_col,
                label=label,
            )
            summary["pb3_variant"] = pb3_label
            allocation_rows.append(summary)
            if p3_weight in {0.8, 0.7, 0.6} and pb3_label == "PB3 gated":
                selected_combo_navs[label] = combo_nav
                nav_samples[label] = nav_sample(combo_nav, label)

    # Direct raw-vs-gated comparison for the same allocation weights.
    deltas = []
    by_key = {(row["p3_weight"], row["pb3_variant"]): row for row in allocation_rows}
    for p3_weight in weights:
        raw = by_key[(p3_weight, "PB3 raw")]
        gated = by_key[(p3_weight, "PB3 gated")]
        deltas.append(
            {
                "p3_weight": p3_weight,
                "pb3_weight": 1.0 - p3_weight,
                "total_return_delta": gated["total_return"] - raw["total_return"],
                "max_drawdown_delta": gated["max_drawdown"] - raw["max_drawdown"],
                "return_2025_delta": gated["yearly_returns"].get("2025", 0.0)
                - raw["yearly_returns"].get("2025", 0.0),
                "return_2026_delta": gated["yearly_returns"].get("2026", 0.0)
                - raw["yearly_returns"].get("2026", 0.0),
            }
        )

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Daily rebalanced synthetic allocation from Rust daily_equity.csv return series.",
        "artifacts": {
            "p3_equity": str(P3_EQUITY),
            "pb3_raw_equity": str(PB3_RAW_EQUITY),
            "pb3_gated_equity": str(PB3_GATED_EQUITY),
        },
        "date_range": {
            "start": str(returns["date"].min()),
            "end": str(returns["date"].max()),
            "trading_days": returns.height,
        },
        "standalone": standalone,
        "correlations": correlations,
        "allocations": allocation_rows,
        "gated_vs_raw_deltas": deltas,
        "nav_samples": nav_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "correlations": correlations}, ensure_ascii=False, indent=2))
    best = sorted(
        [row for row in allocation_rows if row["pb3_variant"] == "PB3 gated"],
        key=lambda row: (row["total_return"] + row["max_drawdown"], row["yearly_returns"].get("2026", 0.0)),
        reverse=True,
    )[:3]
    for row in best:
        print(
            f"{row['label']}: total={row['total_return']*100:+.2f}% "
            f"maxdd={row['max_drawdown']*100:.2f}% "
            f"2026={row['yearly_returns'].get('2026', 0.0)*100:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
