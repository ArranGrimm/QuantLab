"""Cadence robustness for P3 raw vs sector-tailwind rerank.

This is a Python-side no-cost sensitivity diagnostic. It replays static 6td
cadences from signal.parquet by varying the first eligible entry offset.
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_SIGNAL = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0"
)
DEFAULT_RERANK_SIGNAL = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260528_144538_p3_sector_w10_b0p4_penalty_0p02"
)
DEFAULT_OUTPUT = ROOT / "reports" / "amv_p3_sector_tailwind_cadence.json"


@dataclass(frozen=True)
class SignalInput:
    label: str
    path: Path


def resolve_signal_path(path: Path) -> Path:
    if path.is_dir():
        parquet = path / "signal.parquet"
    else:
        parquet = path
    if not parquet.exists():
        raise FileNotFoundError(f"signal.parquet not found: {parquet}")
    return parquet


def price_limit_pct_expr() -> pl.Expr:
    code = pl.col("code")
    return (
        pl.when(code.str.starts_with("bj."))
        .then(0.30)
        .when(code.str.starts_with("sh.688") | code.str.starts_with("sz.300") | code.str.starts_with("sz.301"))
        .then(0.20)
        .otherwise(0.10)
    )


def max_drawdown(nav_values: list[float]) -> float:
    peak = nav_values[0] if nav_values else 1.0
    max_dd = 0.0
    for value in nav_values:
        if value > peak:
            peak = value
        dd = value / peak - 1.0 if peak else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        "min": float(s[0]),
        "median": float(median),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
    }


def load_daily_cycle_returns(signal_path: Path, *, start_date: str, hold_trading_days: int) -> tuple[pl.DataFrame, list[date]]:
    df = pl.read_parquet(signal_path).sort(["date", "code"])
    trading_dates = df.select("date").unique().sort("date")["date"].to_list()
    exit_dates = [
        trading_dates[idx + hold_trading_days] if idx + hold_trading_days < len(trading_dates) else None
        for idx in range(len(trading_dates))
    ]
    exit_map = pl.DataFrame(
        {
            "date": trading_dates,
            "exit_date": exit_dates,
        },
        schema={"date": pl.Date, "exit_date": pl.Date},
    ).drop_nulls("exit_date")

    exit_prices = df.select(
        [
            "code",
            pl.col("date").alias("exit_date"),
            pl.col("close_adj").alias("exit_close_adj"),
        ]
    )

    selected = (
        df.filter(
            (pl.col("date") >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            & pl.col("is_signal")
            & (pl.col("rank") <= 3)
            & pl.col("is_bull_regime")
        )
        .join(exit_map, on="date", how="inner")
        .join(exit_prices, on=["code", "exit_date"], how="inner")
        .with_columns(
            [
                (pl.col("open_adj") / pl.col("pre_close_adj") - 1.0).alias("open_gap"),
                price_limit_pct_expr().alias("limit_pct"),
            ]
        )
        .with_columns((pl.col("open_gap") >= (pl.col("limit_pct") - 0.002)).alias("is_open_limit_up"))
        .filter(~pl.col("is_open_limit_up"))
        .with_columns((pl.col("exit_close_adj") / pl.col("open_adj") - 1.0).alias("trade_ret"))
    )

    daily = (
        selected.group_by("date")
        .agg(
            [
                pl.col("exit_date").first(),
                pl.len().alias("trade_count"),
                pl.col("trade_ret").mean().alias("cycle_ret"),
                pl.col("trade_ret").min().alias("min_trade_ret"),
                pl.col("trade_ret").max().alias("max_trade_ret"),
            ]
        )
        .sort("date")
    )
    return daily, trading_dates


def simulate_offset_path(
    daily: pl.DataFrame,
    trading_dates: list[date],
    *,
    offset: int,
) -> dict[str, Any]:
    rows = daily.to_dicts()
    if not rows or offset >= len(rows):
        return {
            "offset": offset,
            "cycles": 0,
            "first_entry": None,
            "last_entry": None,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "mean_cycle_ret": 0.0,
            "median_cycle_ret": 0.0,
            "worst_cycle_ret": 0.0,
            "best_cycle_ret": 0.0,
        }

    row_dates = [row["date"] for row in rows]
    selected: list[dict[str, Any]] = []
    idx = offset
    while idx < len(rows):
        row = rows[idx]
        selected.append(row)
        exit_date = row["exit_date"]
        next_idx = bisect.bisect_right(row_dates, exit_date)
        idx = next_idx

    cycle_returns = [float(row["cycle_ret"]) for row in selected]
    nav = 1.0
    nav_values = [nav]
    for ret in cycle_returns:
        nav *= 1.0 + ret
        nav_values.append(nav)

    return {
        "offset": offset,
        "cycles": len(selected),
        "first_entry": str(selected[0]["date"]) if selected else None,
        "last_entry": str(selected[-1]["date"]) if selected else None,
        "total_return": nav - 1.0,
        "max_drawdown": max_drawdown(nav_values),
        "mean_cycle_ret": float(sum(cycle_returns) / len(cycle_returns)) if cycle_returns else 0.0,
        "median_cycle_ret": summarize_values(cycle_returns)["median"],
        "worst_cycle_ret": min(cycle_returns) if cycle_returns else 0.0,
        "best_cycle_ret": max(cycle_returns) if cycle_returns else 0.0,
    }


def analyze_signal(signal: SignalInput, *, start_date: str, hold_trading_days: int, offset_count: int) -> dict[str, Any]:
    signal_path = resolve_signal_path(signal.path)
    daily, trading_dates = load_daily_cycle_returns(
        signal_path,
        start_date=start_date,
        hold_trading_days=hold_trading_days,
    )
    paths = [
        simulate_offset_path(daily, trading_dates, offset=offset)
        for offset in range(offset_count)
    ]
    total_returns = [path["total_return"] for path in paths]
    max_drawdowns = [path["max_drawdown"] for path in paths]
    return {
        "label": signal.label,
        "signal_path": str(signal_path),
        "signal_days": daily.height,
        "daily_cycle_distribution": {
            "mean_cycle_ret": float(daily["cycle_ret"].mean()) if daily.height else 0.0,
            "median_cycle_ret": float(daily["cycle_ret"].median()) if daily.height else 0.0,
            "worst_cycle_ret": float(daily["cycle_ret"].min()) if daily.height else 0.0,
            "best_cycle_ret": float(daily["cycle_ret"].max()) if daily.height else 0.0,
            "positive_day_share": float((daily["cycle_ret"] > 0).mean()) if daily.height else 0.0,
        },
        "static_offset_summary": {
            "offset_count": offset_count,
            "worst_total_return": summarize_values(total_returns)["min"],
            "median_total_return": summarize_values(total_returns)["median"],
            "best_total_return": summarize_values(total_returns)["max"],
            "mean_total_return": summarize_values(total_returns)["mean"],
            "worst_max_drawdown": summarize_values(max_drawdowns)["min"],
            "median_max_drawdown": summarize_values(max_drawdowns)["median"],
            "best_max_drawdown": summarize_values(max_drawdowns)["max"],
            "positive_paths": sum(1 for value in total_returns if value > 0),
        },
        "static_offset_paths": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P3 sector tailwind cadence robustness")
    parser.add_argument("--raw-signal", type=Path, default=DEFAULT_RAW_SIGNAL)
    parser.add_argument("--rerank-signal", type=Path, default=DEFAULT_RERANK_SIGNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-label", default="P3 raw")
    parser.add_argument("--rerank-label", default="P3 sector tailwind 10d/bottom40/0.02")
    parser.add_argument(
        "--purpose",
        default="P3 raw vs sector-tailwind rerank Python-side static cadence sensitivity; no costs.",
    )
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--hold-trading-days", type=int, default=6)
    parser.add_argument("--offset-count", type=int, default=7)
    args = parser.parse_args()

    raw = analyze_signal(
        SignalInput(args.raw_label, args.raw_signal),
        start_date=args.start_date,
        hold_trading_days=args.hold_trading_days,
        offset_count=args.offset_count,
    )
    rerank = analyze_signal(
        SignalInput(args.rerank_label, args.rerank_signal),
        start_date=args.start_date,
        hold_trading_days=args.hold_trading_days,
        offset_count=args.offset_count,
    )

    pair_paths = []
    for raw_path, rerank_path in zip(raw["static_offset_paths"], rerank["static_offset_paths"], strict=True):
        pair_paths.append(
            {
                "offset": raw_path["offset"],
                "raw_total_return": raw_path["total_return"],
                "rerank_total_return": rerank_path["total_return"],
                "delta_total_return": rerank_path["total_return"] - raw_path["total_return"],
                "raw_max_drawdown": raw_path["max_drawdown"],
                "rerank_max_drawdown": rerank_path["max_drawdown"],
                "delta_max_drawdown": rerank_path["max_drawdown"] - raw_path["max_drawdown"],
                "raw_cycles": raw_path["cycles"],
                "rerank_cycles": rerank_path["cycles"],
            }
        )

    deltas = [row["delta_total_return"] for row in pair_paths]
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": args.purpose,
        "assumptions": {
            "start_date": args.start_date,
            "hold_trading_days": args.hold_trading_days,
            "entry": "execution date open_adj from shifted signal.parquet",
            "exit": "entry date + hold_trading_days close_adj",
            "selection": "rank <= 3, is_signal, is_bull_regime, skip open limit-up, no refill",
            "costs": "excluded",
        },
        "raw": raw,
        "rerank": rerank,
        "pair_offset_paths": pair_paths,
        "pair_summary": {
            "offset_count": len(pair_paths),
            "positive_delta_offsets": sum(1 for value in deltas if value > 0),
            "worst_delta_total_return": summarize_values(deltas)["min"],
            "median_delta_total_return": summarize_values(deltas)["median"],
            "best_delta_total_return": summarize_values(deltas)["max"],
            "mean_delta_total_return": summarize_values(deltas)["mean"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(args.output)
    print(
        "raw median={:.2f}% worst={:.2f}% | rerank median={:.2f}% worst={:.2f}% | positive_delta={}/{}".format(
            raw["static_offset_summary"]["median_total_return"] * 100,
            raw["static_offset_summary"]["worst_total_return"] * 100,
            rerank["static_offset_summary"]["median_total_return"] * 100,
            rerank["static_offset_summary"]["worst_total_return"] * 100,
            payload["pair_summary"]["positive_delta_offsets"],
            len(pair_paths),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
