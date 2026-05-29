"""Annual restart cadence sensitivity for static AMV signals.

This diagnostic replays each calendar year independently. For every signal, it
starts from offset 0..N-1 within that year and then follows the usual static
6td cadence. It is intentionally no-cost and Python-side, so it can quickly
test whether a strategy depends on a single continuous 2021-start cadence.
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.amv_sector_tailwind_cadence import (
    load_daily_cycle_returns,
    max_drawdown,
    resolve_signal_path,
    summarize_values,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "amv_p3_annual_restart_cadence.json"


@dataclass(frozen=True)
class SignalSpec:
    label: str
    path: Path


def parse_signal(value: str) -> SignalSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("signal must use label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("signal label must not be empty")
    return SignalSpec(label=label, path=Path(path).expanduser())


def simulate_year_offset(rows: list[dict[str, Any]], *, year: int, offset: int) -> dict[str, Any]:
    year_rows = [row for row in rows if row["date"].year == year]
    if offset >= len(year_rows):
        return {
            "offset": offset,
            "cycles": 0,
            "first_entry": None,
            "last_entry": None,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "mean_cycle_ret": 0.0,
            "worst_cycle_ret": 0.0,
            "best_cycle_ret": 0.0,
        }

    year_dates = [row["date"] for row in year_rows]
    selected: list[dict[str, Any]] = []
    idx = offset
    while idx < len(year_rows):
        row = year_rows[idx]
        selected.append(row)
        idx = bisect.bisect_right(year_dates, row["exit_date"])

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
        "worst_cycle_ret": min(cycle_returns) if cycle_returns else 0.0,
        "best_cycle_ret": max(cycle_returns) if cycle_returns else 0.0,
    }


def summarize_year(paths: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [path["total_return"] for path in paths]
    drawdowns = [path["max_drawdown"] for path in paths]
    cycles = [path["cycles"] for path in paths]
    return {
        "worst_total_return": summarize_values(returns)["min"],
        "median_total_return": summarize_values(returns)["median"],
        "best_total_return": summarize_values(returns)["max"],
        "mean_total_return": summarize_values(returns)["mean"],
        "positive_offsets": sum(1 for value in returns if value > 0),
        "offset_count": len(paths),
        "worst_max_drawdown": summarize_values(drawdowns)["min"],
        "median_max_drawdown": summarize_values(drawdowns)["median"],
        "best_max_drawdown": summarize_values(drawdowns)["max"],
        "min_cycles": min(cycles) if cycles else 0,
        "median_cycles": summarize_values([float(value) for value in cycles])["median"],
        "max_cycles": max(cycles) if cycles else 0,
    }


def analyze_signal(
    signal: SignalSpec,
    *,
    start_date: str,
    hold_trading_days: int,
    offset_count: int,
) -> dict[str, Any]:
    path = signal.path
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    signal_path = resolve_signal_path(path)
    daily, _ = load_daily_cycle_returns(
        signal_path,
        start_date=start_date,
        hold_trading_days=hold_trading_days,
    )
    rows = daily.to_dicts()
    years = sorted({row["date"].year for row in rows})
    per_year = []
    for year in years:
        paths = [simulate_year_offset(rows, year=year, offset=offset) for offset in range(offset_count)]
        per_year.append(
            {
                "year": year,
                "summary": summarize_year(paths),
                "offset_paths": paths,
            }
        )

    return {
        "label": signal.label,
        "signal_path": str(signal_path),
        "signal_days": daily.height,
        "years": per_year,
    }


def add_vs_raw(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    raw_by_year = {
        year_record["year"]: year_record["summary"]
        for year_record in records[0]["years"]
    }
    for record in records[1:]:
        for year_record in record["years"]:
            raw = raw_by_year.get(year_record["year"])
            if not raw:
                continue
            summary = year_record["summary"]
            year_record["vs_raw"] = {
                "worst_total_return_delta": summary["worst_total_return"] - raw["worst_total_return"],
                "median_total_return_delta": summary["median_total_return"] - raw["median_total_return"],
                "best_total_return_delta": summary["best_total_return"] - raw["best_total_return"],
                "positive_offsets_delta": summary["positive_offsets"] - raw["positive_offsets"],
            }


def main() -> int:
    parser = argparse.ArgumentParser(description="Annual restart cadence diagnostic for AMV static signals")
    parser.add_argument("--signal", action="append", type=parse_signal, required=True, help="label=signal_dir_or_parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--hold-trading-days", type=int, default=6)
    parser.add_argument("--offset-count", type=int, default=7)
    args = parser.parse_args()

    records = [
        analyze_signal(
            signal,
            start_date=args.start_date,
            hold_trading_days=args.hold_trading_days,
            offset_count=args.offset_count,
        )
        for signal in args.signal
    ]
    add_vs_raw(records)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Annual independent restart static cadence sensitivity; Python-side no-cost diagnostic.",
        "start_date": args.start_date,
        "hold_trading_days": args.hold_trading_days,
        "offset_count": args.offset_count,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(args.output)
    for record in records:
        print(f"\n{record['label']}")
        for year_record in record["years"]:
            summary = year_record["summary"]
            print(
                f"  {year_record['year']}: "
                f"worst={summary['worst_total_return']*100:+.2f}% "
                f"median={summary['median_total_return']*100:+.2f}% "
                f"best={summary['best_total_return']*100:+.2f}% "
                f"positive={summary['positive_offsets']}/{summary['offset_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
