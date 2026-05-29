"""Summarize first-board pullback risk-rerank Rust variants."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT


DEFAULT_OUTPUT = ROOT / "reports" / "amv_limit_first_board_risk_variant_scan.json"


VARIANTS = [
    (
        "base_5td",
        "artifacts/amv_static_sleeve_signals/20260529_111655_limit_first_board_pullback/backtests/"
        "5td_static_strict_top3_no_stop_20260529_111655_limit_first_board_pullback",
    ),
    (
        "atrpen1_5td",
        "artifacts/amv_static_sleeve_signals/20260529_134454_limit_first_board_pullback_atrpen1/backtests/"
        "5td_static_strict_top3_no_stop_20260529_134454_limit_first_board_pullback_atrpen1",
    ),
    (
        "atrpen2_5td",
        "artifacts/amv_static_sleeve_signals/20260529_134455_limit_first_board_pullback_atrpen2/backtests/"
        "5td_static_strict_top3_no_stop_20260529_134455_limit_first_board_pullback_atrpen2",
    ),
    (
        "medium128pen_5td",
        "artifacts/amv_static_sleeve_signals/20260529_134457_limit_first_board_pullback_medium128pen/backtests/"
        "5td_static_strict_top3_no_stop_20260529_134457_limit_first_board_pullback_medium128pen",
    ),
    (
        "staleqpen_5td",
        "artifacts/amv_static_sleeve_signals/20260529_134459_limit_first_board_pullback_staleqpen/backtests/"
        "5td_static_strict_top3_no_stop_20260529_134459_limit_first_board_pullback_staleqpen",
    ),
    (
        "riskmix_5td",
        "artifacts/amv_static_sleeve_signals/20260529_134501_limit_first_board_pullback_riskmix/backtests/"
        "5td_static_strict_top3_no_stop_20260529_134501_limit_first_board_pullback_riskmix",
    ),
    (
        "weakgate_5td",
        "artifacts/amv_static_sleeve_signals/20260529_141322_limit_first_board_pullback_weakgate/backtests/"
        "5td_static_strict_top3_no_stop_20260529_141322_limit_first_board_pullback_weakgate",
    ),
    (
        "weaktop1_5td",
        "artifacts/amv_static_sleeve_signals/20260529_141324_limit_first_board_pullback_weaktop1/backtests/"
        "5td_static_strict_top3_no_stop_20260529_141324_limit_first_board_pullback_weaktop1",
    ),
    (
        "weaktier_5td",
        "artifacts/amv_static_sleeve_signals/20260529_141326_limit_first_board_pullback_weaktier/backtests/"
        "5td_static_strict_top3_no_stop_20260529_141326_limit_first_board_pullback_weaktier",
    ),
    (
        "weakscorepen_5td",
        "artifacts/amv_static_sleeve_signals/20260529_141328_limit_first_board_pullback_weakscorepen/backtests/"
        "5td_static_strict_top3_no_stop_20260529_141328_limit_first_board_pullback_weakscorepen",
    ),
    (
        "quality_6td_defensive_endpoint",
        "artifacts/amv_static_sleeve_signals/20260529_112355_limit_first_board_pullback_quality/backtests/"
        "6td_static_strict_top3_no_stop_20260529_112355_limit_first_board_pullback_quality",
    ),
]


def yearly_returns(backtest_dir: Path) -> dict[str, float]:
    equity = pl.read_csv(backtest_dir / "daily_equity.csv", try_parse_dates=True).sort("date")
    rows = (
        equity.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                pl.col("total_value").first().alias("first_value"),
                pl.col("total_value").last().alias("last_value"),
            ]
        )
        .sort("year")
    )
    return {
        str(row["year"]): round((float(row["last_value"]) / float(row["first_value"]) - 1.0) * 100.0, 2)
        for row in rows.iter_rows(named=True)
    }


def load_variant(name: str, rel_dir: str) -> dict[str, Any]:
    backtest_dir = ROOT / rel_dir
    report = json.loads((backtest_dir / "report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    extra = report.get("extra", {})
    return {
        "name": name,
        "backtest_dir": rel_dir,
        "total_return_pct": round(float(metrics["total_return_pct"]), 2),
        "gross_return_pct": round(float(metrics["gross_return_pct"]), 2),
        "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 2),
        "win_rate_pct": round(float(metrics["win_rate_pct"]), 2),
        "total_trades": int(metrics["total_trades"]),
        "total_costs": round(float(metrics["total_costs"]), 2),
        "limit_up_blocked": int(extra.get("limit_up_blocked", 0)),
        "limit_up_days": int(extra.get("limit_up_days", 0)),
        "drawdown_peak_date": metrics["drawdown_peak_date"],
        "drawdown_trough_date": metrics["drawdown_trough_date"],
        "drawdown_recovery_date": metrics["drawdown_recovery_date"],
        "yearly_returns_pct": yearly_returns(backtest_dir),
    }


def main() -> int:
    rows = [load_variant(name, rel_dir) for name, rel_dir in VARIANTS]
    base = next(row for row in rows if row["name"] == "base_5td")
    for row in rows:
        row["delta_total_return_vs_base_pp"] = round(row["total_return_pct"] - base["total_return_pct"], 2)
        row["delta_max_drawdown_vs_base_pp"] = round(row["max_drawdown_pct"] - base["max_drawdown_pct"], 2)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "Rust 5td static strict scan for first-board pullback risk rerank variants",
        "variants": rows,
        "interpretation": {
            "medium128": (
                "medium128 is not useful as a standalone first-board pullback penalty in this scan: "
                "it lowers total return and does not reduce MaxDD."
            ),
            "atr_penalty": (
                "ATR penalty improves ranking alpha at weight 2.0 but does not solve drawdown; "
                "this suggests high ATR should be handled with regime interaction, not a simple static penalty."
            ),
            "next_step": (
                "The first top1/downshift implementation is rejected because it worsens MaxDD. Continue only by "
                "refining weak-window definition or testing true position sizing/downsize; do not move to rolling/refill yet."
            ),
        },
    }
    DEFAULT_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(DEFAULT_OUTPUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
