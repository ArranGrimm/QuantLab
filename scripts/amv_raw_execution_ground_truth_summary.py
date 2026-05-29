"""Summarize raw-execution rerun metrics for core AMV candidates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT


OUTPUT = ROOT / "reports" / "amv_raw_execution_ground_truth_summary.json"

RAW_RUNS = [
    {
        "name": "ref_p2_6td",
        "group": "core",
        "old_total_return_pct": 170.80,
        "old_max_drawdown_pct": 15.30,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143432_reference_p2_k0p5_b0_c0_r0/backtests/"
        "6td_static_strict_top3_no_stop_rawexec_20260529_143432_reference_p2_k0p5_b0_c0_r0",
    },
    {
        "name": "p3_6td",
        "group": "core",
        "old_total_return_pct": 201.69,
        "old_max_drawdown_pct": 13.52,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143434_candidate_p3_k0p5_b0_c0_r0/backtests/"
        "6td_static_strict_top3_no_stop_rawexec_20260529_143434_candidate_p3_k0p5_b0_c0_r0",
    },
    {
        "name": "context_combo_6td",
        "group": "core",
        "old_total_return_pct": 272.06,
        "old_max_drawdown_pct": 14.05,
        "rel_dir": "artifacts/amv_static_sleeve_signals/"
        "20260529_143517_p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0/backtests/"
        "6td_static_strict_top3_no_stop_rawexec_20260529_143517_p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0",
    },
    {
        "name": "pb3_gated_rolling",
        "group": "pullback",
        "old_total_return_pct": 109.73,
        "old_max_drawdown_pct": 16.20,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143548_pullback_p0_k0_pb3_cp1_rv0/backtests/"
        "6td_rolling21_refill_top10_no_stop_rawexec_20260529_143548_pullback_p0_k0_pb3_cp1_rv0",
    },
    {
        "name": "limit_first_board_5td",
        "group": "limit_ecology",
        "old_total_return_pct": 183.57,
        "old_max_drawdown_pct": 45.35,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143644_limit_first_board_pullback/backtests/"
        "5td_static_strict_top3_no_stop_rawexec_20260529_143644_limit_first_board_pullback",
    },
    {
        "name": "limit_weakgate_5td",
        "group": "limit_ecology",
        "old_total_return_pct": 158.02,
        "old_max_drawdown_pct": 34.14,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143647_limit_first_board_pullback_weakgate/backtests/"
        "5td_static_strict_top3_no_stop_rawexec_20260529_143647_limit_first_board_pullback_weakgate",
    },
    {
        "name": "limit_weaktop1_5td",
        "group": "limit_ecology",
        "old_total_return_pct": 202.64,
        "old_max_drawdown_pct": 50.54,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143649_limit_first_board_pullback_weaktop1/backtests/"
        "5td_static_strict_top3_no_stop_rawexec_20260529_143649_limit_first_board_pullback_weaktop1",
    },
    {
        "name": "limit_weaktier_5td",
        "group": "limit_ecology",
        "old_total_return_pct": 226.38,
        "old_max_drawdown_pct": 51.30,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143651_limit_first_board_pullback_weaktier/backtests/"
        "5td_static_strict_top3_no_stop_rawexec_20260529_143651_limit_first_board_pullback_weaktier",
    },
    {
        "name": "limit_weakscorepen_5td",
        "group": "limit_ecology",
        "old_total_return_pct": 157.07,
        "old_max_drawdown_pct": 41.70,
        "rel_dir": "artifacts/amv_static_sleeve_signals/20260529_143654_limit_first_board_pullback_weakscorepen/backtests/"
        "5td_static_strict_top3_no_stop_rawexec_20260529_143654_limit_first_board_pullback_weakscorepen",
    },
]


def win_long_path(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    if len(text) < 240 or text.startswith("\\\\?\\"):
        return resolved
    return Path(f"\\\\?\\{text}")


def yearly_returns(backtest_dir: Path) -> dict[str, float]:
    equity = pl.read_csv(win_long_path(backtest_dir / "daily_equity.csv"), try_parse_dates=True)
    if equity.schema["date"] != pl.Date:
        equity = equity.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    equity = equity.sort("date")
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


def load_run(row: dict[str, Any]) -> dict[str, Any]:
    backtest_dir = ROOT / row["rel_dir"]
    report = json.loads(win_long_path(backtest_dir / "report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    extra = report.get("extra", {})
    total_return_pct = round(float(metrics["total_return_pct"]), 2)
    max_drawdown_pct = round(float(metrics["max_drawdown_pct"]), 2)
    return {
        "name": row["name"],
        "group": row["group"],
        "backtest_dir": row["rel_dir"],
        "execution_price_basis": extra.get(
            "execution_price_basis",
            report.get("backtest_config", {}).get("execution_price_basis"),
        ),
        "total_return_pct": total_return_pct,
        "gross_return_pct": round(float(metrics["gross_return_pct"]), 2),
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": round(float(metrics["win_rate_pct"]), 2),
        "total_trades": int(metrics["total_trades"]),
        "total_costs": round(float(metrics["total_costs"]), 2),
        "limit_up_blocked": int(extra.get("limit_up_blocked", 0)),
        "limit_up_days": int(extra.get("limit_up_days", 0)),
        "old_total_return_pct": row["old_total_return_pct"],
        "old_max_drawdown_pct": row["old_max_drawdown_pct"],
        "delta_total_return_vs_old_pp": round(total_return_pct - row["old_total_return_pct"], 2),
        "delta_max_drawdown_vs_old_pp": round(max_drawdown_pct - row["old_max_drawdown_pct"], 2),
        "drawdown_peak_date": metrics["drawdown_peak_date"],
        "drawdown_trough_date": metrics["drawdown_trough_date"],
        "drawdown_recovery_date": metrics["drawdown_recovery_date"],
        "yearly_returns_pct": yearly_returns(backtest_dir),
    }


def main() -> int:
    rows = [load_run(row) for row in RAW_RUNS]
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "Core AMV raw-execution ground truth rerun",
        "price_basis": "raw_ohlc_pre_close for all new artifacts",
        "runs": rows,
        "interpretation": {
            "core": (
                "Raw execution materially reduces static Ref/P3/context returns, but the context combo remains "
                "the strongest of the core static candidates."
            ),
            "pb3": (
                "PB3 gated rolling return is lower under raw execution than the previous adjusted-execution "
                "allocation diagnostic input; it still needs portfolio-level re-evaluation."
            ),
            "limit_ecology": (
                "First-board pullback event sleeve loses most of its adjusted-execution appeal under raw execution; "
                "weak-window variants are not allocation-ready."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
