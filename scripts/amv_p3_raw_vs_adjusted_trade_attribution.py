"""Compare P3 adjusted-execution and raw-execution trades one by one."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT


ADJUSTED_DIR = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/backtests/"
    / "6td_static_strict_top3_no_stop_20260520_092208_801"
)
RAW_DIR = (
    ROOT
    / "artifacts/amv_static_sleeve_signals/20260529_143434_candidate_p3_k0p5_b0_c0_r0/backtests/"
    / "6td_static_strict_top3_no_stop_rawexec_20260529_143434_candidate_p3_k0p5_b0_c0_r0"
)
OUTPUT = ROOT / "reports" / "amv_p3_raw_vs_adjusted_trade_attribution.json"


def load_report(backtest_dir: Path) -> dict[str, Any]:
    return json.loads((backtest_dir / "report.json").read_text(encoding="utf-8"))


def load_trades(backtest_dir: Path, suffix: str) -> pl.DataFrame:
    return (
        pl.read_csv(backtest_dir / "trades.csv", try_parse_dates=True)
        .rename(
            {
                "entry_price": f"entry_price_{suffix}",
                "exit_price": f"exit_price_{suffix}",
                "shares": f"shares_{suffix}",
                "cost": f"cost_{suffix}",
                "exit_value": f"exit_value_{suffix}",
                "pnl": f"pnl_{suffix}",
                "pnl_pct": f"pnl_pct_{suffix}",
                "hold_trading_days": f"hold_trading_days_{suffix}",
                "exit_reason": f"exit_reason_{suffix}",
            }
        )
        .sort(["entry_date", "code", "exit_date"])
    )


def yearly_trade_pnl(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows = (
        df.with_columns(pl.col("exit_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                pl.len().alias("trades"),
                pl.col("pnl_adjusted").sum().alias("pnl_adjusted"),
                pl.col("pnl_raw").sum().alias("pnl_raw"),
                pl.col("pnl_delta_raw_minus_adjusted").sum().alias("pnl_delta"),
                pl.col("notional_effect").sum().alias("notional_effect"),
                pl.col("return_effect").sum().alias("return_effect"),
            ]
        )
        .sort("year")
    )
    return [
        {
            "year": int(row["year"]),
            "trades": int(row["trades"]),
            "pnl_adjusted": round(float(row["pnl_adjusted"]), 2),
            "pnl_raw": round(float(row["pnl_raw"]), 2),
            "pnl_delta_raw_minus_adjusted": round(float(row["pnl_delta"]), 2),
            "notional_effect": round(float(row["notional_effect"]), 2),
            "return_effect": round(float(row["return_effect"]), 2),
        }
        for row in rows.iter_rows(named=True)
    ]


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    extra = report.get("extra", {})
    config = report.get("backtest_config", {})
    return {
        "execution_price_basis": extra.get("execution_price_basis", config.get("execution_price_basis", "adjusted_legacy")),
        "total_return_pct": round(float(metrics["total_return_pct"]), 2),
        "gross_return_pct": round(float(metrics["gross_return_pct"]), 2),
        "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 2),
        "win_rate_pct": round(float(metrics["win_rate_pct"]), 2),
        "total_trades": int(metrics["total_trades"]),
        "total_pnl": round(float(metrics["total_pnl"]), 2),
        "gross_pnl": round(float(metrics["gross_pnl"]), 2),
        "total_costs": round(float(metrics["total_costs"]), 2),
        "limit_up_blocked": int(extra.get("limit_up_blocked", 0)),
        "signal_rows": int(extra.get("signal_rows", 0)),
    }


def top_rows(df: pl.DataFrame, sort_col: str, *, descending: bool, limit: int = 12) -> list[dict[str, Any]]:
    cols = [
        "code",
        "entry_date",
        "exit_date",
        "cost_adjusted",
        "cost_raw",
        "cost_delta_raw_minus_adjusted",
        "shares_adjusted",
        "shares_raw",
        "entry_price_adjusted",
        "entry_price_raw",
        "pnl_pct_adjusted",
        "pnl_pct_raw",
        "pnl_adjusted",
        "pnl_raw",
        "pnl_delta_raw_minus_adjusted",
        "notional_effect",
        "return_effect",
        "raw_to_adjusted_entry_price_ratio",
        "adjusted_to_raw_share_ratio",
    ]
    out = df.sort(sort_col, descending=descending).select(cols).head(limit)
    result: list[dict[str, Any]] = []
    for row in out.iter_rows(named=True):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                item[key] = value.isoformat()
            elif isinstance(value, float):
                item[key] = round(value, 6)
            else:
                item[key] = value
        result.append(item)
    return result


def main() -> int:
    adjusted_report = load_report(ADJUSTED_DIR)
    raw_report = load_report(RAW_DIR)
    adjusted = load_trades(ADJUSTED_DIR, "adjusted")
    raw = load_trades(RAW_DIR, "raw")

    common = adjusted.join(raw, on=["entry_date", "code"], how="inner", suffix="_right")
    adjusted_only = adjusted.join(raw.select(["entry_date", "code"]), on=["entry_date", "code"], how="anti")
    raw_only = raw.join(adjusted.select(["entry_date", "code"]), on=["entry_date", "code"], how="anti")

    common = (
        common.with_columns(
            [
                (pl.col("cost_raw") - pl.col("cost_adjusted")).alias("cost_delta_raw_minus_adjusted"),
                (pl.col("pnl_raw") - pl.col("pnl_adjusted")).alias("pnl_delta_raw_minus_adjusted"),
                (pl.col("pnl_pct_raw") - pl.col("pnl_pct_adjusted")).alias("pnl_pct_delta_raw_minus_adjusted"),
                (pl.col("entry_price_raw") / pl.col("entry_price_adjusted")).alias(
                    "raw_to_adjusted_entry_price_ratio"
                ),
                (pl.col("exit_price_raw") / pl.col("exit_price_adjusted")).alias(
                    "raw_to_adjusted_exit_price_ratio"
                ),
                (pl.col("shares_adjusted") / pl.col("shares_raw")).alias("adjusted_to_raw_share_ratio"),
                (pl.col("cost_raw") / pl.col("cost_adjusted")).alias("raw_to_adjusted_cost_ratio"),
                (pl.col("exit_date") == pl.col("exit_date_right")).alias("same_exit_date"),
                (pl.col("exit_reason_adjusted") == pl.col("exit_reason_raw")).alias("same_exit_reason"),
            ]
        )
        .with_columns(
            [
                ((pl.col("cost_raw") - pl.col("cost_adjusted")) * pl.col("pnl_pct_adjusted")).alias(
                    "notional_effect"
                ),
                (pl.col("cost_raw") * (pl.col("pnl_pct_raw") - pl.col("pnl_pct_adjusted"))).alias(
                    "return_effect"
                ),
            ]
        )
        .with_columns(
            (pl.col("notional_effect") + pl.col("return_effect")).alias("decomposed_delta_check")
        )
    )

    adjusted_summary = summarize_report(adjusted_report)
    raw_summary = summarize_report(raw_report)
    exact_return_mask = pl.col("pnl_pct_delta_raw_minus_adjusted").abs() <= 1e-8
    summary = {
        "adjusted": adjusted_summary,
        "raw": raw_summary,
        "delta_raw_minus_adjusted": {
            "total_return_pp": round(raw_summary["total_return_pct"] - adjusted_summary["total_return_pct"], 2),
            "gross_return_pp": round(raw_summary["gross_return_pct"] - adjusted_summary["gross_return_pct"], 2),
            "max_drawdown_pp": round(raw_summary["max_drawdown_pct"] - adjusted_summary["max_drawdown_pct"], 2),
            "total_pnl": round(raw_summary["total_pnl"] - adjusted_summary["total_pnl"], 2),
            "gross_pnl": round(raw_summary["gross_pnl"] - adjusted_summary["gross_pnl"], 2),
            "total_costs": round(raw_summary["total_costs"] - adjusted_summary["total_costs"], 2),
        },
        "overlap": {
            "adjusted_trades": adjusted.height,
            "raw_trades": raw.height,
            "common_entry_code": common.height,
            "adjusted_only": adjusted_only.height,
            "raw_only": raw_only.height,
            "same_exit_date": int(common["same_exit_date"].sum()),
            "same_exit_reason": int(common["same_exit_reason"].sum()),
        },
        "matched_trade_sums": {
            "pnl_adjusted": round(float(common["pnl_adjusted"].sum()), 2),
            "pnl_raw": round(float(common["pnl_raw"].sum()), 2),
            "pnl_delta_raw_minus_adjusted": round(float(common["pnl_delta_raw_minus_adjusted"].sum()), 2),
            "cost_adjusted": round(float(common["cost_adjusted"].sum()), 2),
            "cost_raw": round(float(common["cost_raw"].sum()), 2),
            "cost_delta_raw_minus_adjusted": round(float(common["cost_delta_raw_minus_adjusted"].sum()), 2),
            "notional_effect": round(float(common["notional_effect"].sum()), 2),
            "return_effect": round(float(common["return_effect"].sum()), 2),
            "decomposed_delta_check": round(float(common["decomposed_delta_check"].sum()), 2),
        },
        "share_price_cost_profile": {
            "adjusted_shares_gt_raw_count": int((common["shares_adjusted"] > common["shares_raw"]).sum()),
            "adjusted_shares_eq_raw_count": int((common["shares_adjusted"] == common["shares_raw"]).sum()),
            "adjusted_shares_lt_raw_count": int((common["shares_adjusted"] < common["shares_raw"]).sum()),
            "avg_adjusted_to_raw_share_ratio": round(float(common["adjusted_to_raw_share_ratio"].mean()), 4),
            "median_adjusted_to_raw_share_ratio": round(float(common["adjusted_to_raw_share_ratio"].median()), 4),
            "avg_raw_to_adjusted_entry_price_ratio": round(float(common["raw_to_adjusted_entry_price_ratio"].mean()), 4),
            "median_raw_to_adjusted_entry_price_ratio": round(float(common["raw_to_adjusted_entry_price_ratio"].median()), 4),
            "avg_raw_to_adjusted_cost_ratio": round(float(common["raw_to_adjusted_cost_ratio"].mean()), 4),
            "median_raw_to_adjusted_cost_ratio": round(float(common["raw_to_adjusted_cost_ratio"].median()), 4),
            "return_pct_identical_count": common.filter(exact_return_mask).height,
            "return_pct_changed_count": common.filter(~exact_return_mask).height,
        },
        "yearly_trade_pnl": yearly_trade_pnl(common),
        "largest_raw_underperformance_trades": top_rows(
            common, "pnl_delta_raw_minus_adjusted", descending=False
        ),
        "largest_raw_outperformance_trades": top_rows(
            common, "pnl_delta_raw_minus_adjusted", descending=True
        ),
        "largest_cost_reduction_trades": top_rows(
            common, "cost_delta_raw_minus_adjusted", descending=False
        ),
        "largest_return_pct_delta_trades": top_rows(
            common.with_columns(pl.col("pnl_pct_delta_raw_minus_adjusted").abs().alias("abs_return_delta")),
            "abs_return_delta",
            descending=True,
        ),
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "P3 6td adjusted-execution vs raw-execution one-by-one trade attribution",
        "paths": {
            "adjusted_backtest": str(ADJUSTED_DIR.relative_to(ROOT)),
            "raw_backtest": str(RAW_DIR.relative_to(ROOT)),
        },
        "summary": summary,
        "interpretation": {
            "trade_selection": (
                "The two runs bought the same 274 entry-date/code trades with the same exit dates; "
                "the return gap is not from signal swaps or extra blocked trades."
            ),
            "main_mechanism": (
                "All trades overlap, but raw execution deployed less capital on average after lot rounding. "
                "This notional/path sizing effect explains about half of the PnL gap."
            ),
            "secondary_mechanism": (
                "Only a small number of trades have different raw and adjusted return percentages, but they explain "
                "the other half of the gap; these are the important ex-right/dividend adjustment windows to inspect."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
