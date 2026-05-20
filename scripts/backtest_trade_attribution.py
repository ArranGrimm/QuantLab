from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BacktestArtifact:
    label: str
    path: Path
    report: dict[str, Any]
    trades: list[dict[str, Any]]
    equity: list[dict[str, Any]]


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _load_artifact(path: Path, label: str) -> BacktestArtifact:
    report_path = path / "report.json"
    trades_path = path / "trades.csv"
    equity_path = path / "daily_equity.csv"
    missing = [p.name for p in (report_path, trades_path, equity_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing required files: {', '.join(missing)}")

    return BacktestArtifact(
        label=label,
        path=path,
        report=_read_json(report_path),
        trades=_read_csv(trades_path),
        equity=_read_csv(equity_path),
    )


def _metrics(artifact: BacktestArtifact) -> dict[str, Any]:
    metrics = artifact.report.get("metrics", {})
    extra = artifact.report.get("extra", {})
    return {
        "total_return_pct": metrics.get("total_return_pct"),
        "gross_return_pct": metrics.get("gross_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "total_trades": metrics.get("total_trades"),
        "trading_days": metrics.get("trading_days"),
        "avg_trades_per_day": metrics.get("avg_trades_per_day"),
        "final_portfolio": metrics.get("final_portfolio"),
        "total_pnl": metrics.get("total_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "total_costs": metrics.get("total_costs"),
        "total_commission": metrics.get("total_commission"),
        "total_slippage": metrics.get("total_slippage"),
        "total_stamp_duty": metrics.get("total_stamp_duty"),
        "limit_up_blocked": extra.get("limit_up_blocked"),
        "limit_up_days": extra.get("limit_up_days"),
        "open_gap_blocked": extra.get("open_gap_blocked"),
        "bull_regime_blocked_signals": extra.get("bull_regime_blocked_signals"),
    }


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    out: dict[str, Any] = {}
    for key in keys:
        l_val = left.get(key)
        r_val = right.get(key)
        if isinstance(l_val, (int, float)) and isinstance(r_val, (int, float)):
            out[key] = r_val - l_val
    return out


def _initial_capital(artifact: BacktestArtifact) -> float:
    config = artifact.report.get("backtest_config", {})
    if "initial_capital" in config:
        return _float(config["initial_capital"])
    if artifact.equity:
        return _float(artifact.equity[0]["total_value"])
    return 0.0


def _period_key(value: str, period: str) -> str:
    if period == "year":
        return value[:4]
    if period == "month":
        return value[:7]
    raise ValueError(f"unsupported period: {period}")


def _equity_period_returns(artifact: BacktestArtifact, period: str) -> dict[str, float]:
    if not artifact.equity:
        return {}
    rows = sorted(artifact.equity, key=lambda r: r["date"])
    period_last: dict[str, float] = {}
    for row in rows:
        period_last[_period_key(row["date"], period)] = _float(row["total_value"])

    returns: dict[str, float] = {}
    previous_value = _initial_capital(artifact)
    for key in sorted(period_last):
        end_value = period_last[key]
        if previous_value:
            returns[key] = (end_value / previous_value - 1.0) * 100.0
        previous_value = end_value
    return returns


def _pair_period_returns(
    left: BacktestArtifact,
    right: BacktestArtifact,
    period: str,
) -> list[dict[str, Any]]:
    left_ret = _equity_period_returns(left, period)
    right_ret = _equity_period_returns(right, period)
    rows = []
    for key in sorted(set(left_ret) | set(right_ret)):
        l_val = left_ret.get(key)
        r_val = right_ret.get(key)
        rows.append(
            {
                period: key,
                "left_return_pct": l_val,
                "right_return_pct": r_val,
                "delta_pct": None if l_val is None or r_val is None else r_val - l_val,
            }
        )
    return rows


def _trade_period_pnl(artifact: BacktestArtifact, period: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for trade in artifact.trades:
        key = _period_key(trade["exit_date"], period)
        out[key] += _float(trade["pnl"])
    return dict(out)


def _pair_trade_period_pnl(
    left: BacktestArtifact,
    right: BacktestArtifact,
    period: str,
) -> list[dict[str, Any]]:
    left_pnl = _trade_period_pnl(left, period)
    right_pnl = _trade_period_pnl(right, period)
    initial = _initial_capital(left) or _initial_capital(right)
    rows = []
    for key in sorted(set(left_pnl) | set(right_pnl)):
        l_val = left_pnl.get(key, 0.0)
        r_val = right_pnl.get(key, 0.0)
        delta = r_val - l_val
        rows.append(
            {
                period: key,
                "left_pnl": l_val,
                "right_pnl": r_val,
                "delta_pnl": delta,
                "delta_pct_initial": None if not initial else delta / initial * 100.0,
            }
        )
    return rows


def _trade_key(trade: dict[str, Any]) -> tuple[str, str]:
    return trade["entry_date"], trade["code"]


def _aggregate_by_trade_key(trades: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for trade in trades:
        key = _trade_key(trade)
        if key not in grouped:
            grouped[key] = dict(trade)
            grouped[key]["pnl"] = _float(trade.get("pnl"))
            grouped[key]["pnl_pct"] = _float(trade.get("pnl_pct")) * 100.0
            grouped[key]["count"] = 1
        else:
            grouped[key]["pnl"] += _float(trade.get("pnl"))
            grouped[key]["count"] += 1
    return grouped


def _compact_trade(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": trade.get("code"),
        "entry_date": trade.get("entry_date"),
        "exit_date": trade.get("exit_date"),
        "pnl": _float(trade.get("pnl")),
        "pnl_pct": _float(trade.get("pnl_pct")) * (100.0 if abs(_float(trade.get("pnl_pct"))) <= 2.0 else 1.0),
        "hold_trading_days": int(_float(trade.get("hold_trading_days"))) if trade.get("hold_trading_days") else None,
        "exit_reason": trade.get("exit_reason"),
    }


def _top_trades(
    trades: list[dict[str, Any]],
    *,
    limit: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    return [
        _compact_trade(t)
        for t in sorted(trades, key=lambda r: _float(r.get("pnl")), reverse=reverse)[:limit]
    ]


def _trade_overlap(left: BacktestArtifact, right: BacktestArtifact, top_n: int) -> dict[str, Any]:
    left_by_key = _aggregate_by_trade_key(left.trades)
    right_by_key = _aggregate_by_trade_key(right.trades)
    common_keys = set(left_by_key) & set(right_by_key)
    left_only_keys = set(left_by_key) - common_keys
    right_only_keys = set(right_by_key) - common_keys

    left_only = [left_by_key[k] for k in left_only_keys]
    right_only = [right_by_key[k] for k in right_only_keys]
    left_codes = {t["code"] for t in left.trades}
    right_codes = {t["code"] for t in right.trades}

    common = []
    for key in common_keys:
        l_trade = left_by_key[key]
        r_trade = right_by_key[key]
        common.append(
            {
                "code": key[1],
                "entry_date": key[0],
                "left_pnl": _float(l_trade.get("pnl")),
                "right_pnl": _float(r_trade.get("pnl")),
                "delta_pnl": _float(r_trade.get("pnl")) - _float(l_trade.get("pnl")),
                "left_exit_date": l_trade.get("exit_date"),
                "right_exit_date": r_trade.get("exit_date"),
            }
        )

    left_total = len(left_by_key)
    right_total = len(right_by_key)
    common_left_pnl = sum(_float(left_by_key[k].get("pnl")) for k in common_keys)
    common_right_pnl = sum(_float(right_by_key[k].get("pnl")) for k in common_keys)
    left_only_pnl = sum(_float(t.get("pnl")) for t in left_only)
    right_only_pnl = sum(_float(t.get("pnl")) for t in right_only)

    return {
        "exact_overlap_count": len(common_keys),
        "exact_overlap_pct_of_left": None if not left_total else len(common_keys) / left_total * 100.0,
        "exact_overlap_pct_of_right": None if not right_total else len(common_keys) / right_total * 100.0,
        "left_unique_count": len(left_only_keys),
        "right_unique_count": len(right_only_keys),
        "left_unique_pnl": left_only_pnl,
        "right_unique_pnl": right_only_pnl,
        "unique_delta_pnl": right_only_pnl - left_only_pnl,
        "common_left_pnl": common_left_pnl,
        "common_right_pnl": common_right_pnl,
        "common_delta_pnl": common_right_pnl - common_left_pnl,
        "code_overlap_count": len(left_codes & right_codes),
        "left_code_count": len(left_codes),
        "right_code_count": len(right_codes),
        "code_overlap_pct_of_left": None if not left_codes else len(left_codes & right_codes) / len(left_codes) * 100.0,
        "code_overlap_pct_of_right": None if not right_codes else len(left_codes & right_codes) / len(right_codes) * 100.0,
        "top_left_unique_winners": _top_trades(left_only, limit=top_n, reverse=True),
        "top_left_unique_losers": _top_trades(left_only, limit=top_n, reverse=False),
        "top_right_unique_winners": _top_trades(right_only, limit=top_n, reverse=True),
        "top_right_unique_losers": _top_trades(right_only, limit=top_n, reverse=False),
        "top_common_delta_positive": sorted(common, key=lambda r: r["delta_pnl"], reverse=True)[:top_n],
        "top_common_delta_negative": sorted(common, key=lambda r: r["delta_pnl"])[:top_n],
    }


def _daily_returns(artifact: BacktestArtifact) -> dict[str, float]:
    rows = sorted(artifact.equity, key=lambda r: r["date"])
    out: dict[str, float] = {}
    prev: float | None = None
    for row in rows:
        value = _float(row["total_value"])
        if prev and prev > 0:
            out[row["date"]] = value / prev - 1.0
        prev = value
    return out


def _pearson(left: list[float], right: list[float]) -> float | None:
    n = len(left)
    if n < 2:
        return None
    mean_l = sum(left) / n
    mean_r = sum(right) / n
    cov = sum((l - mean_l) * (r - mean_r) for l, r in zip(left, right))
    var_l = sum((l - mean_l) ** 2 for l in left)
    var_r = sum((r - mean_r) ** 2 for r in right)
    denom = math.sqrt(var_l * var_r)
    return None if denom == 0 else cov / denom


def _daily_return_correlation(left: BacktestArtifact, right: BacktestArtifact) -> dict[str, Any]:
    left_ret = _daily_returns(left)
    right_ret = _daily_returns(right)
    common_dates = sorted(set(left_ret) & set(right_ret))
    left_values = [left_ret[d] for d in common_dates]
    right_values = [right_ret[d] for d in common_dates]
    return {
        "common_days": len(common_dates),
        "pearson": _pearson(left_values, right_values),
    }


def _top_periods(rows: list[dict[str, Any]], key: str, top_n: int, reverse: bool) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r.get(key) if r.get(key) is not None else float("-inf"), reverse=reverse)[:top_n]


def _build_attribution(
    left: BacktestArtifact,
    right: BacktestArtifact,
    *,
    top_n: int,
) -> dict[str, Any]:
    left_metrics = _metrics(left)
    right_metrics = _metrics(right)
    yearly = _pair_period_returns(left, right, "year")
    monthly = _pair_period_returns(left, right, "month")
    trade_year = _pair_trade_period_pnl(left, right, "year")
    trade_month = _pair_trade_period_pnl(left, right, "month")

    return {
        "labels": {
            "left": left.label,
            "right": right.label,
        },
        "paths": {
            "left": str(left.path),
            "right": str(right.path),
        },
        "summary": {
            "left": left_metrics,
            "right": right_metrics,
            "delta_right_minus_left": _metric_delta(left_metrics, right_metrics),
        },
        "cost_drag": {
            "left_total_costs": left_metrics.get("total_costs"),
            "right_total_costs": right_metrics.get("total_costs"),
            "delta_total_costs": (
                None
                if left_metrics.get("total_costs") is None or right_metrics.get("total_costs") is None
                else right_metrics["total_costs"] - left_metrics["total_costs"]
            ),
            "left_cost_to_gross_pnl_pct": (
                None
                if not left_metrics.get("gross_pnl")
                else left_metrics.get("total_costs", 0.0) / left_metrics["gross_pnl"] * 100.0
            ),
            "right_cost_to_gross_pnl_pct": (
                None
                if not right_metrics.get("gross_pnl")
                else right_metrics.get("total_costs", 0.0) / right_metrics["gross_pnl"] * 100.0
            ),
        },
        "yearly_returns": yearly,
        "monthly_returns": monthly,
        "top_monthly_delta": _top_periods(monthly, "delta_pct", top_n, True),
        "bottom_monthly_delta": _top_periods(monthly, "delta_pct", top_n, False),
        "trade_year_pnl": trade_year,
        "trade_month_pnl": trade_month,
        "top_trade_month_delta": _top_periods(trade_month, "delta_pnl", top_n, True),
        "bottom_trade_month_delta": _top_periods(trade_month, "delta_pnl", top_n, False),
        "overlap": _trade_overlap(left, right, top_n),
        "daily_return_correlation": _daily_return_correlation(left, right),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two bt-amv-topn-style backtest artifacts and produce trade attribution JSON.",
    )
    parser.add_argument("--left-backtest", required=True, type=Path, help="Left/reference backtest artifact directory.")
    parser.add_argument("--right-backtest", required=True, type=Path, help="Right/candidate backtest artifact directory.")
    parser.add_argument("--left-label", default="left", help="Human-readable label for left backtest.")
    parser.add_argument("--right-label", default="right", help="Human-readable label for right backtest.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--top-n", default=12, type=int, help="Number of top/bottom rows to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = _load_artifact(args.left_backtest, args.left_label)
    right = _load_artifact(args.right_backtest, args.right_label)
    attribution = _build_attribution(left, right, top_n=args.top_n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8")

    delta = attribution["summary"]["delta_right_minus_left"]
    print(f"Wrote {args.out}")
    print(
        "return_delta={:.2f}pp maxdd_delta={:.2f}pp exact_overlap={}".format(
            delta.get("total_return_pct", 0.0),
            delta.get("max_drawdown_pct", 0.0),
            attribution["overlap"]["exact_overlap_count"],
        )
    )


if __name__ == "__main__":
    main()
