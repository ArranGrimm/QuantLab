from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_report_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()

    if path.is_file():
        path = path.parent
    if not (path / "report.json").exists():
        raise FileNotFoundError(f"report.json 不存在: {path}")
    if not (path / "trades.csv").exists():
        raise FileNotFoundError(f"trades.csv 不存在: {path}")
    if not (path / "daily_equity.csv").exists():
        raise FileNotFoundError(f"daily_equity.csv 不存在: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_drawdown(values: pd.Series) -> float:
    peaks = values.cummax()
    drawdowns = values / peaks - 1.0
    return float(drawdowns.min())


def build_annual(equity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in equity.groupby(equity["date"].dt.year):
        start_value = float(group["total_value"].iloc[0])
        end_value = float(group["total_value"].iloc[-1])
        rows.append(
            {
                "year": int(year),
                "start_date": group["date"].iloc[0].date().isoformat(),
                "end_date": group["date"].iloc[-1].date().isoformat(),
                "start_value": start_value,
                "end_value": end_value,
                "return_pct": (end_value / start_value - 1.0) * 100.0,
                "max_drawdown_pct": max_drawdown(group["total_value"]) * 100.0,
                "trading_days": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_trade_distribution(trades: pd.DataFrame) -> dict[str, Any]:
    pnl_pct = trades["pnl_pct"] * 100.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] <= 0, "pnl"]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_abs = float(losses.abs().mean()) if not losses.empty else 0.0
    return {
        "total_trades": int(len(trades)),
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100.0),
        "avg_pnl_pct": float(pnl_pct.mean()),
        "median_pnl_pct": float(pnl_pct.median()),
        "p10_pnl_pct": float(pnl_pct.quantile(0.10)),
        "p25_pnl_pct": float(pnl_pct.quantile(0.25)),
        "p75_pnl_pct": float(pnl_pct.quantile(0.75)),
        "p90_pnl_pct": float(pnl_pct.quantile(0.90)),
        "best_pnl_pct": float(pnl_pct.max()),
        "worst_pnl_pct": float(pnl_pct.min()),
        "avg_hold_trading_days": float(trades["hold_trading_days"].mean()),
        "median_hold_trading_days": float(trades["hold_trading_days"].median()),
        "total_pnl": float(trades["pnl"].sum()),
        "avg_win": avg_win,
        "avg_loss_abs": avg_loss_abs,
        "payoff_ratio": avg_win / avg_loss_abs if avg_loss_abs > 0 else None,
    }


def summarize_top_contribution(trades: pd.DataFrame) -> dict[str, Any]:
    total_pnl = float(trades["pnl"].sum())
    positive_pnl = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    loss_abs = float(trades.loc[trades["pnl"] < 0, "pnl"].abs().sum())
    top_profit_sum = float(trades.nlargest(10, "pnl")["pnl"].sum())
    top_loss_abs = float(trades.nsmallest(10, "pnl")["pnl"].abs().sum())
    return {
        "total_pnl": total_pnl,
        "positive_pnl": positive_pnl,
        "loss_abs": loss_abs,
        "top10_profit_sum": top_profit_sum,
        "top10_profit_share_of_positive_pnl_pct": top_profit_sum / positive_pnl * 100.0
        if positive_pnl > 0
        else None,
        "top10_profit_share_of_net_pnl_pct": top_profit_sum / total_pnl * 100.0
        if total_pnl != 0
        else None,
        "top10_loss_abs": top_loss_abs,
        "top10_loss_share_of_total_loss_pct": top_loss_abs / loss_abs * 100.0 if loss_abs > 0 else None,
    }


def write_trade_views(trades: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    top_profit = trades.nlargest(10, "pnl").copy()
    top_loss = trades.nsmallest(10, "pnl").copy()
    top_profit.to_csv(output_dir / "top_profit_trades.csv", index=False)
    top_loss.to_csv(output_dir / "top_loss_trades.csv", index=False)

    exit_reason = (
        trades.groupby("exit_reason")
        .agg(trades=("code", "size"), pnl=("pnl", "sum"), avg_pnl_pct=("pnl_pct", "mean"))
        .reset_index()
    )
    exit_reason["avg_pnl_pct"] *= 100.0
    exit_reason.to_csv(output_dir / "exit_reason_summary.csv", index=False)

    hold_dist = (
        trades.groupby("hold_trading_days")
        .agg(trades=("code", "size"), pnl=("pnl", "sum"), avg_pnl_pct=("pnl_pct", "mean"))
        .reset_index()
    )
    hold_dist["avg_pnl_pct"] *= 100.0
    hold_dist.to_csv(output_dir / "hold_distribution.csv", index=False)

    return {
        "top_profit_csv": "top_profit_trades.csv",
        "top_loss_csv": "top_loss_trades.csv",
        "exit_reason_csv": "exit_reason_summary.csv",
        "hold_distribution_csv": "hold_distribution.csv",
    }


def build_stop_loss_analysis(no_stop_trades: pd.DataFrame, stop_trades: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    joined = no_stop_trades.merge(
        stop_trades,
        on=["code", "entry_date"],
        how="inner",
        suffixes=("_no_stop", "_stop"),
    )
    stopped = joined.loc[joined["exit_reason_stop"] == "StopLoss"].copy()
    stopped["pnl_pct_delta"] = (stopped["pnl_pct_no_stop"] - stopped["pnl_pct_stop"]) * 100.0
    stopped["pnl_delta"] = stopped["pnl_no_stop"] - stopped["pnl_stop"]
    missed = stopped.loc[stopped["pnl_pct_delta"] > 0].sort_values("pnl_pct_delta", ascending=False)

    columns = [
        "code",
        "entry_date",
        "exit_date_stop",
        "exit_date_no_stop",
        "pnl_pct_stop",
        "pnl_pct_no_stop",
        "pnl_pct_delta",
        "pnl_stop",
        "pnl_no_stop",
        "pnl_delta",
    ]
    missed[columns].to_csv(output_dir / "stop_loss_missed_trades.csv", index=False)

    return {
        "matched_entry_trades": int(len(joined)),
        "stop_loss_matched_trades": int(len(stopped)),
        "missed_rebound_trades": int(len(missed)),
        "missed_rebound_ratio_pct": float(len(missed) / len(stopped) * 100.0) if len(stopped) else 0.0,
        "avg_missed_rebound_pct": float(missed["pnl_pct_delta"].mean()) if len(missed) else 0.0,
        "total_missed_pnl_delta": float(missed["pnl_delta"].sum()) if len(missed) else 0.0,
        "csv": "stop_loss_missed_trades.csv",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV TopN trade attribution analysis")
    parser.add_argument("--baseline-dir", required=True, help="6td + no stop report dir / report.json")
    parser.add_argument("--stop-dir", help="同持有期 5% stop report dir / report.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    baseline_dir = resolve_report_dir(args.baseline_dir)
    stop_dir = resolve_report_dir(args.stop_dir) if args.stop_dir else None
    output_dir = args.output_dir or baseline_dir.parent / f"analysis_{timestamp_token()}"
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(baseline_dir / "report.json")
    trades = pd.read_csv(baseline_dir / "trades.csv", parse_dates=["entry_date", "exit_date"])
    equity = pd.read_csv(baseline_dir / "daily_equity.csv", parse_dates=["date"])

    annual = build_annual(equity)
    annual.to_csv(output_dir / "annual_performance.csv", index=False)
    trade_distribution = build_trade_distribution(trades)
    trade_views = write_trade_views(trades, output_dir)
    top_contribution = summarize_top_contribution(trades)

    stop_loss_analysis = None
    if stop_dir is not None:
        stop_trades = pd.read_csv(stop_dir / "trades.csv", parse_dates=["entry_date", "exit_date"])
        stop_loss_analysis = build_stop_loss_analysis(trades, stop_trades, output_dir)

    summary = {
        "analysis_id": output_dir.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_dir": str(baseline_dir),
        "stop_dir": str(stop_dir) if stop_dir else None,
        "baseline_metrics": report["metrics"],
        "baseline_config": report["backtest_config"],
        "annual_csv": "annual_performance.csv",
        "trade_distribution": trade_distribution,
        "top_contribution": top_contribution,
        "trade_views": trade_views,
        "stop_loss_analysis": stop_loss_analysis,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"analysis: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
