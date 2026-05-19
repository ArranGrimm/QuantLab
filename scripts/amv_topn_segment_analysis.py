from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(path_str: str | None, base: Path | None = None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ((base or ROOT) / path).resolve()
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_report_dir(path_str: str) -> Path:
    path = resolve_path(path_str)
    assert path is not None
    if path.is_file():
        path = path.parent
    if not (path / "report.json").exists():
        raise FileNotFoundError(f"report.json 不存在: {path}")
    if not (path / "trades.csv").exists():
        raise FileNotFoundError(f"trades.csv 不存在: {path}")
    return path


def resolve_signal_path(report_dir: Path, report: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        path = resolve_path(explicit)
        assert path is not None
        return path
    signal_path = report.get("canonical_signal_file") or report.get("input_signal_file")
    if not signal_path:
        raise ValueError("report.json 缺少 signal path")
    path = resolve_path(signal_path, report_dir)
    assert path is not None
    return path


def as_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def build_signal_index(signal_path: Path) -> tuple[dict[tuple[str, date], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = (
        pl.read_parquet(signal_path)
        .select(
            [
                "date",
                "code",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "pre_close_adj",
                "is_bull_regime",
                "amv_mechanical_regime",
                "market_cap_100m",
                "amount_ma20",
                "score",
                "rank",
                "is_signal",
            ]
        )
        .sort(["code", "date"])
        .to_dicts()
    )
    by_key: dict[tuple[str, date], dict[str, Any]] = {}
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_date = as_date(row["date"])
        row["date"] = row_date
        code = row["code"]
        by_key[(code, row_date)] = row
        by_code.setdefault(code, []).append(row)
    return by_key, by_code


def enrich_trades(trades: pd.DataFrame, signal_path: Path) -> pd.DataFrame:
    by_key, by_code = build_signal_index(signal_path)
    enriched_rows: list[dict[str, Any]] = []

    for _, trade in trades.iterrows():
        entry_date = as_date(trade["entry_date"])
        exit_date = as_date(trade["exit_date"])
        code = str(trade["code"])
        entry = by_key.get((code, entry_date), {})
        path_rows = [
            row
            for row in by_code.get(code, [])
            if entry_date <= row["date"] <= exit_date
        ]

        entry_price = float(trade["entry_price"])
        max_high = max((float(row["high_adj"]) for row in path_rows), default=float(trade["exit_price"]))
        min_low = min((float(row["low_adj"]) for row in path_rows), default=float(trade["exit_price"]))
        min_close = min((float(row["close_adj"]) for row in path_rows), default=float(trade["exit_price"]))
        high_dates = [row["date"] for row in path_rows if float(row["high_adj"]) == max_high]
        low_dates = [row["date"] for row in path_rows if float(row["low_adj"]) == min_low]
        bull_days = sum(1 for row in path_rows if bool(row.get("is_bull_regime")))

        pre_close = entry.get("pre_close_adj")
        close_adj = entry.get("close_adj")
        open_gap_pct = (
            (entry_price / float(pre_close) - 1.0) * 100.0
            if pre_close is not None and float(pre_close) > 0
            else None
        )
        entry_intraday_pct = (
            (float(close_adj) / entry_price - 1.0) * 100.0
            if close_adj is not None and entry_price > 0
            else None
        )
        mfe_pct = (max_high / entry_price - 1.0) * 100.0 if entry_price > 0 else None
        mae_pct = (min_low / entry_price - 1.0) * 100.0 if entry_price > 0 else None
        min_close_pct = (min_close / entry_price - 1.0) * 100.0 if entry_price > 0 else None
        pnl_pct = float(trade["pnl_pct"]) * 100.0

        enriched = trade.to_dict()
        enriched.update(
            {
                "entry_year": entry_date.year,
                "exit_year": exit_date.year,
                "entry_month": entry_date.strftime("%Y-%m"),
                "exit_month": exit_date.strftime("%Y-%m"),
                "score": entry.get("score"),
                "rank": entry.get("rank"),
                "entry_open_gap_pct": open_gap_pct,
                "entry_intraday_pct": entry_intraday_pct,
                "entry_amv_regime": entry.get("amv_mechanical_regime"),
                "entry_is_bull_regime": entry.get("is_bull_regime"),
                "market_cap_100m": entry.get("market_cap_100m"),
                "amount_ma20": entry.get("amount_ma20"),
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "min_close_pct": min_close_pct,
                "capture_of_mfe_pct": pnl_pct / mfe_pct * 100.0 if mfe_pct and mfe_pct > 0 else None,
                "bull_day_ratio": bull_days / len(path_rows) if path_rows else None,
                "days_to_high": (high_dates[0] - entry_date).days if high_dates else None,
                "days_to_low": (low_dates[0] - entry_date).days if low_dates else None,
            }
        )
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def summarize_group(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "pnl": 0.0,
            "win_rate_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "median_pnl_pct": 0.0,
        }
    pnl_pct = df["pnl_pct"] * 100.0
    return {
        "trades": int(len(df)),
        "pnl": float(df["pnl"].sum()),
        "win_rate_pct": float((df["pnl"] > 0).mean() * 100.0),
        "avg_pnl_pct": float(pnl_pct.mean()),
        "median_pnl_pct": float(pnl_pct.median()),
        "avg_open_gap_pct": float(df["entry_open_gap_pct"].mean()),
        "median_open_gap_pct": float(df["entry_open_gap_pct"].median()),
        "avg_entry_intraday_pct": float(df["entry_intraday_pct"].mean()),
        "avg_mfe_pct": float(df["mfe_pct"].mean()),
        "avg_mae_pct": float(df["mae_pct"].mean()),
        "median_market_cap_100m": float(df["market_cap_100m"].median()),
        "median_amount_ma20": float(df["amount_ma20"].median()),
        "avg_score": float(df["score"].mean()),
        "rank_1_ratio_pct": float((df["rank"] == 1).mean() * 100.0),
    }


def summarize_months(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    monthly = (
        df.groupby("entry_month")
        .agg(
            trades=("code", "size"),
            pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100.0),
            avg_gap_pct=("entry_open_gap_pct", "mean"),
        )
        .reset_index()
    )
    monthly["avg_pnl_pct"] *= 100.0
    return monthly.sort_values("pnl", ascending=False)


def top_counter(values: pd.Series, n: int = 5) -> list[dict[str, Any]]:
    counter = Counter(str(v) for v in values.dropna())
    return [{"value": value, "count": count} for value, count in counter.most_common(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV TopN segment attribution analysis")
    parser.add_argument("--baseline-dir", required=True, help="6td + no stop report dir / report.json")
    parser.add_argument("--signal", help="signal.parquet；默认从 report.json 解析")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    baseline_dir = resolve_report_dir(args.baseline_dir)
    report = read_json(baseline_dir / "report.json")
    signal_path = resolve_signal_path(baseline_dir, report, args.signal)

    output_dir = args.output_dir or baseline_dir.parent / f"segment_analysis_{timestamp_token()}"
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(baseline_dir / "trades.csv", parse_dates=["entry_date", "exit_date"])
    enriched = enrich_trades(trades, signal_path)
    enriched.to_csv(output_dir / "enriched_trades.csv", index=False)

    top_2024 = enriched.loc[enriched["exit_year"] == 2024].nlargest(10, "pnl").copy()
    trades_2026 = enriched.loc[enriched["exit_year"] == 2026].copy()
    loss_2026 = trades_2026.loc[trades_2026["pnl"] < 0].copy()
    winners_2024 = enriched.loc[(enriched["exit_year"] == 2024) & (enriched["pnl"] > 0)].copy()
    non_2026 = enriched.loc[enriched["exit_year"] != 2026].copy()

    top_2024.to_csv(output_dir / "top_2024_winners.csv", index=False)
    trades_2026.to_csv(output_dir / "trades_2026.csv", index=False)
    loss_2026.nsmallest(20, "pnl").to_csv(output_dir / "loss_2026_worst.csv", index=False)
    summarize_months(enriched).to_csv(output_dir / "monthly_summary.csv", index=False)

    segment_rows = []
    for name, df in [
        ("all", enriched),
        ("non_2026", non_2026),
        ("2024_winners", winners_2024),
        ("2024_top10_winners", top_2024),
        ("2026_all", trades_2026),
        ("2026_losses", loss_2026),
    ]:
        row = {"segment": name, **summarize_group(df)}
        segment_rows.append(row)
    segment_summary = pd.DataFrame(segment_rows)
    segment_summary.to_csv(output_dir / "segment_summary.csv", index=False)

    summary = {
        "analysis_id": output_dir.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_dir": str(baseline_dir),
        "signal_path": str(signal_path),
        "files": {
            "enriched_trades": "enriched_trades.csv",
            "segment_summary": "segment_summary.csv",
            "monthly_summary": "monthly_summary.csv",
            "top_2024_winners": "top_2024_winners.csv",
            "trades_2026": "trades_2026.csv",
            "loss_2026_worst": "loss_2026_worst.csv",
        },
        "segments": {row["segment"]: {k: v for k, v in row.items() if k != "segment"} for row in segment_rows},
        "top_2024_entry_months": top_counter(top_2024["entry_month"]),
        "top_2026_loss_entry_months": top_counter(loss_2026["entry_month"]),
        "top_2024_codes": top_2024[
            ["code", "entry_date", "exit_date", "pnl", "pnl_pct", "entry_open_gap_pct", "mfe_pct", "mae_pct"]
        ]
        .assign(pnl_pct=lambda df: df["pnl_pct"] * 100.0)
        .to_dict(orient="records"),
        "worst_2026_codes": loss_2026.nsmallest(10, "pnl")[
            ["code", "entry_date", "exit_date", "pnl", "pnl_pct", "entry_open_gap_pct", "mfe_pct", "mae_pct"]
        ]
        .assign(pnl_pct=lambda df: df["pnl_pct"] * 100.0)
        .to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"segment analysis: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
