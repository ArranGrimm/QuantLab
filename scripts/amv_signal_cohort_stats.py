from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
LIMIT_TOLERANCE = 0.001


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_signal_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), resolve_path(path.strip())
    path = resolve_path(value)
    return path.name, path


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def resolve_signal_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.name == "signal.meta.json" or path.name == "signal.parquet":
        return path.parent
    raise ValueError(f"Unsupported signal input: {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def price_limit_pct_expr() -> pl.Expr:
    is_20pct_board = (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sh.688")
    )
    return pl.when(is_20pct_board).then(0.20).otherwise(0.10)


def max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(nav)
    dd = (nav - running_max) / running_max
    return float(dd.min())


def rolling_sleeve_nav(daily_ret: np.ndarray, horizon: int) -> tuple[float, float, list[dict[str, Any]]]:
    n_complete_epochs = len(daily_ret) // horizon
    if n_complete_epochs == 0:
        return 0.0, 0.0, []

    sleeve_navs = np.empty((horizon, n_complete_epochs), dtype=np.float64)
    sleeve_rows: list[dict[str, Any]] = []
    for sleeve_idx in range(horizon):
        sleeve_rets = np.asarray(daily_ret[sleeve_idx::horizon][:n_complete_epochs], dtype=np.float64)
        sleeve_nav = np.cumprod(1.0 + sleeve_rets)
        sleeve_navs[sleeve_idx] = sleeve_nav
        sleeve_rows.append(
            {
                "sleeve_idx": sleeve_idx,
                "cycles": int(len(sleeve_rets)),
                "total_return_pct": float((sleeve_nav[-1] - 1.0) * 100.0),
                "max_drawdown_pct": float(-max_drawdown(sleeve_nav) * 100.0),
                "mean_cycle_ret_pct": float(np.mean(sleeve_rets) * 100.0),
                "median_cycle_ret_pct": float(np.median(sleeve_rets) * 100.0),
                "worst_cycle_ret_pct": float(np.min(sleeve_rets) * 100.0),
                "best_cycle_ret_pct": float(np.max(sleeve_rets) * 100.0),
                "positive_cycle_share_pct": float(np.mean(sleeve_rets > 0.0) * 100.0),
            }
        )

    nav = sleeve_navs.mean(axis=0)
    return float(nav[-1] - 1.0), max_drawdown(nav), sleeve_rows


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def distribution_stats(values: np.ndarray, prefix: str = "") -> dict[str, Any]:
    if values.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean_pct": 0.0,
            f"{prefix}median_pct": 0.0,
            f"{prefix}std_pct": 0.0,
            f"{prefix}min_pct": 0.0,
            f"{prefix}p05_pct": 0.0,
            f"{prefix}p10_pct": 0.0,
            f"{prefix}p25_pct": 0.0,
            f"{prefix}p75_pct": 0.0,
            f"{prefix}p90_pct": 0.0,
            f"{prefix}p95_pct": 0.0,
            f"{prefix}max_pct": 0.0,
            f"{prefix}positive_share_pct": 0.0,
        }

    pct_values = values * 100.0
    return {
        f"{prefix}count": int(values.size),
        f"{prefix}mean_pct": float(np.mean(pct_values)),
        f"{prefix}median_pct": float(np.median(pct_values)),
        f"{prefix}std_pct": float(np.std(pct_values)),
        f"{prefix}min_pct": float(np.min(pct_values)),
        f"{prefix}p05_pct": percentile(pct_values, 5),
        f"{prefix}p10_pct": percentile(pct_values, 10),
        f"{prefix}p25_pct": percentile(pct_values, 25),
        f"{prefix}p75_pct": percentile(pct_values, 75),
        f"{prefix}p90_pct": percentile(pct_values, 90),
        f"{prefix}p95_pct": percentile(pct_values, 95),
        f"{prefix}max_pct": float(np.max(pct_values)),
        f"{prefix}positive_share_pct": float(np.mean(values > 0.0) * 100.0),
    }


def build_candidate_frame(signal_dir: Path, start_date: date, horizon: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    signal_path = signal_dir / "signal.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"signal.parquet not found: {signal_path}")

    frame = pl.read_parquet(signal_path).select(
        "date",
        "code",
        "open_adj",
        "close_adj",
        "pre_close_adj",
        "rank",
        "score",
        "is_signal",
        "is_bull_regime",
    )
    all_dates = frame.select("date").unique().sort("date").with_row_index("idx")
    entry_dates = all_dates.select(pl.col("date"), pl.col("idx").alias("entry_idx"))
    exit_dates = all_dates.with_columns((pl.col("idx") - horizon).alias("entry_idx")).select(
        pl.col("date").alias("exit_date"),
        "entry_idx",
    )
    exit_map = entry_dates.join(exit_dates, on="entry_idx", how="inner").select("date", "exit_date")
    exit_prices = frame.select(
        pl.col("date").alias("exit_date"),
        "code",
        pl.col("close_adj").alias("exit_close_adj"),
    )
    candidates = (
        frame.filter((pl.col("date") >= start_date) & pl.col("is_signal") & pl.col("is_bull_regime"))
        .join(exit_map, on="date", how="inner")
        .join(exit_prices, on=["exit_date", "code"], how="left")
        .with_columns(
            [
                (
                    (pl.col("open_adj") / pl.col("pre_close_adj") - 1.0)
                    >= (price_limit_pct_expr() - LIMIT_TOLERANCE)
                ).alias("open_limit_up"),
                (pl.col("exit_close_adj") / pl.col("open_adj") - 1.0).alias("ret"),
            ]
        )
        .drop_nulls("ret")
        .sort(["date", "rank", "code"])
    )
    dense_dates = all_dates.filter(pl.col("date") >= start_date).select("date").sort("date")
    return candidates, dense_dates


def select_scenario(candidates: pl.DataFrame, scenario: str, top_n: int, scan_rank: int) -> pl.DataFrame:
    if scenario == "strict_top3":
        return (
            candidates.filter((pl.col("rank") <= top_n) & ~pl.col("open_limit_up"))
            .sort(["date", "rank", "code"])
        )
    if scenario == "refill_top10":
        return (
            candidates.filter((pl.col("rank") <= scan_rank) & ~pl.col("open_limit_up"))
            .sort(["date", "rank", "code"])
            .group_by("date", maintain_order=True)
            .head(top_n)
            .sort(["date", "rank", "code"])
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def summarize_scenario(
    selected: pl.DataFrame,
    dense_dates: pl.DataFrame,
    *,
    horizon: int,
    top_n: int,
    round_trip_cost: float,
) -> dict[str, Any]:
    daily = (
        selected.group_by("date")
        .agg(
            [
                pl.col("ret").mean().alias("daily_ret"),
                (pl.col("ret") - round_trip_cost).mean().alias("daily_ret_cost_adjusted"),
                pl.len().alias("n_picks"),
                pl.col("rank").max().alias("max_rank_used"),
                pl.col("ret").min().alias("worst_pick_ret"),
                pl.col("ret").max().alias("best_pick_ret"),
            ]
        )
        .sort("date")
    )
    dense_daily = (
        dense_dates.join(daily.select("date", "daily_ret", "daily_ret_cost_adjusted"), on="date", how="left")
        .with_columns(
            [
                pl.col("daily_ret").fill_null(0.0),
                pl.col("daily_ret_cost_adjusted").fill_null(0.0),
            ]
        )
        .sort("date")
    )

    event_returns = daily["daily_ret"].to_numpy()
    event_returns_cost = daily["daily_ret_cost_adjusted"].to_numpy()
    dense_returns = dense_daily["daily_ret"].to_numpy()

    event_nav, event_max_dd, sleeve_rows = rolling_sleeve_nav(event_returns, horizon)
    event_nav_cost, event_max_dd_cost, sleeve_rows_cost = rolling_sleeve_nav(event_returns_cost, horizon)
    dense_nav, dense_max_dd, _ = rolling_sleeve_nav(dense_returns, horizon)

    sleeve_total_returns = np.asarray([row["total_return_pct"] for row in sleeve_rows], dtype=float)
    sleeve_max_dds = np.asarray([row["max_drawdown_pct"] for row in sleeve_rows], dtype=float)
    sleeve_total_returns_cost = np.asarray(
        [row["total_return_pct"] for row in sleeve_rows_cost],
        dtype=float,
    )

    yearly: dict[str, Any] = {}
    for year in daily.with_columns(pl.col("date").dt.year().alias("year"))["year"].unique().sort().to_list():
        year_daily = daily.with_columns(pl.col("date").dt.year().alias("year")).filter(pl.col("year") == year)
        year_returns = year_daily["daily_ret"].to_numpy()
        if year_returns.size == 0:
            continue
        year_nav, year_dd, _ = rolling_sleeve_nav(year_returns, horizon)
        yearly[str(int(year))] = {
            "signal_days": int(year_daily.height),
            "event_nav_pct": float(year_nav * 100.0),
            "event_max_drawdown_pct": float(-year_dd * 100.0),
            **distribution_stats(year_returns, prefix="daily_ret_"),
        }

    n_full_days = int((daily["n_picks"] == top_n).sum()) if daily.height else 0
    return {
        "signal_days": int(daily.height),
        "dense_trading_days": int(dense_daily.height),
        "selected_rows": int(selected.height),
        "full_pick_days": n_full_days,
        "full_pick_day_share_pct": float(n_full_days / daily.height * 100.0) if daily.height else 0.0,
        "avg_picks_per_day": float(daily["n_picks"].mean()) if daily.height else 0.0,
        "rank_q95": float(selected["rank"].quantile(0.95)) if selected.height else None,
        "daily_distribution": distribution_stats(event_returns, prefix=""),
        "pick_distribution": distribution_stats(selected["ret"].to_numpy(), prefix="pick_")
        if selected.height
        else {},
        "event_time_cohort": {
            "nav_pct": float(event_nav * 100.0),
            "max_drawdown_pct": float(-event_max_dd * 100.0),
            "sleeve_worst_nav_pct": float(np.min(sleeve_total_returns)) if sleeve_total_returns.size else 0.0,
            "sleeve_median_nav_pct": float(np.median(sleeve_total_returns)) if sleeve_total_returns.size else 0.0,
            "sleeve_best_nav_pct": float(np.max(sleeve_total_returns)) if sleeve_total_returns.size else 0.0,
            "sleeve_worst_max_drawdown_pct": float(np.max(sleeve_max_dds)) if sleeve_max_dds.size else 0.0,
            "sleeves": sleeve_rows,
        },
        "event_time_cohort_cost_adjusted": {
            "round_trip_cost_pct": round_trip_cost * 100.0,
            "nav_pct": float(event_nav_cost * 100.0),
            "max_drawdown_pct": float(-event_max_dd_cost * 100.0),
            "sleeve_worst_nav_pct": float(np.min(sleeve_total_returns_cost))
            if sleeve_total_returns_cost.size
            else 0.0,
            "sleeve_median_nav_pct": float(np.median(sleeve_total_returns_cost))
            if sleeve_total_returns_cost.size
            else 0.0,
            "sleeve_best_nav_pct": float(np.max(sleeve_total_returns_cost))
            if sleeve_total_returns_cost.size
            else 0.0,
        },
        "dense_calendar_zero_return_cohort": {
            "nav_pct": float(dense_nav * 100.0),
            "max_drawdown_pct": float(-dense_max_dd * 100.0),
        },
        "yearly": yearly,
    }


def summarize_signal(
    label: str,
    signal_dir: Path,
    *,
    start_date: date,
    horizon: int,
    top_n: int,
    scan_rank: int,
    round_trip_cost: float,
) -> dict[str, Any]:
    meta_path = signal_dir / "signal.meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    candidates, dense_dates = build_candidate_frame(signal_dir, start_date, horizon)
    scenarios = {}
    for scenario in ["strict_top3", "refill_top10"]:
        selected = select_scenario(candidates, scenario, top_n=top_n, scan_rank=scan_rank)
        scenarios[scenario] = summarize_scenario(
            selected,
            dense_dates,
            horizon=horizon,
            top_n=top_n,
            round_trip_cost=round_trip_cost,
        )

    return {
        "label": label,
        "signal_dir": str(signal_dir.relative_to(ROOT)),
        "feature_mode": meta.get("feature_mode"),
        "signal_id": meta.get("signal_id"),
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV signal rolling cohort statistics")
    parser.add_argument("--signal", action="append", required=True, help="label=signal_dir")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/amv_signal_cohort_stats.json")
    parser.add_argument("--start-date", type=parse_date, default=date(2021, 1, 1))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--scan-rank", type=int, default=10)
    parser.add_argument("--round-trip-cost", type=float, default=0.0035)
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.scan_rank < args.top_n:
        raise ValueError("--scan-rank must be >= --top-n")

    records = []
    for signal_arg in args.signal:
        label, path = parse_signal_arg(signal_arg)
        records.append(
            summarize_signal(
                label,
                resolve_signal_dir(path),
                start_date=args.start_date,
                horizon=args.horizon,
                top_n=args.top_n,
                scan_rank=args.scan_rank,
                round_trip_cost=args.round_trip_cost,
            )
        )

    output = resolve_path(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Rolling cohort signal diagnostics beyond mean return.",
        "config": {
            "start_date": args.start_date.isoformat(),
            "horizon": args.horizon,
            "top_n": args.top_n,
            "scan_rank": args.scan_rank,
            "round_trip_cost": args.round_trip_cost,
        },
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    for record in records:
        print(record["label"])
        for scenario, stats in record["scenarios"].items():
            cohort = stats["event_time_cohort"]
            daily = stats["daily_distribution"]
            print(
                f"  {scenario}: nav={cohort['nav_pct']:+.2f}% "
                f"median={daily['median_pct']:+.3f}% "
                f"p10={daily['p10_pct']:+.3f}% "
                f"win={daily['positive_share_pct']:.1f}% "
                f"sleeve_worst={cohort['sleeve_worst_nav_pct']:+.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
