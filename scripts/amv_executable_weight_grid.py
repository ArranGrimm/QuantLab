from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scripts.amv_bull_pool_export_signals import _git_commit, _rel_path
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_QMT_DB,
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    _finite_expr,
    _rolling_sleeve_nav,
    add_combo_scores,
    build_dataset,
)
from scripts.amv_yearly_weight_grid import build_rankers


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_executable_weight_grid"
LIMIT_TOLERANCE = 0.001


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _price_limit_pct_expr() -> pl.Expr:
    is_20pct_board = (
        pl.col("code").str.starts_with("sz.300")
        | pl.col("code").str.starts_with("sz.301")
        | pl.col("code").str.starts_with("sh.688")
        | pl.col("code").str.starts_with("sh.689")
        | pl.col("code").str.starts_with("300")
        | pl.col("code").str.starts_with("301")
        | pl.col("code").str.starts_with("688")
        | pl.col("code").str.starts_with("689")
    )
    return pl.when(is_20pct_board).then(0.20).otherwise(0.10)


def _ranker_factor_cols(ranker: dict[str, Any]) -> list[str]:
    if "factor_cols" in ranker:
        return [str(col) for col in ranker["factor_cols"]]
    if "components" in ranker:
        return [str(component["factor"]) for component in ranker["components"]]
    return [str(ranker["factor"])]


def _yearly_metric(row: dict[str, Any], year: int, key: str) -> float | None:
    year_row = row["yearly"].get(str(year))
    if year_row is None:
        return None
    value = year_row.get(key)
    return None if value is None else float(value)


def _tradeoff(nav_end: float, max_dd: float) -> float:
    return nav_end + max_dd


def _mean_bool(series: pl.Series) -> float:
    if series.is_empty():
        return 0.0
    return float(series.cast(pl.Float64).mean())


def _selected_days(selected: pl.DataFrame, predicate: pl.Expr) -> int:
    if selected.is_empty():
        return 0
    return selected.filter(predicate).select("date").n_unique()


def _build_dataset_for_horizon(args: argparse.Namespace, horizon: int) -> pl.DataFrame:
    dataset_args = copy.copy(args)
    dataset_args.horizon = horizon
    dataset_args.label_mode = "next_open_to_close"
    dataset_args.execution_lag_days = args.execution_lag_days
    dataset_args.price_limit_tolerance = args.price_limit_tolerance
    dataset_args.horizons = sorted({horizon, horizon + args.execution_lag_days})
    return build_dataset(dataset_args).with_columns(
        (
            (pl.col("close_adj") / pl.col("pre_close_adj") - 1.0)
            >= (_price_limit_pct_expr() - args.price_limit_tolerance)
        ).alias("is_close_limit_up")
    )


def _valid_ranker_pool(df_pool: pl.DataFrame, ranker: dict[str, Any]) -> pl.DataFrame:
    factor = ranker["factor"]
    finite_checks = [_finite_expr(col_name) for col_name in _ranker_factor_cols(ranker)]
    valid_expr = finite_checks[0]
    for finite_check in finite_checks[1:]:
        valid_expr = valid_expr & finite_check
    score_expr = pl.col(factor) if bool(ranker["descending"]) else -pl.col(factor)
    return (
        df_pool.filter(valid_expr & _finite_expr(factor))
        .with_columns(
            [
                score_expr.alias("score"),
                score_expr.rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt32)
                .alias("rank"),
            ]
        )
    )


def _select_scenarios(
    df_valid: pl.DataFrame,
    *,
    eligible_dates: list[object],
    top_n: int,
    max_scan_rank: int,
) -> dict[str, pl.DataFrame]:
    df_eligible = df_valid.filter(pl.col("date").is_in(eligible_dates))
    if max_scan_rank > 0:
        df_eligible = df_eligible.filter(pl.col("rank") <= max_scan_rank)

    original = (
        df_eligible.sort(["date", "score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "rank", "code"])
    )
    refill = (
        df_eligible.filter(~pl.col("is_close_limit_up"))
        .sort(["date", "score", "code"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "rank", "code"])
    )
    return {
        "original_top3": original,
        "skip_close_limit_refill_top3": refill,
    }


def _daily_baseline(
    df_pool: pl.DataFrame,
    dates: list[object],
    *,
    horizon: int,
) -> pl.DataFrame:
    return (
        df_pool.filter(pl.col("date").is_in(dates))
        .group_by("date")
        .agg(
            [
                pl.col(f"fwd_exec_ret_{horizon}d").mean().alias("baseline_exec_ret"),
                (pl.col(f"fwd_exec_mfe_{horizon}d") >= 0.15).mean().alias("baseline_exec_hit15"),
                pl.col(f"fwd_ret_{horizon}d").mean().alias("baseline_ctc_ret"),
                (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("baseline_ctc_hit15"),
            ]
        )
        .sort("date")
    )


def _daily_selected(
    dates_df: pl.DataFrame,
    selected: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    horizon: int,
) -> pl.DataFrame:
    if selected.is_empty():
        selected_daily = pl.DataFrame(
            schema={
                "date": pl.Date,
                "exec_daily_ret": pl.Float64,
                "exec_hit15": pl.Float64,
                "ctc_daily_ret": pl.Float64,
                "ctc_hit15": pl.Float64,
                "n_picks": pl.UInt32,
            }
        )
    else:
        selected_daily = (
            selected.group_by("date")
            .agg(
                [
                    pl.col(f"fwd_exec_ret_{horizon}d").mean().alias("exec_daily_ret"),
                    (pl.col(f"fwd_exec_mfe_{horizon}d") >= 0.15).mean().alias("exec_hit15"),
                    pl.col(f"fwd_ret_{horizon}d").mean().alias("ctc_daily_ret"),
                    (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("ctc_hit15"),
                    pl.len().cast(pl.UInt32).alias("n_picks"),
                ]
            )
            .sort("date")
        )

    return (
        dates_df.join(selected_daily, on="date", how="left")
        .join(baseline, on="date", how="left")
        .with_columns(
            [
                pl.col("exec_daily_ret").fill_null(0.0),
                pl.col("exec_hit15").fill_null(0.0),
                pl.col("ctc_daily_ret").fill_null(0.0),
                pl.col("ctc_hit15").fill_null(0.0),
                pl.col("n_picks").fill_null(0).cast(pl.UInt32),
            ]
        )
        .sort("date")
    )


def _summarize_daily(daily: pl.DataFrame, horizon: int) -> dict[str, Any]:
    exec_nav, exec_max_dd = _rolling_sleeve_nav(daily["exec_daily_ret"].to_numpy(), horizon)
    ctc_nav, ctc_max_dd = _rolling_sleeve_nav(daily["ctc_daily_ret"].to_numpy(), horizon)
    baseline_exec_nav, baseline_exec_max_dd = _rolling_sleeve_nav(
        daily["baseline_exec_ret"].to_numpy(), horizon
    )
    baseline_ctc_nav, baseline_ctc_max_dd = _rolling_sleeve_nav(
        daily["baseline_ctc_ret"].to_numpy(), horizon
    )
    exec_mean = float(daily["exec_daily_ret"].mean())
    baseline_exec_mean = float(daily["baseline_exec_ret"].mean())
    ctc_mean = float(daily["ctc_daily_ret"].mean())
    baseline_ctc_mean = float(daily["baseline_ctc_ret"].mean())
    return {
        "exec_mean_ret": exec_mean,
        "baseline_exec_mean_ret": baseline_exec_mean,
        "exec_edge_ret": exec_mean - baseline_exec_mean,
        "exec_hit15": float(daily["exec_hit15"].mean()),
        "baseline_exec_hit15": float(daily["baseline_exec_hit15"].mean()),
        "exec_nav_end": exec_nav,
        "baseline_exec_nav_end": baseline_exec_nav,
        "exec_edge_nav_end": exec_nav - baseline_exec_nav,
        "exec_max_dd": exec_max_dd,
        "baseline_exec_max_dd": baseline_exec_max_dd,
        "exec_tradeoff": _tradeoff(exec_nav, exec_max_dd),
        "ctc_mean_ret": ctc_mean,
        "baseline_ctc_mean_ret": baseline_ctc_mean,
        "ctc_edge_ret": ctc_mean - baseline_ctc_mean,
        "ctc_hit15": float(daily["ctc_hit15"].mean()),
        "baseline_ctc_hit15": float(daily["baseline_ctc_hit15"].mean()),
        "ctc_nav_end": ctc_nav,
        "baseline_ctc_nav_end": baseline_ctc_nav,
        "ctc_edge_nav_end": ctc_nav - baseline_ctc_nav,
        "ctc_max_dd": ctc_max_dd,
        "baseline_ctc_max_dd": baseline_ctc_max_dd,
        "ctc_tradeoff": _tradeoff(ctc_nav, ctc_max_dd),
    }


def _selection_diagnostics(
    selected: pl.DataFrame,
    *,
    dates_df: pl.DataFrame,
    top_n: int,
    high_open_pct: float,
) -> dict[str, Any]:
    if selected.is_empty():
        return {
            "selected_rows": 0,
            "full_days": 0,
            "avg_picks_per_day": 0.0,
            "close_limit_up_rows": 0,
            "close_limit_up_days": 0,
            "close_limit_up_row_share": 0.0,
            "close_limit_up_day_share": 0.0,
            "entry_limit_up_rows": 0,
            "entry_limit_up_days": 0,
            "entry_limit_up_row_share": 0.0,
            "entry_limit_up_day_share": 0.0,
            "high_open_rows": 0,
            "high_open_days": 0,
            "high_open_row_share": 0.0,
            "high_open_day_share": 0.0,
            "rank_mean": None,
            "rank_q95": None,
            "rank_max": None,
        }

    daily_counts = (
        dates_df.join(
            selected.group_by("date").agg(pl.len().cast(pl.UInt32).alias("n_picks")),
            on="date",
            how="left",
        )
        .with_columns(pl.col("n_picks").fill_null(0).cast(pl.UInt32))
        .sort("date")
    )
    close_limit = pl.col("is_close_limit_up")
    entry_limit = pl.col("fwd_exec_entry_limit_up")
    high_open = pl.col("fwd_exec_entry_gap") >= high_open_pct
    rank_q95 = selected["rank"].quantile(0.95)
    return {
        "selected_rows": selected.height,
        "full_days": int((daily_counts["n_picks"] == top_n).sum()),
        "avg_picks_per_day": float(daily_counts["n_picks"].mean()),
        "close_limit_up_rows": int(selected["is_close_limit_up"].sum()),
        "close_limit_up_days": _selected_days(selected, close_limit),
        "close_limit_up_row_share": _mean_bool(selected["is_close_limit_up"]),
        "close_limit_up_day_share": _selected_days(selected, close_limit) / dates_df.height,
        "entry_limit_up_rows": int(selected["fwd_exec_entry_limit_up"].sum()),
        "entry_limit_up_days": _selected_days(selected, entry_limit),
        "entry_limit_up_row_share": _mean_bool(selected["fwd_exec_entry_limit_up"]),
        "entry_limit_up_day_share": _selected_days(selected, entry_limit) / dates_df.height,
        "high_open_rows": int((selected["fwd_exec_entry_gap"] >= high_open_pct).sum()),
        "high_open_days": _selected_days(selected, high_open),
        "high_open_row_share": _mean_bool(selected["fwd_exec_entry_gap"] >= high_open_pct),
        "high_open_day_share": _selected_days(selected, high_open) / dates_df.height,
        "rank_mean": float(selected["rank"].mean()),
        "rank_q95": None if rank_q95 is None else float(rank_q95),
        "rank_max": int(selected["rank"].max()),
    }


def _evaluate_scenario(
    df_pool: pl.DataFrame,
    selected: pl.DataFrame,
    *,
    ranker: dict[str, Any],
    scenario: str,
    horizon: int,
    top_n: int,
    eligible_dates: list[object],
    high_open_pct: float,
) -> tuple[dict[str, Any], pl.DataFrame]:
    dates_df = pl.DataFrame({"date": eligible_dates})
    baseline = _daily_baseline(df_pool, eligible_dates, horizon=horizon)
    daily = _daily_selected(dates_df, selected, baseline, horizon=horizon)
    diagnostics = _selection_diagnostics(
        selected,
        dates_df=dates_df,
        top_n=top_n,
        high_open_pct=high_open_pct,
    )
    all_metrics = _summarize_daily(daily, horizon)

    yearly: dict[str, Any] = {}
    daily_by_year = daily.with_columns(pl.col("date").dt.year().alias("year"))
    for year in daily_by_year["year"].unique().sort().to_list():
        year_daily = daily_by_year.filter(pl.col("year") == year)
        if year_daily.height < horizon:
            continue
        yearly[str(int(year))] = {
            "year": int(year),
            "days": int(year_daily.height),
            **_summarize_daily(year_daily, horizon),
        }

    edge_values = [row["exec_edge_ret"] for row in yearly.values()]
    result = {
        "id": ranker["id"],
        "label": ranker["label"],
        "group": ranker["group"],
        "weights": ranker.get("weights"),
        "components": ranker.get("components"),
        "top_n": top_n,
        "horizon": horizon,
        "scenario": scenario,
        "eligible_days": len(eligible_dates),
        "all": all_metrics,
        "yearly": yearly,
        "diagnostics": diagnostics,
        "stable_positive_years": int(sum(edge > 0 for edge in edge_values)),
        "negative_edge_years": int(sum(edge <= 0 for edge in edge_values)),
        "exec_edge_mean_by_year": float(np.mean(edge_values)) if edge_values else None,
        "exec_edge_std_by_year": float(np.std(edge_values)) if edge_values else None,
        "exec_edge_sharpe_by_year": float(np.mean(edge_values) / np.std(edge_values))
        if len(edge_values) > 1 and np.std(edge_values) > 0
        else None,
    }
    return result, daily.with_columns(
        [
            pl.lit(ranker["id"]).alias("ranker_id"),
            pl.lit(scenario).alias("scenario"),
            pl.lit(horizon).alias("horizon"),
        ]
    )


def evaluate_ranker(
    df_pool: pl.DataFrame,
    ranker: dict[str, Any],
    *,
    horizon: int,
    top_n: int,
    max_scan_rank: int,
    high_open_pct: float,
) -> tuple[list[dict[str, Any]], list[pl.DataFrame]]:
    df_valid = _valid_ranker_pool(df_pool, ranker)
    daily_counts = (
        df_valid.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= top_n)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    if not eligible_dates:
        return (
            [
                {
                    **ranker,
                    "horizon": horizon,
                    "top_n": top_n,
                    "error": f"no dates with at least {top_n} finite factor values",
                }
            ],
            [],
        )

    selected_by_scenario = _select_scenarios(
        df_valid,
        eligible_dates=eligible_dates,
        top_n=top_n,
        max_scan_rank=max_scan_rank,
    )
    results: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    for scenario, selected in selected_by_scenario.items():
        result, daily = _evaluate_scenario(
            df_pool,
            selected,
            ranker=ranker,
            scenario=scenario,
            horizon=horizon,
            top_n=top_n,
            eligible_dates=eligible_dates,
            high_open_pct=high_open_pct,
        )
        results.append(result)
        daily_frames.append(daily)
    return results, daily_frames


def _compact_row(result: dict[str, Any]) -> dict[str, Any]:
    all_metrics = result["all"]
    diagnostics = result["diagnostics"]
    weights = result.get("weights")
    return {
        "id": result["id"],
        "label": result["label"],
        "group": result["group"],
        "weights": "" if weights is None else json.dumps(weights, ensure_ascii=False, sort_keys=True),
        "scenario": result["scenario"],
        "top_n": result["top_n"],
        "horizon": result["horizon"],
        "eligible_days": result["eligible_days"],
        "selected_rows": diagnostics["selected_rows"],
        "full_days": diagnostics["full_days"],
        "avg_picks_per_day": diagnostics["avg_picks_per_day"],
        "exec_mean_ret": all_metrics["exec_mean_ret"],
        "exec_edge_ret": all_metrics["exec_edge_ret"],
        "exec_nav_end": all_metrics["exec_nav_end"],
        "exec_max_dd": all_metrics["exec_max_dd"],
        "exec_tradeoff": all_metrics["exec_tradeoff"],
        "exec_hit15": all_metrics["exec_hit15"],
        "ctc_mean_ret": all_metrics["ctc_mean_ret"],
        "ctc_edge_ret": all_metrics["ctc_edge_ret"],
        "ctc_nav_end": all_metrics["ctc_nav_end"],
        "ctc_max_dd": all_metrics["ctc_max_dd"],
        "ctc_tradeoff": all_metrics["ctc_tradeoff"],
        "ctc_hit15": all_metrics["ctc_hit15"],
        "close_limit_up_rows": diagnostics["close_limit_up_rows"],
        "close_limit_up_day_share": diagnostics["close_limit_up_day_share"],
        "entry_limit_up_rows": diagnostics["entry_limit_up_rows"],
        "entry_limit_up_day_share": diagnostics["entry_limit_up_day_share"],
        "high_open_rows": diagnostics["high_open_rows"],
        "high_open_day_share": diagnostics["high_open_day_share"],
        "rank_mean": diagnostics["rank_mean"],
        "rank_q95": diagnostics["rank_q95"],
        "rank_max": diagnostics["rank_max"],
        "stable_positive_years": result["stable_positive_years"],
        "exec_edge_sharpe_by_year": result["exec_edge_sharpe_by_year"],
        "exec_edge_2025": _yearly_metric(result, 2025, "exec_edge_ret"),
        "exec_edge_2026": _yearly_metric(result, 2026, "exec_edge_ret"),
    }


def _top_rows(rows: list[dict[str, Any]], *, scenario: str, top_k: int) -> dict[str, list[dict[str, Any]]]:
    candidates = [row for row in rows if row["scenario"] == scenario]
    return {
        "by_exec_tradeoff": sorted(
            candidates,
            key=lambda row: (
                row["exec_tradeoff"],
                row["exec_nav_end"],
                row["exec_edge_ret"],
            ),
            reverse=True,
        )[:top_k],
        "by_exec_edge": sorted(
            candidates,
            key=lambda row: (
                row["exec_edge_ret"],
                row["exec_mean_ret"],
                row["exec_tradeoff"],
            ),
            reverse=True,
        )[:top_k],
        "by_low_pollution_exec": sorted(
            candidates,
            key=lambda row: (
                row["exec_tradeoff"],
                -row["close_limit_up_day_share"],
                -row["high_open_day_share"],
            ),
            reverse=True,
        )[:top_k],
    }


def _write_outputs(
    *,
    output_root: Path,
    started_at: datetime,
    args: argparse.Namespace,
    results_by_horizon: dict[str, Any],
    compact_rows: list[dict[str, Any]],
    daily_frames: list[pl.DataFrame],
) -> Path:
    output_dir = output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    compact_path = output_dir / "compact.csv"
    daily_path = output_dir / "daily.csv"

    compact = pl.DataFrame(compact_rows)
    compact.write_csv(compact_path)
    if daily_frames:
        pl.concat(daily_frames, how="vertical").write_csv(daily_path)

    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "git_commit": _git_commit(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "horizons": args.horizons,
            "top_n": args.top_n,
            "top_k": args.top_k,
            "max_scan_rank": args.max_scan_rank,
            "execution_lag_days": args.execution_lag_days,
            "high_open_pct": args.high_open_pct,
            "price_limit_tolerance": args.price_limit_tolerance,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "files": {
            "compact": _rel_path(compact_path, output_dir),
            "daily": _rel_path(daily_path, output_dir),
        },
        "results_by_horizon": results_by_horizon,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV executable-aware factor/weight grid v2")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-scan-rank", type=int, default=0, help="0 表示补位时扫描全候选池")
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--high-open-pct", type=float, default=0.098)
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--max-rankers", type=int, default=0, help="smoke test 用；0 表示评估全部 ranker")
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.execution_lag_days <= 0:
        raise ValueError("--execution-lag-days must be positive")
    if args.max_scan_rank < 0:
        raise ValueError("--max-scan-rank must be non-negative")

    started_at = datetime.now()
    raw_rankers = build_rankers()
    if args.max_rankers > 0:
        raw_rankers = raw_rankers[: args.max_rankers]

    results_by_horizon: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []

    print(f"Rankers: {len(raw_rankers)}")
    for horizon in args.horizons:
        print(f"\nBuilding executable dataset for horizon={horizon}d...")
        df_pool = _build_dataset_for_horizon(args, horizon)
        rankers = copy.deepcopy(raw_rankers)
        df_pool = add_combo_scores(df_pool, rankers)
        print(f"Rows: {df_pool.height:,}")
        print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
        print(f"Unique codes: {df_pool['code'].n_unique():,}")

        horizon_results: list[dict[str, Any]] = []
        for idx, ranker in enumerate(rankers, start=1):
            if idx % 25 == 0 or idx == 1 or idx == len(rankers):
                print(f"  Evaluating {idx}/{len(rankers)}: {ranker['id']}")
            evaluated, daily = evaluate_ranker(
                df_pool,
                ranker,
                horizon=horizon,
                top_n=args.top_n,
                max_scan_rank=args.max_scan_rank,
                high_open_pct=args.high_open_pct,
            )
            horizon_results.extend(evaluated)
            daily_frames.extend(daily)

        valid_results = [result for result in horizon_results if "error" not in result]
        rows = [_compact_row(result) for result in valid_results]
        compact_rows.extend(rows)
        reference = next(
            (
                row
                for row in rows
                if row["id"] == "ref_p2_k0p5_r0" and row["scenario"] == "original_top3"
            ),
            None,
        )
        results_by_horizon[str(horizon)] = {
            "reference": reference,
            "original_top3": _top_rows(rows, scenario="original_top3", top_k=args.top_k),
            "skip_close_limit_refill_top3": _top_rows(
                rows,
                scenario="skip_close_limit_refill_top3",
                top_k=args.top_k,
            ),
            "all_rows": rows,
        }

        if reference:
            print(
                "Reference original_top3: "
                f"exec_nav={reference['exec_nav_end'] * 100:+.2f}% "
                f"exec_ret={reference['exec_mean_ret'] * 100:+.3f}% "
                f"ctc_nav={reference['ctc_nav_end'] * 100:+.2f}% "
                f"close_limit_days={reference['close_limit_up_day_share'] * 100:.1f}%"
            )

    summary_path = _write_outputs(
        output_root=args.output_root,
        started_at=started_at,
        args=args,
        results_by_horizon=results_by_horizon,
        compact_rows=compact_rows,
        daily_frames=daily_frames,
    )
    print(f"\nSaved: {summary_path}")

    target_horizon = str(args.horizons[0])
    print(f"\nTop executable tradeoff horizon={target_horizon} original_top3:")
    for row in results_by_horizon[target_horizon]["original_top3"]["by_exec_tradeoff"][:10]:
        print(
            f"- {row['id']:<32} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"exec_ret={row['exec_mean_ret'] * 100:+.3f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"close_limit_days={row['close_limit_up_day_share'] * 100:.1f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
