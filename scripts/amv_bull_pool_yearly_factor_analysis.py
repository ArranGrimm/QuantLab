from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scripts.amv_bull_pool_combo_grid import _build_grid_rankers, _parse_int_list
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
    _finite_expr,
    _rolling_sleeve_nav,
)


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_yearly_factor"


def _reference_ranker() -> dict[str, Any]:
    return {
        "id": "ref_high_pos_kbar_p2_k0p5_r0",
        "label": "当前基线: 高位+K线确认 P2/K0.5/R0",
        "group": "当前基线",
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 0.5},
            {"factor": "KMID2", "higher_is_better": True, "weight": 0.5},
        ],
    }


def _component_rankers() -> list[dict[str, Any]]:
    return [
        {
            "id": "component_price_pos_20d",
            "label": "单因子: 20日高位",
            "group": "组成因子",
            "factor": "price_pos_20d",
            "descending": True,
        },
        {
            "id": "component_near_high_20d",
            "label": "单因子: 接近20日新高",
            "group": "组成因子",
            "factor": "close_to_high_20d",
            "descending": False,
        },
        {
            "id": "component_klen_contract",
            "label": "单因子: K线振幅收缩",
            "group": "组成因子",
            "factor": "KLEN",
            "descending": False,
        },
        {
            "id": "component_kmid2_strong",
            "label": "单因子: 实体占比偏强",
            "group": "组成因子",
            "factor": "KMID2",
            "descending": True,
        },
    ]


def _ranker_factor_cols(ranker: dict[str, Any]) -> list[str]:
    if "factor_cols" in ranker:
        return [str(col) for col in ranker["factor_cols"]]
    if "components" in ranker:
        return [str(component["factor"]) for component in ranker["components"]]
    return [str(ranker["factor"])]


def _daily_baseline(df_pool: pl.DataFrame, dates: list[object], horizon: int) -> pl.DataFrame:
    return (
        df_pool.filter(pl.col("date").is_in(dates))
        .group_by("date")
        .agg(
            [
                pl.col(f"fwd_ret_{horizon}d").mean().alias("baseline_daily_ret"),
                (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("baseline_hit15"),
            ]
        )
        .sort("date")
    )


def _summarize_daily(per_day: pl.DataFrame, horizon: int) -> dict[str, float]:
    daily_ret = per_day["daily_ret"].to_numpy()
    baseline_ret = per_day["baseline_daily_ret"].to_numpy()
    nav_end, max_dd = _rolling_sleeve_nav(daily_ret, horizon)
    baseline_nav_end, baseline_max_dd = _rolling_sleeve_nav(baseline_ret, horizon)
    mean_ret = float(per_day["daily_ret"].mean())
    baseline_mean_ret = float(per_day["baseline_daily_ret"].mean())
    return {
        "mean_ret": mean_ret,
        "baseline_mean_ret": baseline_mean_ret,
        "edge_ret": mean_ret - baseline_mean_ret,
        "hit15": float(per_day["hit15"].mean()),
        "baseline_hit15": float(per_day["baseline_hit15"].mean()),
        "nav_end": nav_end,
        "baseline_nav_end": baseline_nav_end,
        "edge_nav_end": nav_end - baseline_nav_end,
        "max_dd": max_dd,
        "baseline_max_dd": baseline_max_dd,
    }


def evaluate_ranker_by_year(
    df_pool: pl.DataFrame,
    ranker: dict[str, Any],
    *,
    horizon: int,
    top_n: int,
) -> dict[str, Any]:
    factor = ranker["factor"]
    finite_checks = [_finite_expr(col_name) for col_name in _ranker_factor_cols(ranker)]
    valid_expr = finite_checks[0]
    for finite_check in finite_checks[1:]:
        valid_expr = valid_expr & finite_check
    df_valid = df_pool.filter(valid_expr & _finite_expr(factor))
    daily_counts = (
        df_valid.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= top_n)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    if not eligible_dates:
        return {
            **ranker,
            "error": f"no dates with at least {top_n} finite factor values",
        }

    selected = (
        df_valid.filter(pl.col("date").is_in(eligible_dates))
        .sort(["date", factor, "code"], descending=[False, bool(ranker["descending"]), False])
        .group_by("date", maintain_order=True)
        .head(top_n)
        .sort(["date", "code"])
    )
    per_day = (
        selected.group_by("date")
        .agg(
            [
                pl.col(f"fwd_ret_{horizon}d").mean().alias("daily_ret"),
                (pl.col(f"fwd_mfe_{horizon}d") >= 0.15).mean().alias("hit15"),
            ]
        )
        .join(_daily_baseline(df_pool, eligible_dates, horizon), on="date", how="inner")
        .with_columns(pl.col("date").dt.year().alias("year"))
        .sort("date")
    )

    yearly: list[dict[str, Any]] = []
    for year in per_day["year"].unique().sort().to_list():
        year_day = per_day.filter(pl.col("year") == year)
        if year_day.height < horizon:
            continue
        metrics = _summarize_daily(year_day, horizon)
        yearly.append(
            {
                "year": int(year),
                "days": int(year_day.height),
                **metrics,
            }
        )

    all_metrics = _summarize_daily(per_day, horizon)
    edge_values = [row["edge_ret"] for row in yearly]
    return {
        "id": ranker["id"],
        "label": ranker["label"],
        "group": ranker["group"],
        "top_n": top_n,
        "horizon": horizon,
        "eligible_days": len(eligible_dates),
        "selected_rows": selected.height,
        "all": all_metrics,
        "yearly": yearly,
        "stable_positive_years": int(sum(edge > 0 for edge in edge_values)),
        "negative_edge_years": int(sum(edge <= 0 for edge in edge_values)),
        "edge_mean_by_year": float(np.mean(edge_values)) if edge_values else None,
        "edge_std_by_year": float(np.std(edge_values)) if edge_values else None,
        "edge_sharpe_by_year": float(np.mean(edge_values) / np.std(edge_values))
        if len(edge_values) > 1 and np.std(edge_values) > 0
        else None,
    }


def _row_for_year(result: dict[str, Any], year: int) -> dict[str, Any] | None:
    for row in result["yearly"]:
        if row["year"] == year:
            return row
    return None


def _compact_grid_row(result: dict[str, Any]) -> dict[str, Any]:
    row_2024 = _row_for_year(result, 2024)
    row_2026 = _row_for_year(result, 2026)
    return {
        "id": result["id"],
        "label": result["label"],
        "weights": result.get("weights"),
        "all_edge_ret": result["all"]["edge_ret"],
        "all_mean_ret": result["all"]["mean_ret"],
        "all_nav_end": result["all"]["nav_end"],
        "all_max_dd": result["all"]["max_dd"],
        "stable_positive_years": result["stable_positive_years"],
        "negative_edge_years": result["negative_edge_years"],
        "edge_mean_by_year": result["edge_mean_by_year"],
        "edge_std_by_year": result["edge_std_by_year"],
        "edge_sharpe_by_year": result["edge_sharpe_by_year"],
        "edge_2024": row_2024["edge_ret"] if row_2024 else None,
        "mean_2024": row_2024["mean_ret"] if row_2024 else None,
        "edge_2026": row_2026["edge_ret"] if row_2026 else None,
        "mean_2026": row_2026["mean_ret"] if row_2026 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool yearly factor stability analysis")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[5, 6, 10, 20])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    started_at = datetime.now()
    print("Building AMV bull pool factor dataset...")
    df_pool = build_dataset(args)
    focus_rankers = [_reference_ranker(), *_component_rankers()]
    grid_rankers = _build_grid_rankers()
    rankers_for_score = [dict(ranker) for ranker in [*focus_rankers, *grid_rankers]]
    df_pool = add_combo_scores(df_pool, rankers_for_score)
    scored_focus_rankers = rankers_for_score[: len(focus_rankers)]
    scored_grid_rankers = rankers_for_score[len(focus_rankers) :]

    results_by_horizon: dict[str, Any] = {}
    for horizon in args.horizons:
        print(f"\nEvaluating horizon={horizon}d...")
        focus_results = [
            evaluate_ranker_by_year(df_pool, ranker, horizon=horizon, top_n=args.top_n)
            for ranker in scored_focus_rankers
        ]
        grid_results = [
            evaluate_ranker_by_year(df_pool, ranker, horizon=horizon, top_n=args.top_n)
            for ranker in scored_grid_rankers
        ]
        compact_grid = [_compact_grid_row(result) for result in grid_results if "error" not in result]
        results_by_horizon[str(horizon)] = {
            "focus_rankers": focus_results,
            "grid": {
                "by_all_edge": sorted(
                    compact_grid,
                    key=lambda row: (row["all_edge_ret"], row["all_mean_ret"]),
                    reverse=True,
                )[:10],
                "by_yearly_stability": sorted(
                    compact_grid,
                    key=lambda row: (
                        row["stable_positive_years"],
                        row["edge_sharpe_by_year"] if row["edge_sharpe_by_year"] is not None else -999.0,
                        row["all_edge_ret"],
                    ),
                    reverse=True,
                )[:10],
                "by_2026_edge": sorted(
                    compact_grid,
                    key=lambda row: (
                        row["edge_2026"] if row["edge_2026"] is not None else -999.0,
                        row["all_edge_ret"],
                    ),
                    reverse=True,
                )[:10],
                "all_rows": compact_grid,
            },
        }

    output_dir = args.output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "qmt_db": str(args.qmt_db),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "st_snapshot_date": args.st_snapshot_date,
            "mv_min": args.mv_min,
            "amount_ma20_min": args.amount_ma20_min,
            "horizons": args.horizons,
            "top_n": args.top_n,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "pool": {
            "rows": df_pool.height,
            "date_min": str(df_pool["date"].min()),
            "date_max": str(df_pool["date"].max()),
            "unique_codes": df_pool["code"].n_unique(),
        },
        "results_by_horizon": results_by_horizon,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    for horizon in args.horizons:
        ref = results_by_horizon[str(horizon)]["focus_rankers"][0]
        print(f"\nReference {horizon}d by year:")
        for row in ref["yearly"]:
            print(
                f"- {row['year']}: ret={row['mean_ret'] * 100:+.3f}% "
                f"edge={row['edge_ret'] * 100:+.3f}pp "
                f"nav={row['nav_end'] * 100:+.2f}% dd={row['max_dd'] * 100:.2f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
