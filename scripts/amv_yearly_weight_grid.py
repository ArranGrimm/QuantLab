from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.amv_bull_pool_combo_grid import _parse_int_list
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
)
from scripts.amv_bull_pool_yearly_factor_analysis import evaluate_ranker_by_year
from strategies.amv.rankers import build_rankers


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_yearly_weight_grid"


def _year_row(result: dict[str, Any], year: int) -> dict[str, Any] | None:
    for row in result["yearly"]:
        if row["year"] == year:
            return row
    return None


def _metric_for_year(result: dict[str, Any], year: int, key: str) -> float | None:
    row = _year_row(result, year)
    if row is None:
        return None
    value = row.get(key)
    return float(value) if value is not None else None


def _tradeoff(row: dict[str, Any]) -> float:
    return float(row["nav_end"] + row["max_dd"])


def _compact_result(result: dict[str, Any], years: list[int]) -> dict[str, Any]:
    yearly = {str(row["year"]): row for row in result["yearly"]}
    edge_values = [
        float(row["edge_ret"])
        for row in result["yearly"]
        if row.get("edge_ret") is not None
    ]
    weak_year_edges = [
        _metric_for_year(result, year, "edge_ret")
        for year in (2025, 2026)
        if _metric_for_year(result, year, "edge_ret") is not None
    ]
    return {
        "id": result["id"],
        "label": result["label"],
        "group": result["group"],
        "weights": result.get("weights"),
        "top_n": result["top_n"],
        "horizon": result["horizon"],
        "eligible_days": result["eligible_days"],
        "all": result["all"],
        "yearly": yearly,
        "stable_positive_years": result["stable_positive_years"],
        "negative_edge_years": result["negative_edge_years"],
        "edge_mean_by_year": result["edge_mean_by_year"],
        "edge_std_by_year": result["edge_std_by_year"],
        "edge_sharpe_by_year": result["edge_sharpe_by_year"],
        "weak_year_edge_mean": float(np.mean(weak_year_edges)) if weak_year_edges else None,
        "year_edges": {
            str(year): _metric_for_year(result, year, "edge_ret") for year in years
        },
    }


def _train_stats(row: dict[str, Any], train_years: list[int]) -> dict[str, Any] | None:
    year_rows = [
        row["yearly"].get(str(year))
        for year in train_years
        if row["yearly"].get(str(year)) is not None
    ]
    if not year_rows:
        return None
    edge_values = [float(year_row["edge_ret"]) for year_row in year_rows]
    tradeoff_values = [_tradeoff(year_row) for year_row in year_rows]
    return {
        "train_years": train_years,
        "train_edge_mean": float(np.mean(edge_values)),
        "train_tradeoff_mean": float(np.mean(tradeoff_values)),
        "train_positive_years": int(sum(edge > 0 for edge in edge_values)),
        "train_edge_std": float(np.std(edge_values)),
    }


def _selection_payload(
    row: dict[str, Any],
    *,
    test_year: int,
    selector: str,
    train_stats: dict[str, Any],
) -> dict[str, Any]:
    test_row = row["yearly"].get(str(test_year))
    return {
        "selector": selector,
        "id": row["id"],
        "label": row["label"],
        "group": row["group"],
        "weights": row.get("weights"),
        **train_stats,
        "test_year": test_year,
        "test": test_row,
    }


def _walk_forward(compact_rows: list[dict[str, Any]], years: list[int]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for test_year in (2025, 2026):
        train_years = [year for year in years if year < test_year]
        candidates = []
        for row in compact_rows:
            train_stats = _train_stats(row, train_years)
            test_row = row["yearly"].get(str(test_year))
            if train_stats is None or test_row is None:
                continue
            candidates.append((row, train_stats))
        if not candidates:
            continue

        best_by_train_edge = max(
            candidates,
            key=lambda item: (
                item[1]["train_edge_mean"],
                item[1]["train_positive_years"],
                item[1]["train_tradeoff_mean"],
            ),
        )
        best_by_train_tradeoff = max(
            candidates,
            key=lambda item: (
                item[1]["train_tradeoff_mean"],
                item[1]["train_positive_years"],
                item[1]["train_edge_mean"],
            ),
        )
        checks.extend(
            [
                _selection_payload(
                    best_by_train_edge[0],
                    test_year=test_year,
                    selector="train_edge_mean",
                    train_stats=best_by_train_edge[1],
                ),
                _selection_payload(
                    best_by_train_tradeoff[0],
                    test_year=test_year,
                    selector="train_tradeoff_mean",
                    train_stats=best_by_train_tradeoff[1],
                ),
            ]
        )
    return checks


def _top_by_year(
    compact_rows: list[dict[str, Any]],
    years: list[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for year in years:
        rows_with_year = [
            row for row in compact_rows if row["yearly"].get(str(year)) is not None
        ]
        output[str(year)] = {
            "by_edge": sorted(
                rows_with_year,
                key=lambda row: (
                    row["yearly"][str(year)]["edge_ret"],
                    row["yearly"][str(year)]["mean_ret"],
                ),
                reverse=True,
            )[:top_k],
            "by_tradeoff": sorted(
                rows_with_year,
                key=lambda row: (
                    _tradeoff(row["yearly"][str(year)]),
                    row["yearly"][str(year)]["edge_ret"],
                ),
                reverse=True,
            )[:top_k],
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV yearly explainable weight grid diagnostic")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    started_at = datetime.now()
    print("Building AMV bull pool factor dataset...")
    df_pool = build_dataset(args)
    raw_rankers = build_rankers()
    rankers = copy.deepcopy(raw_rankers)
    df_pool = add_combo_scores(df_pool, rankers)
    years = [int(year) for year in df_pool["date"].dt.year().unique().sort().to_list()]

    print(f"AMV bull LF2 rows: {df_pool.height:,}")
    print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
    print(f"Unique codes: {df_pool['code'].n_unique():,}")
    print(f"Rankers: {len(rankers)}")

    results_by_horizon: dict[str, Any] = {}
    for horizon in args.horizons:
        print(f"\nEvaluating horizon={horizon}d...")
        evaluated = []
        for ranker in rankers:
            result = evaluate_ranker_by_year(df_pool, ranker, horizon=horizon, top_n=args.top_n)
            if "weights" in ranker:
                result["weights"] = ranker["weights"]
            if "components" in ranker:
                result["components"] = ranker["components"]
            evaluated.append(result)
        valid_results = [result for result in evaluated if "error" not in result]
        compact_rows = [_compact_result(result, years) for result in valid_results]
        reference = next(row for row in compact_rows if row["id"] == "ref_p2_k0p5_r0")
        results_by_horizon[str(horizon)] = {
            "reference": reference,
            "by_year": _top_by_year(compact_rows, years, top_k=args.top_k),
            "overall": {
                "by_all_edge": sorted(
                    compact_rows,
                    key=lambda row: (
                        row["all"]["edge_ret"],
                        row["all"]["mean_ret"],
                    ),
                    reverse=True,
                )[: args.top_k],
                "by_yearly_stability": sorted(
                    compact_rows,
                    key=lambda row: (
                        row["stable_positive_years"],
                        row["edge_sharpe_by_year"]
                        if row["edge_sharpe_by_year"] is not None
                        else -999.0,
                        row["all"]["edge_ret"],
                    ),
                    reverse=True,
                )[: args.top_k],
                "by_weak_year_edge": sorted(
                    compact_rows,
                    key=lambda row: (
                        row["weak_year_edge_mean"]
                        if row["weak_year_edge_mean"] is not None
                        else -999.0,
                        row["all"]["edge_ret"],
                    ),
                    reverse=True,
                )[: args.top_k],
            },
            "walk_forward": _walk_forward(compact_rows, years),
            "all_rows": compact_rows,
        }

        print("Reference by year:")
        for year, row in reference["yearly"].items():
            print(
                f"- {year}: ret={row['mean_ret'] * 100:+.3f}% "
                f"edge={row['edge_ret'] * 100:+.3f}pp "
                f"nav={row['nav_end'] * 100:+.2f}% dd={row['max_dd'] * 100:.2f}%"
            )

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
            "top_k": args.top_k,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "ranker_count": len(rankers),
        },
        "pool": {
            "rows": df_pool.height,
            "date_min": str(df_pool["date"].min()),
            "date_max": str(df_pool["date"].max()),
            "unique_codes": df_pool["code"].n_unique(),
            "years": years,
        },
        "results_by_horizon": results_by_horizon,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
