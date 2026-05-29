from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_export_signals import _git_commit, _rel_path
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_QMT_DB,
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    add_combo_scores,
)
from scripts.amv_executable_factor_scan import build_factor_scan_rankers
from scripts.amv_executable_pullback_grid import build_pullback_rankers
from scripts.amv_executable_weight_grid import (
    LIMIT_TOLERANCE,
    _build_dataset_for_horizon,
    _compact_row,
    _evaluate_scenario,
    _parse_int_list,
    _top_rows,
    evaluate_ranker,
)
from strategies.amv.rankers import build_rankers as build_yearly_rankers
from scripts.b1_executable_base_lab import _base_b1_expr, _base_definition, _build_b1_indicators


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_executable_trend_filter_grid"
RANKER_SETS = ("factor", "pullback", "yearly")


def _parse_ranker_set(value: str) -> list[str]:
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("ranker set must not be empty")
    if "all" in parts:
        if len(parts) > 1:
            raise argparse.ArgumentTypeError("'all' cannot be combined with other ranker sets")
        return list(RANKER_SETS)
    unknown = sorted(set(parts) - set(RANKER_SETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown ranker set(s): {', '.join(unknown)}")
    ordered = [ranker_set for ranker_set in RANKER_SETS if ranker_set in set(parts)]
    return ordered


def _prefix_ranker(ranker: dict[str, Any], prefix: str) -> dict[str, Any]:
    copied = copy.deepcopy(ranker)
    copied["original_id"] = copied["id"]
    copied["id"] = f"{prefix}_{copied['id']}"
    copied["label"] = f"{prefix}: {copied['label']}"
    copied["group"] = f"{prefix}_{copied.get('group', 'ranker')}"
    return copied


def build_trend_filter_rankers(*, ranker_sets: list[str], grid_preset: str) -> list[dict[str, Any]]:
    rankers: list[dict[str, Any]] = []
    if "factor" in ranker_sets:
        rankers.extend(
            _prefix_ranker(ranker, "factor")
            for ranker in build_factor_scan_rankers(include_combos=True)
        )
    if "pullback" in ranker_sets:
        rankers.extend(
            _prefix_ranker(ranker, "pullback")
            for ranker in build_pullback_rankers(preset=grid_preset)
        )
    if "yearly" in ranker_sets:
        rankers.extend(_prefix_ranker(ranker, "yearly") for ranker in build_yearly_rankers())
    return rankers


def _evaluate_all_candidates(
    df_pool: pl.DataFrame,
    *,
    horizon: int,
    top_n: int,
    high_open_pct: float,
) -> tuple[dict[str, Any], pl.DataFrame] | tuple[None, None]:
    daily_counts = (
        df_pool.group_by("date")
        .agg(pl.len().alias("n_candidates"))
        .filter(pl.col("n_candidates") >= top_n)
        .sort("date")
    )
    eligible_dates = daily_counts["date"].to_list()
    if not eligible_dates:
        return None, None
    selected = (
        df_pool.filter(pl.col("date").is_in(eligible_dates))
        .sort(["date", "code"])
        .with_columns(pl.col("code").rank("ordinal").over("date").cast(pl.Int64).alias("rank"))
    )
    return _evaluate_scenario(
        df_pool,
        selected,
        ranker={
            "id": "trend_only_all_candidates",
            "label": "trend-only 全部候选平均",
            "group": "trend_filter",
            "weights": None,
            "components": None,
        },
        scenario="all_candidates",
        horizon=horizon,
        top_n=top_n,
        eligible_dates=eligible_dates,
        high_open_pct=high_open_pct,
    )


def _write_outputs(
    *,
    output_root: Path,
    started_at: datetime,
    args: argparse.Namespace,
    pool_stats: dict[str, Any],
    results_by_horizon: dict[str, Any],
    compact_rows: list[dict[str, Any]],
    daily_frames: list[pl.DataFrame],
) -> Path:
    output_dir = output_root / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    compact_path = output_dir / "compact.csv"
    daily_path = output_dir / "daily.csv"

    pl.DataFrame(compact_rows).write_csv(compact_path)
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
            "ranker_set": args.ranker_set,
            "ranker_sets": args.ranker_sets,
            "grid_preset": args.grid_preset,
            "max_rankers": args.max_rankers,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "definition": {
            "base_mode": "trend_only",
            "filter": _base_definition("trend_only"),
        },
        "pool": pool_stats,
        "files": {
            "compact": _rel_path(compact_path, output_dir),
            "daily": _rel_path(daily_path, output_dir),
        },
        "results_by_horizon": results_by_horizon,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable-aware grid inside AMV trend-only filter")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-scan-rank", type=int, default=0, help="0 表示补位时扫描全候选池")
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--high-open-pct", type=float, default=0.098)
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--ranker-set", type=_parse_ranker_set, default=list(RANKER_SETS))
    parser.add_argument("--grid-preset", choices=["focused", "full"], default="focused")
    parser.add_argument("--max-rankers", type=int, default=0, help="smoke test 用；0 表示评估全部 ranker")
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()
    args.ranker_sets = args.ranker_set
    args.ranker_set = ",".join(args.ranker_sets)
    return args


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.execution_lag_days <= 0:
        raise ValueError("--execution-lag-days must be positive")
    if args.max_scan_rank < 0:
        raise ValueError("--max-scan-rank must be non-negative")

    started_at = datetime.now()
    b1_indicators = _build_b1_indicators(args)
    raw_rankers = build_trend_filter_rankers(
        ranker_sets=args.ranker_sets,
        grid_preset=args.grid_preset,
    )
    if args.max_rankers > 0:
        raw_rankers = raw_rankers[: args.max_rankers]

    results_by_horizon: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    pool_stats: dict[str, Any] = {"rankers": len(raw_rankers), "by_horizon": {}}

    print(f"Ranker sets: {args.ranker_set}")
    print(f"Rankers: {len(raw_rankers)}")
    for horizon in args.horizons:
        print(f"\nBuilding executable AMV pool for horizon={horizon}d...")
        df_pool = _build_dataset_for_horizon(args, horizon).join(
            b1_indicators, on=["date", "code"], how="left"
        )
        df_pool = df_pool.with_columns(_base_b1_expr("trend_only", 13.0).alias("trend_only"))
        trend_pool = df_pool.filter(pl.col("trend_only"))
        rankers = copy.deepcopy(raw_rankers)
        trend_pool = add_combo_scores(trend_pool, rankers)

        horizon_key = str(horizon)
        pool_stats["by_horizon"][horizon_key] = {
            "rows": trend_pool.height,
            "date_min": str(trend_pool["date"].min()),
            "date_max": str(trend_pool["date"].max()),
            "unique_dates": trend_pool.select("date").n_unique(),
            "unique_codes": trend_pool.select("code").n_unique(),
            "avg_candidates_per_day": float(trend_pool.group_by("date").len()["len"].mean()),
        }
        print(
            "Trend-only pool: rows={:,}, days={}, codes={}".format(
                trend_pool.height,
                pool_stats["by_horizon"][horizon_key]["unique_dates"],
                pool_stats["by_horizon"][horizon_key]["unique_codes"],
            )
        )

        horizon_results: list[dict[str, Any]] = []
        all_result, all_daily = _evaluate_all_candidates(
            trend_pool,
            horizon=horizon,
            top_n=args.top_n,
            high_open_pct=args.high_open_pct,
        )
        if all_result is not None and all_daily is not None:
            horizon_results.append(all_result)
            daily_frames.append(all_daily)

        for idx, ranker in enumerate(rankers, start=1):
            if idx % 50 == 0 or idx == 1 or idx == len(rankers):
                print(f"  Evaluating {idx}/{len(rankers)}: {ranker['id']}")
            evaluated, daily = evaluate_ranker(
                trend_pool,
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
        results_by_horizon[horizon_key] = {
            "all_candidates": [row for row in rows if row["scenario"] == "all_candidates"],
            "original_top3": _top_rows(rows, scenario="original_top3", top_k=args.top_k),
            "skip_close_limit_refill_top3": _top_rows(
                rows,
                scenario="skip_close_limit_refill_top3",
                top_k=args.top_k,
            ),
            "all_rows": rows,
        }

    summary_path = _write_outputs(
        output_root=args.output_root,
        started_at=started_at,
        args=args,
        pool_stats=pool_stats,
        results_by_horizon=results_by_horizon,
        compact_rows=compact_rows,
        daily_frames=daily_frames,
    )
    print(f"\nSaved: {summary_path}")

    target_horizon = str(args.horizons[0])
    all_candidates = results_by_horizon[target_horizon]["all_candidates"]
    if all_candidates:
        row = all_candidates[0]
        print(
            "\nTrend-only all candidates: "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"dd={row['exec_max_dd'] * 100:.2f}% "
            f"close_limit_days={row['close_limit_up_day_share'] * 100:.1f}%"
        )

    print(f"\nTop executable tradeoff horizon={target_horizon} refill:")
    for row in results_by_horizon[target_horizon]["skip_close_limit_refill_top3"][
        "by_exec_tradeoff"
    ][:12]:
        print(
            f"- {row['id']:<48} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"dd={row['exec_max_dd'] * 100:.2f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"rank_q95={row['rank_q95']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
