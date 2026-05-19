from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.amv_bull_pool_ranker_lab import (
    COMBO_RANKERS,
    DEFAULT_QMT_DB,
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    RANKERS,
    add_combo_scores,
)
from scripts.amv_executable_weight_grid import (
    LIMIT_TOLERANCE,
    _build_dataset_for_horizon,
    _compact_row,
    _parse_int_list,
    _top_rows,
    _write_outputs,
    evaluate_ranker,
)


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_executable_factor_scan"


def build_factor_scan_rankers(*, include_combos: bool) -> list[dict[str, Any]]:
    rankers = copy.deepcopy(RANKERS)
    if include_combos:
        rankers.extend(copy.deepcopy(COMBO_RANKERS))
    return rankers


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV executable-aware early factor scan")
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
    parser.add_argument("--no-combos", action="store_true", help="只跑早期单因子 rankers，不跑组合 rankers")
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
    raw_rankers = build_factor_scan_rankers(include_combos=not args.no_combos)
    if args.max_rankers > 0:
        raw_rankers = raw_rankers[: args.max_rankers]

    results_by_horizon: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    daily_frames = []

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
            if idx % 15 == 0 or idx == 1 or idx == len(rankers):
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
        results_by_horizon[str(horizon)] = {
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
            f"group={row['group']:<12} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"exec_ret={row['exec_mean_ret'] * 100:+.3f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"close_limit_days={row['close_limit_up_day_share'] * 100:.1f}%"
        )

    print(f"\nTop executable tradeoff horizon={target_horizon} refill:")
    for row in results_by_horizon[target_horizon]["skip_close_limit_refill_top3"][
        "by_exec_tradeoff"
    ][:10]:
        print(
            f"- {row['id']:<32} "
            f"group={row['group']:<12} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"exec_ret={row['exec_mean_ret'] * 100:+.3f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"rank_q95={row['rank_q95']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
