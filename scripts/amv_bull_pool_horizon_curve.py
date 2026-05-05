from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.amv_bull_pool_combo_grid import _parse_int_list
from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
    evaluate_ranker,
)


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_horizon_curve"


def _build_reference_ranker() -> dict[str, Any]:
    return {
        "id": "horizon_top3_high_pos_kbar_p2_k0p5_r0",
        "label": "top3 高位+K线确认 P2/K0.5/R0",
        "group": "持有期曲线",
        "weights": {
            "price": 2.0,
            "kbar": 0.5,
            "risk": 0.0,
        },
        "components": [
            {"factor": "price_pos_20d", "higher_is_better": True, "weight": 2.0},
            {"factor": "close_to_high_20d", "higher_is_better": False, "weight": 2.0},
            {"factor": "KLEN", "higher_is_better": False, "weight": 0.5},
            {"factor": "KMID2", "higher_is_better": True, "weight": 0.5},
        ],
    }


def _curve_rows(result: dict[str, Any], horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_mean_ret = None
    prev_horizon = None
    for horizon in horizons:
        metrics = result["horizons"][str(horizon)]
        mean_ret = metrics["mean_ret"]
        edge_ret = metrics["edge_ret"]
        marginal_ret = None if prev_mean_ret is None else mean_ret - prev_mean_ret
        marginal_per_day = (
            None
            if prev_mean_ret is None or prev_horizon is None
            else (mean_ret - prev_mean_ret) / (horizon - prev_horizon)
        )
        rows.append(
            {
                "horizon": horizon,
                "mean_ret": mean_ret,
                "edge_ret": edge_ret,
                "mean_ret_per_day": mean_ret / horizon,
                "edge_ret_per_day": edge_ret / horizon,
                "marginal_ret": marginal_ret,
                "marginal_ret_per_day": marginal_per_day,
                "nav_end": metrics["nav_end"],
                "max_dd": metrics["max_dd"],
                "hit15": metrics["hit15"],
                "baseline_mean_ret": metrics["random_baseline"]["mean_ret"],
                "baseline_nav_end": metrics["random_baseline"]["nav_end"],
            }
        )
        prev_mean_ret = mean_ret
        prev_horizon = horizon
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool top3 horizon realization curve")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[1, 2, 3, 5, 10, 15, 20, 30])
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
    ranker = _build_reference_ranker()
    rankers = [ranker]
    df_pool = add_combo_scores(df_pool, rankers)
    print(f"AMV bull LF2 rows: {df_pool.height:,}")
    print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
    print(f"Unique codes: {df_pool['code'].n_unique():,}")
    print(f"Evaluating {ranker['label']} with top{args.top_n}...")

    result = evaluate_ranker(
        df_pool,
        ranker,
        horizons=args.horizons,
        top_n=args.top_n,
    )
    curve = _curve_rows(result, args.horizons)

    best_by_mean = max(curve, key=lambda row: row["mean_ret"])
    best_by_edge_per_day = max(curve, key=lambda row: row["edge_ret_per_day"])
    best_by_tradeoff = max(curve, key=lambda row: row["nav_end"] + row["max_dd"])

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
        "ranker": result,
        "curve": curve,
        "best": {
            "by_mean_ret": best_by_mean,
            "by_edge_ret_per_day": best_by_edge_per_day,
            "by_tradeoff": best_by_tradeoff,
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    print("\nHorizon realization curve:")
    for row in curve:
        marginal = "n/a" if row["marginal_ret"] is None else f"{row['marginal_ret'] * 100:+.3f}%"
        print(
            f"- {row['horizon']:>2}d "
            f"ret={row['mean_ret'] * 100:+.3f}% "
            f"edge={row['edge_ret'] * 100:+.3f}pp "
            f"ret/day={row['mean_ret_per_day'] * 100:+.3f}% "
            f"marginal={marginal} "
            f"nav={row['nav_end'] * 100:+.2f}% "
            f"dd={row['max_dd'] * 100:.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
