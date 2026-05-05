from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
    DEFAULT_QMT_DB,
    add_combo_scores,
    build_dataset,
    evaluate_ranker,
)


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_bull_pool_combo_grid"


def _parse_int_list(value: str) -> list[int]:
    items = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return items


def _weight_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _build_grid_rankers() -> list[dict[str, Any]]:
    rankers: list[dict[str, Any]] = []
    price_weights = (1.0, 2.0, 3.0)
    kbar_weights = (0.5, 1.0, 2.0)
    risk_weights = (0.0, 0.5, 1.0, 1.5)

    for price_weight in price_weights:
        for kbar_weight in kbar_weights:
            for risk_weight in risk_weights:
                components = [
                    {"factor": "price_pos_20d", "higher_is_better": True, "weight": price_weight},
                    {"factor": "close_to_high_20d", "higher_is_better": False, "weight": price_weight},
                    {"factor": "KLEN", "higher_is_better": False, "weight": kbar_weight},
                    {"factor": "KMID2", "higher_is_better": True, "weight": kbar_weight},
                ]
                if risk_weight > 0:
                    components.extend(
                        [
                            {
                                "factor": "atr_14_pct",
                                "higher_is_better": False,
                                "weight": risk_weight,
                            },
                            {
                                "factor": "panic_vol_ratio_20d",
                                "higher_is_better": False,
                                "weight": risk_weight,
                            },
                        ]
                    )

                rankers.append(
                    {
                        "id": (
                            "grid_high_pos_kbar"
                            f"_p{_weight_token(price_weight)}"
                            f"_k{_weight_token(kbar_weight)}"
                            f"_r{_weight_token(risk_weight)}"
                        ),
                        "label": (
                            "高位+K线确认"
                            f" P{price_weight:g}/K{kbar_weight:g}/R{risk_weight:g}"
                        ),
                        "group": "组合网格",
                        "weights": {
                            "price": price_weight,
                            "kbar": kbar_weight,
                            "risk": risk_weight,
                        },
                        "components": components,
                    }
                )

    return rankers


def _result_row(result: dict[str, Any], horizon: int) -> dict[str, Any]:
    horizon_key = str(horizon)
    metrics = result["horizons"][horizon_key]
    tradeoff_score = metrics["nav_end"] + metrics["max_dd"]
    return {
        "id": result["id"],
        "label": result["label"],
        "top_n": result["top_n"],
        "weights": result["weights"],
        "mean_ret": metrics["mean_ret"],
        "edge_ret": metrics["edge_ret"],
        "nav_end": metrics["nav_end"],
        "max_dd": metrics["max_dd"],
        "hit15": metrics["hit15"],
        "edge_nav_end": metrics["edge_nav_end"],
        "edge_max_dd": metrics["edge_max_dd"],
        "tradeoff_score": tradeoff_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV bull pool combo weight grid lab")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[5, 10, 20])
    parser.add_argument("--top-n-list", type=_parse_int_list, default=[3, 5, 10])
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now()
    print("Building AMV bull pool factor dataset...")
    df_pool = build_dataset(args)
    rankers = _build_grid_rankers()
    df_pool = add_combo_scores(df_pool, rankers)
    print(f"AMV bull LF2 rows: {df_pool.height:,}")
    print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
    print(f"Unique codes: {df_pool['code'].n_unique():,}")
    print(f"Grid rankers: {len(rankers)}; top_n_list={args.top_n_list}")

    results = []
    for top_n in args.top_n_list:
        print(f"\nEvaluating top{top_n}...")
        for ranker in rankers:
            result = evaluate_ranker(
                df_pool,
                ranker,
                horizons=args.horizons,
                top_n=top_n,
            )
            result["top_n"] = top_n
            results.append(result)

    best_by_horizon: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for horizon in args.horizons:
        rows = [_result_row(result, horizon) for result in results if "error" not in result]
        best_by_horizon[str(horizon)] = {
            "by_tradeoff": sorted(
                rows,
                key=lambda row: (row["tradeoff_score"], row["nav_end"], row["edge_ret"]),
                reverse=True,
            )[:15],
            "by_nav": sorted(
                rows,
                key=lambda row: (row["nav_end"], row["tradeoff_score"], row["edge_ret"]),
                reverse=True,
            )[:15],
            "by_edge_ret": sorted(
                rows,
                key=lambda row: (row["edge_ret"], row["mean_ret"], row["tradeoff_score"]),
                reverse=True,
            )[:15],
        }

    best_by_top_n: dict[str, list[dict[str, Any]]] = {}
    target_horizon = max(args.horizons)
    for top_n in args.top_n_list:
        rows = [
            _result_row(result, target_horizon)
            for result in results
            if "error" not in result and result["top_n"] == top_n
        ]
        best_by_top_n[str(top_n)] = sorted(
            rows,
            key=lambda row: (row["tradeoff_score"], row["nav_end"], row["edge_ret"]),
            reverse=True,
        )[:10]

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
            "top_n_list": args.top_n_list,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
            "grid": {
                "price_weights": [1.0, 2.0, 3.0],
                "kbar_weights": [0.5, 1.0, 2.0],
                "risk_weights": [0.0, 0.5, 1.0, 1.5],
                "objective_hint": "tradeoff_score = nav_end + max_dd",
            },
        },
        "pool": {
            "rows": df_pool.height,
            "date_min": str(df_pool["date"].min()),
            "date_max": str(df_pool["date"].max()),
            "unique_codes": df_pool["code"].n_unique(),
        },
        "results": results,
        "best_by_horizon": best_by_horizon,
        "best_by_top_n": best_by_top_n,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    print(f"\nTop {target_horizon}d combos by tradeoff score:")
    for row in best_by_horizon[str(target_horizon)]["by_tradeoff"][:10]:
        print(
            f"- top{row['top_n']} {row['label']:<24} "
            f"ret={row['mean_ret'] * 100:+.3f}% "
            f"edge={row['edge_ret'] * 100:+.3f}pp "
            f"nav={row['nav_end'] * 100:+.2f}% "
            f"dd={row['max_dd'] * 100:.2f}% "
            f"score={row['tradeoff_score'] * 100:+.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
