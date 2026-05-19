from __future__ import annotations

import argparse
import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_QMT_DB,
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
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


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_executable_pullback_grid"


def _weight_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _component(factor: str, *, higher_is_better: bool, weight: float) -> dict[str, Any]:
    return {
        "factor": factor,
        "higher_is_better": higher_is_better,
        "weight": weight,
    }


def _ratio_key(values: tuple[float, ...]) -> tuple[int, ...]:
    scaled = tuple(int(round(value * 2)) for value in values)
    divisor = 0
    for value in scaled:
        divisor = math.gcd(divisor, value)
    divisor = divisor or 1
    return tuple(value // divisor for value in scaled)


def _reference_rankers() -> list[dict[str, Any]]:
    return [
        _make_ranker(
            price_weight=2.0,
            kbar_weight=0.5,
            bias_weight=0.0,
            close_pullback_weight=0.0,
            risk_weight=0.0,
            group="reference",
            label_prefix="reference",
        ),
        _make_ranker(
            price_weight=3.0,
            kbar_weight=0.5,
            bias_weight=0.0,
            close_pullback_weight=0.0,
            risk_weight=0.0,
            group="reference",
            label_prefix="candidate",
        ),
    ]


def _make_ranker(
    *,
    price_weight: float,
    kbar_weight: float,
    bias_weight: float,
    close_pullback_weight: float,
    risk_weight: float,
    group: str = "grid_pullback",
    label_prefix: str = "pullback",
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    if price_weight > 0:
        components.extend(
            [
                _component("price_pos_20d", higher_is_better=True, weight=price_weight),
                _component("close_to_high_20d", higher_is_better=False, weight=price_weight),
            ]
        )
    if kbar_weight > 0:
        components.extend(
            [
                _component("KLEN", higher_is_better=False, weight=kbar_weight),
                _component("KMID2", higher_is_better=True, weight=kbar_weight),
            ]
        )
    if bias_weight > 0:
        components.extend(
            [
                _component("ma_bias_20", higher_is_better=False, weight=bias_weight),
                _component("disp_bias_20", higher_is_better=False, weight=bias_weight),
            ]
        )
    if close_pullback_weight > 0:
        components.extend(
            [
                _component("KSFT", higher_is_better=False, weight=close_pullback_weight),
                _component("intraday_pos", higher_is_better=False, weight=close_pullback_weight),
            ]
        )
    if risk_weight > 0:
        components.extend(
            [
                _component("atr_14_pct", higher_is_better=False, weight=risk_weight),
                _component("panic_vol_ratio_20d", higher_is_better=False, weight=risk_weight),
            ]
        )

    return {
        "id": (
            f"{label_prefix}"
            f"_p{_weight_token(price_weight)}"
            f"_k{_weight_token(kbar_weight)}"
            f"_b{_weight_token(bias_weight)}"
            f"_c{_weight_token(close_pullback_weight)}"
            f"_r{_weight_token(risk_weight)}"
        ),
        "label": (
            f"{label_prefix} "
            f"P{price_weight:g}/K{kbar_weight:g}/B{bias_weight:g}/"
            f"C{close_pullback_weight:g}/R{risk_weight:g}"
        ),
        "group": group,
        "weights": {
            "price": price_weight,
            "kbar": kbar_weight,
            "bias_pullback": bias_weight,
            "close_pullback": close_pullback_weight,
            "risk": risk_weight,
        },
        "components": components,
    }


def build_pullback_rankers(*, preset: str = "focused") -> list[dict[str, Any]]:
    if preset == "full":
        price_weights = (0.0, 1.0, 2.0, 3.0)
        kbar_weights = (0.0, 0.5, 1.0)
        bias_weights = (0.0, 0.5, 1.0, 2.0, 3.0)
        close_pullback_weights = (0.0, 0.5, 1.0, 2.0)
        risk_weights = (0.0, 0.5, 1.0)
    elif preset == "focused":
        price_weights = (0.0, 1.0, 2.0, 3.0)
        kbar_weights = (0.0, 0.5, 1.0)
        bias_weights = (0.0, 1.0, 2.0, 3.0)
        close_pullback_weights = (0.0, 1.0)
        risk_weights = (0.0, 0.5)
    else:
        raise ValueError(f"unknown grid preset: {preset}")

    rankers = _reference_rankers()
    seen = {
        _ratio_key((2.0, 0.5, 0.0, 0.0, 0.0)),
        _ratio_key((3.0, 0.5, 0.0, 0.0, 0.0)),
    }
    for price_weight in price_weights:
        for kbar_weight in kbar_weights:
            for bias_weight in bias_weights:
                for close_pullback_weight in close_pullback_weights:
                    for risk_weight in risk_weights:
                        if bias_weight == 0 and close_pullback_weight == 0:
                            continue
                        if (
                            price_weight == 0
                            and kbar_weight == 0
                            and bias_weight + close_pullback_weight < 1.0
                        ):
                            continue

                        ratio = _ratio_key(
                            (
                                price_weight,
                                kbar_weight,
                                bias_weight,
                                close_pullback_weight,
                                risk_weight,
                            )
                        )
                        if ratio in seen:
                            continue
                        seen.add(ratio)
                        rankers.append(
                            _make_ranker(
                                price_weight=price_weight,
                                kbar_weight=kbar_weight,
                                bias_weight=bias_weight,
                                close_pullback_weight=close_pullback_weight,
                                risk_weight=risk_weight,
                            )
                        )
    return rankers


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV executable-aware pullback combo grid")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-scan-rank", type=int, default=0, help="0 表示补位时扫描全候选池")
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--high-open-pct", type=float, default=0.098)
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--max-rankers", type=int, default=0, help="smoke test 用；0 表示评估全部 ranker")
    parser.add_argument("--grid-preset", choices=["focused", "full"], default="focused")
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
    raw_rankers = build_pullback_rankers(preset=args.grid_preset)
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
            if idx % 50 == 0 or idx == 1 or idx == len(rankers):
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
    for row in results_by_horizon[target_horizon]["original_top3"]["by_exec_tradeoff"][:12]:
        print(
            f"- {row['id']:<42} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"exec_ret={row['exec_mean_ret'] * 100:+.3f}% "
            f"dd={row['exec_max_dd'] * 100:.2f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"close_limit_days={row['close_limit_up_day_share'] * 100:.1f}%"
        )

    print(f"\nTop executable tradeoff horizon={target_horizon} refill:")
    for row in results_by_horizon[target_horizon]["skip_close_limit_refill_top3"][
        "by_exec_tradeoff"
    ][:12]:
        print(
            f"- {row['id']:<42} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"exec_ret={row['exec_mean_ret'] * 100:+.3f}% "
            f"dd={row['exec_max_dd'] * 100:.2f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"rank_q95={row['rank_q95']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
