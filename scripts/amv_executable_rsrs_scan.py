from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_bull_pool_ranker_lab import (
    DEFAULT_QMT_DB,
    DEFAULT_OUTPUT_ROOT as RANKER_OUTPUT_ROOT,
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


DEFAULT_OUTPUT_ROOT = RANKER_OUTPUT_ROOT.parent / "amv_executable_rsrs_scan"


def parse_window_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise argparse.ArgumentTypeError("window pairs must be formatted like 18:120,18:250")
        n = int(left)
        m = int(right)
        if n <= 1 or m <= n:
            raise argparse.ArgumentTypeError("RSRS windows require m > n > 1")
        pairs.append((n, m))
    if not pairs:
        raise argparse.ArgumentTypeError("window pair list must not be empty")
    return sorted(set(pairs))


def _finite_or_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def add_rsrs_features(df: pl.DataFrame, window_pairs: list[tuple[int, int]]) -> pl.DataFrame:
    """Add RSRS features using only Polars rolling expressions.

    beta is the rolling OLS slope in high = alpha + beta * low + error.
    R2 is rolling_corr(low, high)^2. z/right variants are standardized per code.
    """
    out = df.sort(["code", "date"])
    beta_windows = sorted({n for n, _ in window_pairs})

    beta_exprs: list[pl.Expr] = []
    for n in beta_windows:
        cov = pl.rolling_cov(
            pl.col("low_adj"),
            pl.col("high_adj"),
            window_size=n,
            min_samples=n,
            ddof=1,
        ).over("code")
        var = pl.col("low_adj").rolling_var(window_size=n, min_samples=n, ddof=1).over("code")
        corr = pl.rolling_corr(
            pl.col("low_adj"),
            pl.col("high_adj"),
            window_size=n,
            min_samples=n,
        ).over("code")
        beta = cov / var
        beta_exprs.extend(
            [
                _finite_or_null(beta).alias(f"rsrs_beta_{n}"),
                _finite_or_null(corr.pow(2)).alias(f"rsrs_r2_{n}"),
            ]
        )
    out = out.with_columns(beta_exprs)

    z_exprs: list[pl.Expr] = []
    for n, m in window_pairs:
        beta_col = pl.col(f"rsrs_beta_{n}")
        beta_mean = beta_col.rolling_mean(window_size=m, min_samples=m).over("code")
        beta_std = beta_col.rolling_std(window_size=m, min_samples=m).over("code")
        z = (beta_col - beta_mean) / beta_std
        r2_col = pl.col(f"rsrs_r2_{n}")
        z_name = f"rsrs_z_{n}_{m}"
        r2_name = f"rsrs_r2adj_{n}_{m}"
        right_name = f"rsrs_right_{n}_{m}"
        z_exprs.extend(
            [
                _finite_or_null(z).alias(z_name),
                _finite_or_null(z * r2_col).alias(r2_name),
                _finite_or_null(z * beta_col * r2_col).alias(right_name),
            ]
        )
    return out.with_columns(z_exprs).sort(["date", "code"])


def build_rsrs_rankers(window_pairs: list[tuple[int, int]], *, include_reverse: bool) -> list[dict[str, Any]]:
    rankers: list[dict[str, Any]] = []
    beta_windows = sorted({n for n, _ in window_pairs})

    def add(factor: str, label: str, descending: bool) -> None:
        suffix = "high" if descending else "low"
        rankers.append(
            {
                "id": f"{factor}_{suffix}",
                "label": f"{label} ({suffix})",
                "group": "rsrs",
                "factor": factor,
                "factor_cols": [factor],
                "descending": descending,
            }
        )

    for n in beta_windows:
        add(f"rsrs_beta_{n}", f"RSRS beta N={n}", True)
        if include_reverse:
            add(f"rsrs_beta_{n}", f"RSRS beta N={n}", False)

    for n, m in window_pairs:
        for factor, label in [
            (f"rsrs_z_{n}_{m}", f"RSRS z N={n} M={m}"),
            (f"rsrs_r2adj_{n}_{m}", f"RSRS z*R2 N={n} M={m}"),
            (f"rsrs_right_{n}_{m}", f"RSRS right-skew N={n} M={m}"),
        ]:
            add(factor, label, True)
            if include_reverse:
                add(factor, label, False)
    return rankers


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV executable-aware RSRS factor scan")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature-start-date", default="2019-01-01")
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
    parser.add_argument("--rsrs-windows", type=parse_window_pairs, default=[(18, 120), (18, 250)])
    parser.add_argument("--include-reverse", action="store_true", help="同时评估低 RSRS 排序方向")
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
    rankers = build_rsrs_rankers(args.rsrs_windows, include_reverse=args.include_reverse)
    if args.max_rankers > 0:
        rankers = rankers[: args.max_rankers]

    results_by_horizon: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []

    print(f"RSRS windows: {args.rsrs_windows}")
    print(f"Rankers: {len(rankers)}")
    for horizon in args.horizons:
        print(f"\nBuilding executable dataset for horizon={horizon}d...")
        dataset_args = copy.copy(args)
        dataset_args.start_date = args.feature_start_date
        df_pool = _build_dataset_for_horizon(dataset_args, horizon)
        print("Adding RSRS features with Polars rolling expressions...")
        df_pool = add_rsrs_features(df_pool, args.rsrs_windows).filter(
            pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
        print(f"Rows: {df_pool.height:,}")
        print(f"Date range: {df_pool['date'].min()} -> {df_pool['date'].max()}")
        print(f"Unique codes: {df_pool['code'].n_unique():,}")

        horizon_results: list[dict[str, Any]] = []
        for idx, ranker in enumerate(rankers, start=1):
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
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["rsrs_config"] = {
        "feature_start_date": args.feature_start_date,
        "rsrs_windows": args.rsrs_windows,
        "include_reverse": args.include_reverse,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {summary_path}")

    target_horizon = str(args.horizons[0])
    print(f"\nTop RSRS horizon={target_horizon} refill:")
    for row in results_by_horizon[target_horizon]["skip_close_limit_refill_top3"][
        "by_exec_tradeoff"
    ][:10]:
        print(
            f"- {row['id']:<28} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"maxdd={row['exec_max_dd'] * 100:.2f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"limit_days={row['close_limit_up_day_share'] * 100:.1f}% "
            f"rank_q95={row['rank_q95']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
