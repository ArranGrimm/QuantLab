from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from scripts.amv_bull_pool_export_signals import ROOT, _git_commit, _rel_path
from scripts.amv_bull_pool_ranker_lab import DEFAULT_QMT_DB, _finite_expr, add_combo_scores
from scripts.amv_executable_weight_grid import (
    LIMIT_TOLERANCE,
    _build_dataset_for_horizon,
    _compact_row,
    _evaluate_scenario,
    _parse_int_list,
    evaluate_ranker,
)
from scripts.amv_executable_pullback_grid import _make_ranker as _make_pullback_ranker
from utils import get_st_blacklist_pl, load_daily_data_full


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "b1_executable_base_lab"


def _build_b1_indicators(args: argparse.Namespace) -> pl.DataFrame:
    conn = duckdb.connect(str(args.qmt_db), read_only=True)
    try:
        st_blacklist = get_st_blacklist_pl(args.st_snapshot_date)
        st_blacklist_df = pl.DataFrame({"code": st_blacklist}, schema={"code": pl.Utf8}).lazy()
        q_full = (
            load_daily_data_full(conn)
            .filter(pl.col("date") >= pl.lit(args.start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(args.end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .join(st_blacklist_df, on="code", how="anti")
            .sort(["code", "date"])
        )

        return (
            q_full.with_columns(
                [
                    pl.col("close_adj")
                    .ewm_mean(span=10, adjust=False)
                    .over("code")
                    .ewm_mean(span=10, adjust=False)
                    .over("code")
                    .alias("b1_WL"),
                    (
                        (
                            pl.col("close_adj").rolling_mean(14).over("code")
                            + pl.col("close_adj").rolling_mean(28).over("code")
                            + pl.col("close_adj").rolling_mean(57).over("code")
                            + pl.col("close_adj").rolling_mean(114).over("code")
                        )
                        / 4.0
                    ).alias("b1_YL"),
                    (
                        pl.col("high_adj").rolling_max(9).over("code")
                        - pl.col("low_adj").rolling_min(9).over("code")
                    ).alias("_b1_kdj_den"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_b1_kdj_den") == 0)
                .then(50.0)
                .otherwise(
                    (
                        pl.col("close_adj")
                        - pl.col("low_adj").rolling_min(9).over("code")
                    )
                    / pl.col("_b1_kdj_den")
                    * 100.0
                )
                .alias("_b1_rsv")
            )
            .with_columns(pl.col("_b1_rsv").ewm_mean(com=2, adjust=False).over("code").alias("_b1_K"))
            .with_columns(pl.col("_b1_K").ewm_mean(com=2, adjust=False).over("code").alias("_b1_D"))
            .with_columns((3.0 * pl.col("_b1_K") - 2.0 * pl.col("_b1_D")).alias("b1_J"))
            .select(["date", "code", "b1_WL", "b1_YL", "b1_J"])
            .collect()
        )
    finally:
        conn.close()


def _base_b1_expr(base_mode: str, j_threshold: float) -> pl.Expr:
    expr = (
        _finite_expr("b1_WL")
        & _finite_expr("b1_YL")
        & _finite_expr("b1_J")
        & (pl.col("close_adj") > pl.col("b1_YL"))
        & (pl.col("b1_WL") > pl.col("b1_YL"))
    )
    if base_mode == "classic":
        return expr & (pl.col("b1_J") <= j_threshold)
    if base_mode == "trend_only":
        return expr
    raise ValueError(f"Unsupported base_mode: {base_mode}")


def _base_definition(base_mode: str) -> str:
    if base_mode == "classic":
        return "close_adj > YL and WL > YL and J <= threshold"
    if base_mode == "trend_only":
        return "close_adj > YL and WL > YL"
    raise ValueError(f"Unsupported base_mode: {base_mode}")


def _single_ranker(
    ranker_id: str,
    label: str,
    factor: str,
    *,
    descending: bool,
    group: str = "b1_base",
) -> dict[str, Any]:
    return {
        "id": ranker_id,
        "label": label,
        "group": group,
        "factor": factor,
        "descending": descending,
    }


def _build_rankers() -> list[dict[str, Any]]:
    rankers = [
        _single_ranker("b1_j_asc", "B1 base / J 越低", "b1_J", descending=False),
        _single_ranker("b1_j_desc", "B1 base / J 越高", "b1_J", descending=True),
        _single_ranker(
            "b1_close_to_high_asc",
            "B1 base / 更贴近20日高点",
            "close_to_high_20d",
            descending=False,
        ),
        _single_ranker(
            "b1_ma_bias_asc",
            "B1 base / 20日均线回调",
            "ma_bias_20",
            descending=False,
        ),
        _single_ranker(
            "b1_disp_bias_asc",
            "B1 base / 成本线下回归",
            "disp_bias_20",
            descending=False,
        ),
        _single_ranker(
            "b1_intraday_pos_asc",
            "B1 base / 收盘靠低",
            "intraday_pos",
            descending=False,
        ),
    ]
    rankers.extend(
        [
            _make_pullback_ranker(
                price_weight=0.0,
                kbar_weight=0.0,
                bias_weight=3.0,
                close_pullback_weight=1.0,
                risk_weight=0.0,
                group="b1_base_combo",
                label_prefix="b1_base_pb3_cp1",
            ),
            _make_pullback_ranker(
                price_weight=0.0,
                kbar_weight=0.0,
                bias_weight=2.0,
                close_pullback_weight=0.5,
                risk_weight=0.0,
                group="b1_base_combo",
                label_prefix="b1_base_pb2_cp0p5",
            ),
            _make_pullback_ranker(
                price_weight=3.0,
                kbar_weight=0.5,
                bias_weight=0.0,
                close_pullback_weight=0.0,
                risk_weight=0.0,
                group="b1_base_combo",
                label_prefix="b1_base_p3_k0p5",
            ),
        ]
    )
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
            "id": "b1_base_all_candidates",
            "label": "B1 base 全部候选平均",
            "group": "b1_base",
            "weights": None,
            "components": None,
        },
        scenario="all_candidates",
        horizon=horizon,
        top_n=top_n,
        eligible_dates=eligible_dates,
        high_open_pct=high_open_pct,
    )


def _top_rows(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "by_exec_tradeoff": sorted(
            rows,
            key=lambda row: (row["exec_tradeoff"], row["exec_nav_end"], row["exec_edge_ret"]),
            reverse=True,
        )[:top_k],
        "by_exec_nav": sorted(rows, key=lambda row: row["exec_nav_end"], reverse=True)[:top_k],
        "by_low_pollution": sorted(
            rows,
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
    compact_path = output_dir / "compact.csv"
    daily_path = output_dir / "daily.csv"
    summary_path = output_dir / "summary.json"

    pl.DataFrame(compact_rows).write_csv(compact_path)
    if daily_frames:
        pl.concat(daily_frames, how="vertical").write_csv(daily_path)

    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
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
            "execution_lag_days": args.execution_lag_days,
            "base_mode": args.base_mode,
            "j_threshold": args.j_threshold,
            "high_open_pct": args.high_open_pct,
            "price_limit_tolerance": args.price_limit_tolerance,
            "amv_bull_trigger_pct": args.amv_bull_trigger_pct,
            "amv_bull_lookback_days": args.amv_bull_lookback_days,
            "amv_bear_trigger_1d_pct": args.amv_bear_trigger_1d_pct,
            "amv_effective_lag_days": args.amv_effective_lag_days,
        },
        "definition": {
            "b1_base": _base_definition(args.base_mode),
            "WL": "EMA(EMA(close, 10), 10)",
            "YL": "mean(MA14, MA28, MA57, MA114)",
            "J": "3*K - 2*D, K/D from 9-day RSV ewm_mean(com=2)",
        },
        "files": {
            "compact": _rel_path(compact_path, output_dir),
            "daily": _rel_path(daily_path, output_dir),
        },
        "results_by_horizon": results_by_horizon,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable-aware lab for original B1 base conditions")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--st-snapshot-date", default="2026-03-31")
    parser.add_argument("--mv-min", type=float, default=100.0)
    parser.add_argument("--amount-ma20-min", type=float, default=5e7)
    parser.add_argument("--horizons", type=_parse_int_list, default=[6])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--high-open-pct", type=float, default=0.098)
    parser.add_argument("--price-limit-tolerance", type=float, default=LIMIT_TOLERANCE)
    parser.add_argument("--base-mode", choices=["classic", "trend_only"], default="classic")
    parser.add_argument("--j-threshold", type=float, default=13.0)
    parser.add_argument("--amv-bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--amv-bull-lookback-days", type=int, default=2)
    parser.add_argument("--amv-bear-trigger-1d-pct", type=float, default=-2.3)
    parser.add_argument("--amv-effective-lag-days", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now()
    b1_indicators = _build_b1_indicators(args)
    raw_rankers = _build_rankers()

    results_by_horizon: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []

    for horizon in args.horizons:
        print(f"\nBuilding executable AMV pool for horizon={horizon}d, base_mode={args.base_mode}...")
        dataset_args = copy.copy(args)
        dataset_args.horizon = horizon
        df_pool = _build_dataset_for_horizon(dataset_args, horizon).join(
            b1_indicators, on=["date", "code"], how="left"
        )
        df_pool = add_combo_scores(df_pool, raw_rankers).with_columns(
            _base_b1_expr(args.base_mode, args.j_threshold).alias("b1_base")
        )
        b1_pool = df_pool.filter(pl.col("b1_base"))
        print(
            "B1 base candidates: rows={:,}, days={}, codes={}".format(
                b1_pool.height,
                b1_pool.select("date").n_unique(),
                b1_pool.select("code").n_unique(),
            )
        )

        horizon_results: list[dict[str, Any]] = []
        all_result, all_daily = _evaluate_all_candidates(
            b1_pool,
            horizon=horizon,
            top_n=args.top_n,
            high_open_pct=args.high_open_pct,
        )
        if all_result is not None and all_daily is not None:
            horizon_results.append(all_result)
            daily_frames.append(all_daily)

        for ranker in raw_rankers:
            evaluated, daily = evaluate_ranker(
                b1_pool,
                ranker,
                horizon=horizon,
                top_n=args.top_n,
                max_scan_rank=0,
                high_open_pct=args.high_open_pct,
            )
            horizon_results.extend(evaluated)
            daily_frames.extend(daily)

        valid_results = [result for result in horizon_results if "error" not in result]
        rows = [_compact_row(result) for result in valid_results]
        compact_rows.extend(rows)
        results_by_horizon[str(horizon)] = {
            "all_rows": rows,
            "top": _top_rows(rows, args.top_k),
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
    for row in results_by_horizon[target_horizon]["top"]["by_exec_tradeoff"][:10]:
        print(
            f"- {row['id']:<36} {row['scenario']:<28} "
            f"exec_nav={row['exec_nav_end'] * 100:+.2f}% "
            f"dd={row['exec_max_dd'] * 100:.2f}% "
            f"ctc_nav={row['ctc_nav_end'] * 100:+.2f}% "
            f"close_limit_days={row['close_limit_up_day_share'] * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
