from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from scripts.amv_bull_pool_export_signals import DEFAULT_QMT_DB, ROOT, _git_commit
from utils import load_daily_data_full


DEFAULT_CANDIDATE_PATH = (
    ROOT / "artifacts" / "b3_tdx_signals" / "20260517_184650" / "raw_candidate_signals.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "b3_candidate_ranking_lab"
HORIZONS = (3, 6, 10, 20)

FIELD_SPECS: dict[str, str] = {
    "prev_trigger_ret_pct": "触发K涨幅",
    "prev_trigger_volume_ratio": "触发K放量倍数",
    "prev_deviation_distance": "触发K开盘偏离短/长线",
    "range_amp_30": "30日区间振幅",
    "rw_dif_pct": "B1周线DIF强度",
    "rw_hist": "B1周线红柱",
    "market_cap_100m": "流通市值",
    "amount_ma20": "20日成交额",
    "open_vs_trigger_low_pct": "执行开盘距触发K低点",
    "open_vs_white_line_pct": "执行开盘距白线",
    "open_gap_from_trigger_close_pct": "执行开盘相对触发收盘缺口",
    "white_yellow_spread_pct": "白黄线距离",
    "trigger_close_vs_yellow_pct": "触发收盘距黄线",
}


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_price_frame(qmt_db: Path, start_date: str, end_date: str) -> pl.DataFrame:
    conn = duckdb.connect(str(qmt_db), read_only=True)
    try:
        price = (
            load_daily_data_full(conn)
            .filter(pl.col("date") >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col("date") <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d"))
            .select(["date", "code", "open_adj", "high_adj", "low_adj", "close_adj"])
            .collect()
            .sort(["code", "date"])
        )
    finally:
        conn.close()

    trading_dates = price.select("date").unique().sort("date").with_row_index("td_idx")
    return price.join(trading_dates, on="date", how="left")


def add_execution_and_path_labels(
    candidates: pl.DataFrame,
    price: pl.DataFrame,
    *,
    backtest_start_date: str,
    top_n: int,
) -> pl.DataFrame:
    trading_dates = price.select(["date", "td_idx"]).unique().sort("date")
    next_dates = (
        trading_dates.with_columns(pl.col("date").shift(-1).alias("execution_date"))
        .drop_nulls("execution_date")
        .select([pl.col("date").alias("signal_date"), "execution_date"])
    )
    exec_price = price.select(
        [
            "code",
            pl.col("date").alias("execution_date"),
            pl.col("td_idx").alias("execution_idx"),
            pl.col("open_adj").alias("entry_open"),
            pl.col("high_adj").alias("execution_high"),
            pl.col("low_adj").alias("execution_low"),
            pl.col("close_adj").alias("execution_close"),
        ]
    )

    df = (
        candidates.join(next_dates, on="signal_date", how="inner")
        .join(exec_price, on=["code", "execution_date"], how="inner")
        .filter(pl.col("execution_date") >= pl.lit(backtest_start_date).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    max_horizon = max(HORIZONS)
    for offset in range(max_horizon):
        future = price.select(
            [
                "code",
                (pl.col("td_idx") - offset).alias("execution_idx"),
                pl.col("high_adj").alias(f"high_o{offset}"),
                pl.col("low_adj").alias(f"low_o{offset}"),
                pl.col("close_adj").alias(f"close_o{offset}"),
            ]
        )
        df = df.join(future, on=["code", "execution_idx"], how="left")

    label_exprs: list[pl.Expr] = [
        ((pl.col("entry_open") / pl.col("trigger_low") - 1.0) * 100.0).alias(
            "open_vs_trigger_low_pct"
        ),
        ((pl.col("entry_open") / pl.col("white_line") - 1.0) * 100.0).alias("open_vs_white_line_pct"),
        ((pl.col("entry_open") / pl.col("trigger_close") - 1.0) * 100.0).alias(
            "open_gap_from_trigger_close_pct"
        ),
        ((pl.col("white_line") / pl.col("yellow_line") - 1.0) * 100.0).alias(
            "white_yellow_spread_pct"
        ),
        ((pl.col("trigger_close") / pl.col("yellow_line") - 1.0) * 100.0).alias(
            "trigger_close_vs_yellow_pct"
        ),
    ]

    for horizon in HORIZONS:
        high_cols = [pl.col(f"high_o{offset}") for offset in range(horizon)]
        low_cols = [pl.col(f"low_o{offset}") for offset in range(horizon)]
        label_exprs.extend(
            [
                (pl.col(f"close_o{horizon - 1}") / pl.col("entry_open") - 1.0).alias(
                    f"fwd_ret_{horizon}td"
                ),
                (pl.max_horizontal(high_cols) / pl.col("entry_open") - 1.0).alias(
                    f"fwd_mfe_{horizon}td"
                ),
                (pl.min_horizontal(low_cols) / pl.col("entry_open") - 1.0).alias(
                    f"fwd_mae_{horizon}td"
                ),
            ]
        )

    keep_cols = [
        "signal_date",
        "execution_date",
        "code",
        "trigger_type",
        "trigger_low",
        "trigger_high",
        "trigger_close",
        "entry_open",
        "prev_trigger_ret_pct",
        "prev_trigger_volume_ratio",
        "prev_deviation_distance",
        "range_amp_30",
        "white_line",
        "yellow_line",
        "rw_dif_pct",
        "rw_hist",
        "rw_dif",
        "market_cap_100m",
        "amount_ma20",
    ]
    derived_cols = [
        "open_vs_trigger_low_pct",
        "open_vs_white_line_pct",
        "open_gap_from_trigger_close_pct",
        "white_yellow_spread_pct",
        "trigger_close_vs_yellow_pct",
    ]
    label_cols = [
        f"{kind}_{horizon}td"
        for horizon in HORIZONS
        for kind in ("fwd_ret", "fwd_mfe", "fwd_mae")
    ]

    return (
        df.with_columns(label_exprs)
        .with_columns(
            [
                (pl.col("fwd_mae_3td") <= -0.03).alias("fast_fail_proxy"),
                (pl.col("fwd_mfe_20td") >= 0.12).alias("mfe20_ge12"),
                pl.col("fwd_ret_6td").gt(0).alias("ret6_positive"),
            ]
        )
        .select(keep_cols + derived_cols + label_cols + ["fast_fail_proxy", "mfe20_ge12", "ret6_positive"])
        .drop_nulls(["fwd_ret_6td", "fwd_mfe_20td", "fwd_mae_3td"])
        .sort(["signal_date", "code"])
    )


def metric_row(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "rows": 0,
            "days": 0,
            "ret_6td_mean_pct": None,
            "ret_10td_mean_pct": None,
            "mfe_20td_mean_pct": None,
            "mae_3td_mean_pct": None,
            "fast_fail_proxy_rate_pct": None,
            "mfe20_ge12_rate_pct": None,
            "ret6_positive_rate_pct": None,
        }
    return {
        "rows": df.height,
        "days": df.select("signal_date").n_unique(),
        "ret_6td_mean_pct": df["fwd_ret_6td"].mean() * 100.0,
        "ret_10td_mean_pct": df["fwd_ret_10td"].mean() * 100.0,
        "mfe_20td_mean_pct": df["fwd_mfe_20td"].mean() * 100.0,
        "mae_3td_mean_pct": df["fwd_mae_3td"].mean() * 100.0,
        "fast_fail_proxy_rate_pct": df["fast_fail_proxy"].mean() * 100.0,
        "mfe20_ge12_rate_pct": df["mfe20_ge12"].mean() * 100.0,
        "ret6_positive_rate_pct": df["ret6_positive"].mean() * 100.0,
    }


def evaluate_sort_fields(df: pl.DataFrame, top_n: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    base = metric_row(df)
    rows.append(
        {
            "field": "all_candidates",
            "field_label": "全部候选",
            "direction": "all",
            "direction_label": "不排序",
            **base,
            "ret_6td_edge_vs_all_pct": 0.0,
        }
    )
    base_ret6 = base["ret_6td_mean_pct"] or 0.0

    for field, label in FIELD_SPECS.items():
        for descending, direction_label in [(True, "降序Top3"), (False, "升序Top3")]:
            selected = (
                df.filter(pl.col(field).is_not_null() & pl.col(field).is_finite())
                .sort(["signal_date", field, "code"], descending=[False, descending, False])
                .group_by("signal_date", maintain_order=True)
                .head(top_n)
            )
            metrics = metric_row(selected)
            rows.append(
                {
                    "field": field,
                    "field_label": label,
                    "direction": "desc" if descending else "asc",
                    "direction_label": direction_label,
                    **metrics,
                    "ret_6td_edge_vs_all_pct": (
                        metrics["ret_6td_mean_pct"] - base_ret6
                        if metrics["ret_6td_mean_pct"] is not None
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows).sort(
        ["ret_6td_mean_pct", "mfe20_ge12_rate_pct"],
        descending=[True, True],
        nulls_last=True,
    )


def evaluate_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    bucket_defs = [
        ("Q1低", None, 0.25),
        ("Q2", 0.25, 0.50),
        ("Q3", 0.50, 0.75),
        ("Q4高", 0.75, None),
    ]
    for field, label in FIELD_SPECS.items():
        values = df.filter(pl.col(field).is_not_null() & pl.col(field).is_finite())[field]
        if values.is_empty():
            continue
        qs = {
            0.25: values.quantile(0.25, interpolation="nearest"),
            0.50: values.quantile(0.50, interpolation="nearest"),
            0.75: values.quantile(0.75, interpolation="nearest"),
        }
        for bucket, low_q, high_q in bucket_defs:
            expr = pl.col(field).is_not_null() & pl.col(field).is_finite()
            low_value = qs[low_q] if low_q is not None else None
            high_value = qs[high_q] if high_q is not None else None
            if low_value is not None:
                expr = expr & (pl.col(field) > low_value)
            if high_value is not None:
                expr = expr & (pl.col(field) <= high_value)
            part = df.filter(expr)
            metrics = metric_row(part)
            rows.append(
                {
                    "field": field,
                    "field_label": label,
                    "bucket": bucket,
                    "low_value": low_value,
                    "high_value": high_value,
                    **metrics,
                }
            )
    return pl.DataFrame(rows)


def field_quantiles(df: pl.DataFrame, field: str) -> dict[float, float]:
    values = df.filter(pl.col(field).is_not_null() & pl.col(field).is_finite())[field]
    return {
        0.25: values.quantile(0.25, interpolation="nearest"),
        0.50: values.quantile(0.50, interpolation="nearest"),
        0.75: values.quantile(0.75, interpolation="nearest"),
    }


def evaluate_filter_rules(df: pl.DataFrame) -> pl.DataFrame:
    q = {field: field_quantiles(df, field) for field in FIELD_SPECS}
    rule_exprs: list[tuple[str, str, pl.Expr]] = [
        (
            "drop_range_amp_q4",
            "剔除30日振幅最高25%",
            pl.col("range_amp_30") <= q["range_amp_30"][0.75],
        ),
        (
            "drop_trigger_ret_q4",
            "剔除触发K涨幅最高25%",
            pl.col("prev_trigger_ret_pct") <= q["prev_trigger_ret_pct"][0.75],
        ),
        (
            "drop_open_vs_white_q4",
            "剔除执行开盘距白线最高25%",
            pl.col("open_vs_white_line_pct") <= q["open_vs_white_line_pct"][0.75],
        ),
        (
            "drop_trigger_vs_yellow_q4",
            "剔除触发收盘距黄线最高25%",
            pl.col("trigger_close_vs_yellow_pct") <= q["trigger_close_vs_yellow_pct"][0.75],
        ),
        (
            "rw_dif_q1",
            "仅B1周线DIF最低25%",
            pl.col("rw_dif_pct") <= q["rw_dif_pct"][0.25],
        ),
        (
            "rw_hist_q2",
            "仅B1周线红柱Q2",
            (pl.col("rw_hist") > q["rw_hist"][0.25]) & (pl.col("rw_hist") <= q["rw_hist"][0.50]),
        ),
        (
            "range_amp_q2",
            "仅30日振幅Q2",
            (pl.col("range_amp_30") > q["range_amp_30"][0.25])
            & (pl.col("range_amp_30") <= q["range_amp_30"][0.50]),
        ),
        (
            "trigger_ret_q1_q2",
            "仅触发K涨幅低/中50%",
            pl.col("prev_trigger_ret_pct") <= q["prev_trigger_ret_pct"][0.50],
        ),
        (
            "structure_mid",
            "开盘距白线Q2-Q3",
            (pl.col("open_vs_white_line_pct") > q["open_vs_white_line_pct"][0.25])
            & (pl.col("open_vs_white_line_pct") <= q["open_vs_white_line_pct"][0.75]),
        ),
        (
            "drop_overheat_combo",
            "剔除振幅/触发涨幅/白线距离最高25%",
            (pl.col("range_amp_30") <= q["range_amp_30"][0.75])
            & (pl.col("prev_trigger_ret_pct") <= q["prev_trigger_ret_pct"][0.75])
            & (pl.col("open_vs_white_line_pct") <= q["open_vs_white_line_pct"][0.75]),
        ),
        (
            "mid_momentum_combo",
            "触发涨幅Q2-Q3且振幅Q2-Q3",
            (pl.col("prev_trigger_ret_pct") > q["prev_trigger_ret_pct"][0.25])
            & (pl.col("prev_trigger_ret_pct") <= q["prev_trigger_ret_pct"][0.75])
            & (pl.col("range_amp_30") > q["range_amp_30"][0.25])
            & (pl.col("range_amp_30") <= q["range_amp_30"][0.75]),
        ),
    ]

    base = metric_row(df)
    base_ret6 = base["ret_6td_mean_pct"] or 0.0
    rows = [
        {
            "rule": "all_candidates",
            "rule_label": "全部候选",
            **base,
            "ret_6td_edge_vs_all_pct": 0.0,
        }
    ]
    for rule, label, expr in rule_exprs:
        part = df.filter(expr)
        metrics = metric_row(part)
        rows.append(
            {
                "rule": rule,
                "rule_label": label,
                **metrics,
                "ret_6td_edge_vs_all_pct": (
                    metrics["ret_6td_mean_pct"] - base_ret6
                    if metrics["ret_6td_mean_pct"] is not None
                    else None
                ),
            }
        )
    return pl.DataFrame(rows).sort(
        ["ret_6td_mean_pct", "fast_fail_proxy_rate_pct"],
        descending=[True, False],
        nulls_last=True,
    )


def top_records(df: pl.DataFrame, n: int = 8) -> list[dict[str, Any]]:
    keep = [
        "field",
        "field_label",
        "direction_label",
        "rows",
        "days",
        "ret_6td_mean_pct",
        "ret_6td_edge_vs_all_pct",
        "mfe_20td_mean_pct",
        "mae_3td_mean_pct",
        "fast_fail_proxy_rate_pct",
        "mfe20_ge12_rate_pct",
    ]
    return df.filter(pl.col("field") != "all_candidates").head(n).select(keep).to_dicts()


def top_filter_records(df: pl.DataFrame, n: int = 8) -> list[dict[str, Any]]:
    keep = [
        "rule",
        "rule_label",
        "rows",
        "days",
        "ret_6td_mean_pct",
        "ret_6td_edge_vs_all_pct",
        "mfe_20td_mean_pct",
        "mae_3td_mean_pct",
        "fast_fail_proxy_rate_pct",
        "mfe20_ge12_rate_pct",
    ]
    return df.filter(pl.col("rule") != "all_candidates").head(n).select(keep).to_dicts()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose B3 candidate ranking fields")
    parser.add_argument("--qmt-db", type=Path, default=DEFAULT_QMT_DB)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2026-05-10")
    parser.add_argument("--backtest-start-date", default="2021-01-01")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading B3 candidates: {args.candidate_path}")
    candidates = pl.read_csv(args.candidate_path, try_parse_dates=True)
    print(f"Candidates: {candidates.height:,} rows")

    print("Loading price frame and computing forward path labels...")
    price = load_price_frame(args.qmt_db, args.start_date, args.end_date)
    enriched = add_execution_and_path_labels(
        candidates,
        price,
        backtest_start_date=args.backtest_start_date,
        top_n=args.top_n,
    )

    sort_summary = evaluate_sort_fields(enriched, args.top_n)
    quantile_summary = evaluate_quantiles(enriched)
    filter_summary = evaluate_filter_rules(enriched)

    enriched_path = output_dir / "candidate_labels.parquet"
    sort_path = output_dir / "sort_field_summary.csv"
    quantile_path = output_dir / "quantile_summary.csv"
    filter_path = output_dir / "filter_rule_summary.csv"
    summary_path = output_dir / "summary.json"

    enriched.write_parquet(enriched_path)
    sort_summary.write_csv(sort_path)
    quantile_summary.write_csv(quantile_path)
    filter_summary.write_csv(filter_path)

    base_metrics = sort_summary.filter(pl.col("field") == "all_candidates").to_dicts()[0]
    summary = {
        "artifact_dir": str(output_dir),
        "candidate_path": str(args.candidate_path),
        "rows": enriched.height,
        "days": enriched.select("signal_date").n_unique(),
        "date_min": str(enriched["signal_date"].min()),
        "date_max": str(enriched["signal_date"].max()),
        "top_n": args.top_n,
        "backtest_start_date": args.backtest_start_date,
        "base_metrics": base_metrics,
        "top_sort_fields": top_records(sort_summary),
        "top_filter_rules": top_filter_records(filter_summary),
        "files": {
            "candidate_labels": enriched_path.name,
            "sort_field_summary": sort_path.name,
            "quantile_summary": quantile_path.name,
            "filter_rule_summary": filter_path.name,
        },
        "git_commit": _git_commit(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
