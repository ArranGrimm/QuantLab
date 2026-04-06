from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import stats


IC_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "factor": pl.String,
    "IC_mean": pl.Float64,
    "IC_std": pl.Float64,
    "ICIR": pl.Float64,
    "t_stat": pl.Float64,
    "n_days": pl.Int64,
    "abs_ICIR": pl.Float64,
}

GROUP_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "group_key": pl.String,
    "group_name": pl.String,
    "n_factors": pl.Int64,
    "mean_abs_icir": pl.Float64,
    "max_abs_icir": pl.Float64,
    "top_factor": pl.String,
    "top_ic_mean": pl.Float64,
    "top_icir": pl.Float64,
    "top_abs_icir": pl.Float64,
}


def empty_ic_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=IC_SUMMARY_SCHEMA)


def build_ic_summary_frame(ic_results: dict[str, dict[str, float]]) -> pl.DataFrame:
    rows = []
    for factor_name, result in ic_results.items():
        rows.append(
            {
                "factor": factor_name,
                "IC_mean": round(float(result["ic_mean"]), 4),
                "IC_std": round(float(result["ic_std"]), 4),
                "ICIR": round(float(result["icir"]), 4),
                "t_stat": round(float(result["t_stat"]), 2),
                "n_days": int(result["n_days"]),
                "abs_ICIR": round(abs(float(result["icir"])), 4),
            }
        )

    if not rows:
        return empty_ic_summary_frame()

    return pl.DataFrame(rows, schema=IC_SUMMARY_SCHEMA).sort("abs_ICIR", descending=True)


def empty_group_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=GROUP_SUMMARY_SCHEMA)


def summarize_factor_groups(
    ic_results: dict[str, dict[str, float]],
    factor_groups: dict[str, Sequence[str]],
    group_labels: dict[str, str] | None = None,
) -> pl.DataFrame:
    group_labels = group_labels or {}
    rows = []

    for group_key, factor_cols in factor_groups.items():
        available_group_factors = [factor for factor in factor_cols if factor in ic_results]
        group_icirs = [
            float(ic_results[factor]["icir"])
            for factor in available_group_factors
            if ic_results[factor].get("icir") is not None
        ]
        group_abs_icirs = [abs(value) for value in group_icirs]

        top_factor = ""
        top_ic_mean = 0.0
        top_icir = 0.0
        top_abs_icir = 0.0
        if available_group_factors:
            top_factor = max(
                available_group_factors,
                key=lambda factor: abs(float(ic_results[factor]["icir"])),
            )
            top_ic_mean = float(ic_results[top_factor]["ic_mean"])
            top_icir = float(ic_results[top_factor]["icir"])
            top_abs_icir = abs(top_icir)

        rows.append(
            {
                "group_key": group_key,
                "group_name": group_labels.get(group_key, group_key),
                "n_factors": len(available_group_factors),
                "mean_abs_icir": float(sum(group_abs_icirs) / len(group_abs_icirs)) if group_abs_icirs else 0.0,
                "max_abs_icir": float(top_abs_icir),
                "top_factor": top_factor,
                "top_ic_mean": float(top_ic_mean),
                "top_icir": float(top_icir),
                "top_abs_icir": float(top_abs_icir),
            }
        )

    if not rows:
        return empty_group_summary_frame()

    return pl.DataFrame(rows, schema=GROUP_SUMMARY_SCHEMA).sort("mean_abs_icir", descending=True)


def extract_group_top_factor_cols(df_group_summary: pl.DataFrame) -> list[str]:
    if df_group_summary.height == 0:
        return []
    return [factor for factor in df_group_summary["top_factor"].to_list() if factor]


def build_daily_ic_frame(
    df: pl.DataFrame,
    factor_cols: Sequence[str],
    label: str,
    min_samples: int = 30,
) -> pl.DataFrame:
    available = [factor for factor in factor_cols if factor in df.columns]
    schema = {"date": pl.Date, **{factor: pl.Float64 for factor in available}}
    if not available:
        return pl.DataFrame(schema=schema)

    df_valid = df.filter(pl.col(label).is_not_null() & pl.col(label).is_not_nan())
    date_counts = df_valid.group_by("date").agg(pl.len().alias("n"))
    valid_dates = date_counts.filter(pl.col("n") >= min_samples)["date"]
    if valid_dates.is_empty():
        return pl.DataFrame(schema=schema)

    return (
        df_valid.filter(pl.col("date").is_in(valid_dates))
        .group_by("date")
        .agg([pl.corr(factor, label, method="spearman").alias(factor) for factor in available])
        .sort("date")
    )


def resolve_decay_factor_cols(
    decay_source: str,
    *,
    rotation_ic_summary: pl.DataFrame | None = None,
    alpha158_top1: pl.DataFrame | None = None,
    custom_factor_cols: Sequence[str] | None = None,
    rotation_top_n: int = 15,
) -> list[str]:
    normalized = decay_source.strip().lower()
    if normalized == "rotation":
        if rotation_ic_summary is None or rotation_ic_summary.height == 0:
            return []
        return rotation_ic_summary["factor"].head(rotation_top_n).to_list()

    if normalized == "alpha158_top1":
        if alpha158_top1 is None or alpha158_top1.height == 0:
            return []
        return extract_group_top_factor_cols(alpha158_top1)

    if normalized == "custom_list":
        return [str(factor).strip() for factor in (custom_factor_cols or []) if str(factor).strip()]

    raise ValueError(
        f"Unsupported decay source: {decay_source}. "
        "Expected one of: rotation, alpha158_top1, custom_list"
    )


def compute_factor_decay(
    df: pl.DataFrame,
    factor_cols: Sequence[str],
    *,
    horizons: Sequence[str] = ("fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"),
    min_samples: int = 30,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[int, float]]:
    factor_cols = [factor for factor in factor_cols if factor in df.columns]
    if not factor_cols:
        return {}, {}

    select_cols = list(dict.fromkeys([*factor_cols, *horizons, "date"]))
    df_sub = df.select(select_cols).sort("date")

    dates_arr = df_sub["date"].to_numpy()
    factor_np = {factor: df_sub[factor].to_numpy() for factor in factor_cols}
    ret_np = {horizon: df_sub[horizon].to_numpy() for horizon in horizons}

    unique_dates = np.unique(dates_arr)
    unique_dates.sort()
    d_start = np.searchsorted(dates_arr, unique_dates, side="left")
    d_end = np.searchsorted(dates_arr, unique_dates, side="right")

    ic_lists = {horizon: {factor: [] for factor in factor_cols} for horizon in horizons}

    for date_idx in range(len(unique_dates)):
        start_idx, end_idx = d_start[date_idx], d_end[date_idx]
        if end_idx - start_idx < min_samples:
            continue

        for horizon in horizons:
            ret_values = ret_np[horizon][start_idx:end_idx]
            valid_ret = np.isfinite(ret_values)
            if valid_ret.sum() < min_samples:
                continue

            for factor in factor_cols:
                factor_values = factor_np[factor][start_idx:end_idx]
                mask = np.isfinite(factor_values) & valid_ret
                if mask.sum() < min_samples:
                    continue

                corr, _ = stats.spearmanr(factor_values[mask], ret_values[mask])
                if np.isfinite(corr):
                    ic_lists[horizon][factor].append(float(corr))

    decay_summary: dict[str, dict[str, dict[str, float]]] = {}
    avg_abs_icir: dict[int, float] = {}
    for horizon in horizons:
        horizon_summary: dict[str, dict[str, float]] = {}
        abs_icirs = []
        for factor in factor_cols:
            values = ic_lists[horizon][factor]
            if values:
                arr = np.asarray(values, dtype=float)
                ic_mean = float(np.mean(arr))
                ic_std = float(np.std(arr))
                icir = float(ic_mean / max(ic_std, 1e-8))
            else:
                ic_mean = 0.0
                icir = 0.0

            horizon_summary[factor] = {
                "ic_mean": ic_mean,
                "icir": icir,
            }
            abs_icirs.append(abs(icir))

        decay_summary[horizon] = horizon_summary
        avg_abs_icir[_extract_horizon_days(horizon)] = float(np.mean(abs_icirs)) if abs_icirs else 0.0

    return decay_summary, avg_abs_icir


def _extract_horizon_days(horizon: str) -> int:
    digits = "".join(ch for ch in horizon if ch.isdigit())
    return int(digits) if digits else 0
