from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import datetime

import polars as pl

from manifests.b1_textbook_cases import B1_TEXTBOOK_CASES

from .alpha158_factors import calc_alpha158_factors
from .b1_factors_opt import calc_b1_factors_wmacd
from .rotation_factors import calc_rotation_factors


B1_MINING_CORE_FEATURE_COLS: list[str] = [
    "Bias_C_WL",
    "Bias_C_YL",
    "Bias_WL_YL",
    "rw_dif_pct",
    "rw_hist",
    "rm_hist",
    "J",
    "bad_k_count",
    "vol_shrink_20",
    "vol_shrink_40",
    "red_green_ratio_20",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "retrace_ratio_20",
    "days_since_key_k",
]

B1_MINING_SECOND_BATCH_FEATURE_COLS: list[str] = [
    "trigger_recent_10",
    "key_k_recent_20",
    "plry_cluster_recent_10",
    "days_since_key_k_inv",
    "range_pct",
    "body_to_range",
    "close_pos_in_bar",
    "gap_from_prev_close_pct",
    "vol_to_prev_vol",
    "vol_to_avg40",
    "vol_shrink_20_delta_5",
    "red_green_ratio_delta_5",
    "rw_hist_delta_5",
    "rm_hist_delta_5",
    "bias_wl_yl_delta_5",
    "close_above_yl_pct_5",
    "close_above_wl_pct_5",
]

B1_MINING_FEATURE_COLS: list[str] = [
    *B1_MINING_CORE_FEATURE_COLS,
    *B1_MINING_SECOND_BATCH_FEATURE_COLS,
]

B1_SELECTED_FEATURE_COLS: list[str] = [
    "Bias_WL_YL",
    "Bias_C_YL",
    "rw_dif_pct",
    "rw_hist",
    "rm_hist",
    "range_pct",
    "vol_shrink_20",
    "vol_to_avg40",
    "body_pct",
    "vol_shrink_40",
    "rw_hist_delta_5",
    "rm_hist_delta_5",
    "close_above_yl_pct_5",
    "key_k_recent_20",
]

# 基于 2026-04-14 的 B1 Lab 输出，将 B1 骨架与 Rotation/KBAR 强因子做受控融合。
B1_SELECTED_ROTATION_HYBRID_V1_FEATURE_COLS: list[str] = [
    "Bias_WL_YL",
    "Bias_C_YL",
    "rw_dif_pct",
    "rw_hist",
    "rm_hist",
    "body_pct",
    "vol_shrink_40",
    "atr_14_pct",
    "KLEN",
    "vol_60d",
    "turnover_rate",
    "KUP",
]

# 与 manifests/rotation_feature_sets.py::CORE_12_FEATURES 保持一致。
B1_ROTATION_CORE12_FEATURE_COLS: list[str] = [
    "ret_max_5d",
    "vol_60d",
    "turnover_rate",
    "atr_14_pct",
    "amplitude",
    "intraday_ret_ma5",
    "disp_bias_20",
    "high_open_pct",
    "vol_std_20d",
    "abnormal_vol",
    "intraday_pos",
    "vol_price_corr_20d",
]

B1_ALPHA158_KBAR_FEATURE_COLS: list[str] = [
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
]

B1_ROTATION_CORE12_KBAR_FEATURE_COLS: list[str] = [
    *B1_ROTATION_CORE12_FEATURE_COLS,
    *B1_ALPHA158_KBAR_FEATURE_COLS,
]

B1_TEXTBOOK_SCORE_FEATURE_COLS: list[str] = [
    "Bias_C_WL",
    "Bias_C_YL",
    "Bias_WL_YL",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "retrace_ratio_20",
    "vol_shrink_40",
    "red_green_ratio_20",
    "rw_hist",
    "rm_hist",
    "trigger_recent_10",
    "key_k_recent_20",
    "days_since_key_k_inv",
]

# V3 textbook 特征清单 (基于 2026-04-18 Cohen's d + 共线诊断, 详见
# experiments/b1-next-phase.md). 10 个 similarity 特征均经过 |d| 阈值 0.35 + 共线
# Pearson 0.85 双重筛选, 每个共线簇只保留 |d| 最大的代表。
B1_TEXTBOOK_SCORE_FEATURE_COLS_V3: list[str] = [
    "turnover_rate",          # |d|=0.86  rotation_core12   NEW (单因子最强)
    "lower_shadow_pct",       # |d|=0.65  price_structure   KEEP
    "plry_cluster_recent_10", # |d|=0.62  trigger_context   NEW
    "close_pos_in_bar",       # |d|=0.60  price_structure   NEW (簇 2 代表)
    "KLOW",                   # |d|=0.47  alpha158_kbar     NEW
    "K",                      # |d|=0.42  kdj               NEW (KDJ K, 超跌)
    "Bias_C_YL",              # |d|=0.42  trend_strength    KEEP
    "rw_dif_pct",             # |d|=0.40  weekly_momentum   NEW
    "body_pct",               # |d|=0.39  price_structure   KEEP
    "key_k_recent_20",        # |d|=0.36  trigger_context   KEEP
]

# V3 hard rules: 在 case 上方差为 0 的特征改成 AND 进 is_textbook_b1
# (而不是当软相似度用)。每个元组是 (column_name, expected_int_value)。
B1_TEXTBOOK_HARD_RULES_V3: list[tuple[str, int]] = [
    ("bad_k_count", 0),         # case 全部 0, |d|=0.57
    ("trigger_recent_10", 1),   # case 全部 1, |d|=0.45 (从 v1 软相似度升级)
]

B1_TEXTBOOK_COMPONENT_FEATURE_COLS: dict[str, tuple[str, ...]] = {
    "trend": (
        "Bias_C_WL",
        "Bias_C_YL",
        "Bias_WL_YL",
        "rw_hist",
        "rm_hist",
    ),
    "kbar": (
        "body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "retrace_ratio_20",
    ),
    "volume": (
        "vol_shrink_40",
        "red_green_ratio_20",
    ),
    "trigger": (
        "trigger_recent_10",
        "key_k_recent_20",
        "days_since_key_k_inv",
    ),
}

B1_TEXTBOOK_RULE_COLS: list[str] = [
    "TRIGGER",
    "KEY_K_EXIST",
    "GOOD28",
    "MAX28_OK",
    "YANGYIN_OK",
    "seed_mid",
    "seed_strict",
]

B1_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend_strength": (
        "Bias_C_WL",
        "Bias_C_YL",
        "Bias_WL_YL",
        "bias_wl_yl_delta_5",
        "close_above_yl_pct_5",
        "close_above_wl_pct_5",
    ),
    "weekly_momentum": (
        "rw_dif_pct",
        "rw_hist",
        "rm_hist",
        "rw_hist_delta_5",
        "rm_hist_delta_5",
    ),
    "trigger_context": (
        "J",
        "bad_k_count",
        "days_since_key_k",
        "trigger_recent_10",
        "key_k_recent_20",
        "plry_cluster_recent_10",
        "days_since_key_k_inv",
    ),
    "price_structure": (
        "body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "retrace_ratio_20",
        "range_pct",
        "body_to_range",
        "close_pos_in_bar",
        "gap_from_prev_close_pct",
    ),
    "volume_structure": (
        "vol_shrink_20",
        "vol_shrink_40",
        "red_green_ratio_20",
        "vol_to_prev_vol",
        "vol_to_avg40",
        "vol_shrink_20_delta_5",
        "red_green_ratio_delta_5",
    ),
    "rotation_core12": tuple(B1_ROTATION_CORE12_FEATURE_COLS),
    "alpha158_kbar": tuple(B1_ALPHA158_KBAR_FEATURE_COLS),
    "kdj": ("K", "D", "J"),
}

B1_FEATURE_GROUP_LABELS: dict[str, str] = {
    "trend_strength": "趋势强度",
    "weekly_momentum": "周月动能",
    "trigger_context": "触发上下文",
    "price_structure": "价格结构",
    "volume_structure": "量价结构",
    "rotation_core12": "Rotation Core12",
    "alpha158_kbar": "Alpha158 KBAR",
    "kdj": "KDJ",
}

B1_FEATURE_TO_GROUP: dict[str, str] = {
    feature_name: group_key
    for group_key, feature_cols in B1_FEATURE_GROUPS.items()
    for feature_name in feature_cols
}

B1_FEATURE_SET_REGISTRY: dict[str, tuple[str, ...]] = {
    "core": tuple(B1_MINING_CORE_FEATURE_COLS),
    "candidate": tuple(B1_MINING_FEATURE_COLS),
    "selected": tuple(B1_SELECTED_FEATURE_COLS),
    "selected_rotation_hybrid_v1": tuple(B1_SELECTED_ROTATION_HYBRID_V1_FEATURE_COLS),
    "rotation_core12_kbar": tuple(B1_ROTATION_CORE12_KBAR_FEATURE_COLS),
}


def resolve_b1_feature_set(name: str = "selected") -> tuple[str, ...]:
    try:
        return B1_FEATURE_SET_REGISTRY[name]
    except KeyError as exc:
        valid_names = ", ".join(B1_FEATURE_SET_REGISTRY)
        raise ValueError(
            f"Unsupported B1 feature set: {name}. Expected one of: {valid_names}"
        ) from exc


def describe_b1_feature_set(name: str = "selected") -> str:
    feature_cols = resolve_b1_feature_set(name)
    return f"name={name}, feature_count={len(feature_cols)}, features={', '.join(feature_cols)}"


def b1_feature_set_requires_rotation_kbar(name: str = "selected") -> bool:
    feature_cols = resolve_b1_feature_set(name)
    return any(
        feature_col in B1_ROTATION_CORE12_KBAR_FEATURE_COLS
        for feature_col in feature_cols
    )


def _empty_textbook_columns() -> list[pl.Expr]:
    """v1/v3 共用的占位列, 避免 case 为空时下游列缺失。"""
    return [
        pl.lit(0.0).alias("textbook_trend_score"),
        pl.lit(0.0).alias("textbook_kbar_score"),
        pl.lit(0.0).alias("textbook_volume_score"),
        pl.lit(0.0).alias("textbook_trigger_score"),
        pl.lit(0.0).alias("textbook_similarity_score"),
        pl.lit(0.0).alias("textbook_rule_score"),
        pl.lit(0.0).alias("textbook_b1_score"),
        pl.lit(0.65).alias("textbook_b1_threshold"),
        pl.lit(False).alias("is_textbook_b1"),
    ]


def _build_similarity_exprs(
    case_rows: pl.DataFrame, feature_cols: Sequence[str]
) -> tuple[list[pl.Expr], list[str], dict[str, str]]:
    """对每个特征用 case median 中心 + IQR/极差/35%|median| 的最大值做 scale,
    返回 (sim_exprs, sim_col_names, feature -> sim_col_name)。"""
    sim_exprs: list[pl.Expr] = []
    sim_col_names: list[str] = []
    feature_sim_col_names: dict[str, str] = {}
    for feature_col in feature_cols:
        if feature_col not in case_rows.columns:
            continue
        case_series = case_rows[feature_col].drop_nulls().drop_nans()
        if case_series.is_empty():
            continue
        median = float(case_series.median())
        q1 = float(case_series.quantile(0.25, interpolation="linear"))
        q3 = float(case_series.quantile(0.75, interpolation="linear"))
        case_min = float(case_series.min())
        case_max = float(case_series.max())
        scale = max(
            (q3 - q1) * 2.0,
            (case_max - case_min),
            abs(median) * 0.35,
            1e-4,
        )
        sim_col_name = f"_textbook_sim_{feature_col}"
        sim_col_names.append(sim_col_name)
        feature_sim_col_names[feature_col] = sim_col_name
        sim_exprs.append(
            (
                1.0
                - (pl.col(feature_col) - pl.lit(median)).abs() / pl.lit(scale)
            )
            .clip(0.0, 1.0)
            .alias(sim_col_name)
        )
    return sim_exprs, sim_col_names, feature_sim_col_names


def _resolve_textbook_threshold(
    case_rows: pl.DataFrame, df_scored: pl.DataFrame, default: float = 0.65
) -> float:
    case_score_series = case_rows.join(
        df_scored.select(["code", "date", "textbook_b1_score"]),
        on=["code", "date"],
        how="left",
    )["textbook_b1_score"].drop_nulls().drop_nans()
    if case_score_series.is_empty():
        return default
    return max(
        min(float(case_score_series.quantile(0.20, interpolation="linear")), 0.80),
        0.55,
    )


def _score_textbook_v1(
    df_with_case_flag: pl.DataFrame, case_rows: pl.DataFrame
) -> pl.DataFrame:
    sim_exprs, sim_col_names, feature_sim_col_names = _build_similarity_exprs(
        case_rows, B1_TEXTBOOK_SCORE_FEATURE_COLS
    )
    df_scored = df_with_case_flag
    if sim_exprs:
        df_scored = df_scored.with_columns(sim_exprs)

    rule_exprs = [
        pl.col(rule_col).cast(pl.Float64)
        for rule_col in B1_TEXTBOOK_RULE_COLS
        if rule_col in df_scored.columns
    ]
    similarity_expr = (
        pl.mean_horizontal(sim_col_names) if sim_col_names else pl.lit(0.0)
    )
    rule_expr = pl.mean_horizontal(rule_exprs) if rule_exprs else pl.lit(0.0)

    component_exprs: list[pl.Expr] = []
    for component_name, feature_cols in B1_TEXTBOOK_COMPONENT_FEATURE_COLS.items():
        component_score_col = f"textbook_{component_name}_score"
        component_sim_cols = [
            feature_sim_col_names[feature_col]
            for feature_col in feature_cols
            if feature_col in feature_sim_col_names
        ]
        component_expr = (
            pl.mean_horizontal(component_sim_cols)
            if component_sim_cols
            else pl.lit(0.0)
        )
        if component_name == "trigger":
            component_expr = 0.7 * component_expr + 0.3 * rule_expr
        component_exprs.append(
            component_expr.fill_nan(0.0).alias(component_score_col)
        )

    df_scored = df_scored.with_columns(
        [rule_expr.fill_nan(0.0).alias("textbook_rule_score"), *component_exprs]
    ).with_columns(
        [
            similarity_expr.fill_nan(0.0).alias("textbook_similarity_score"),
            (
                0.30 * pl.col("textbook_trend_score")
                + 0.25 * pl.col("textbook_kbar_score")
                + 0.20 * pl.col("textbook_volume_score")
                + 0.25 * pl.col("textbook_trigger_score")
            ).alias("textbook_b1_score"),
        ]
    )

    threshold = _resolve_textbook_threshold(case_rows, df_scored)
    df_scored = df_scored.with_columns(
        [
            pl.lit(threshold).alias("textbook_b1_threshold"),
            (pl.col("textbook_b1_score") >= pl.lit(threshold)).alias("is_textbook_b1"),
        ]
    )
    if sim_col_names:
        df_scored = df_scored.drop(sim_col_names)
    return df_scored


def _score_textbook_v3(
    df_with_case_flag: pl.DataFrame, case_rows: pl.DataFrame
) -> pl.DataFrame:
    """V3: 等权 mean_horizontal 10 个 similarity, 不再分 component;
    `bad_k_count == 0` 与 `trigger_recent_10 == 1` 升为 hard rule, AND 进 is_textbook_b1。
    保留 v1 列名 (textbook_*_score) 以维持下游兼容。"""
    sim_exprs, sim_col_names, _ = _build_similarity_exprs(
        case_rows, B1_TEXTBOOK_SCORE_FEATURE_COLS_V3
    )
    df_scored = df_with_case_flag
    if sim_exprs:
        df_scored = df_scored.with_columns(sim_exprs)

    similarity_expr = (
        pl.mean_horizontal(sim_col_names) if sim_col_names else pl.lit(0.0)
    )
    df_scored = df_scored.with_columns(
        similarity_expr.fill_nan(0.0).alias("textbook_b1_score")
    )

    threshold = _resolve_textbook_threshold(case_rows, df_scored)

    hard_rule_expr: pl.Expr = pl.lit(True)
    missing_rules: list[str] = []
    for rule_col, expected_value in B1_TEXTBOOK_HARD_RULES_V3:
        if rule_col not in df_scored.columns:
            missing_rules.append(rule_col)
            continue
        hard_rule_expr = hard_rule_expr & (
            pl.col(rule_col) == pl.lit(expected_value)
        )
    if missing_rules:
        warnings.warn(
            f"[textbook v3] hard-rule columns missing in df, AND-skipped: {missing_rules}",
            stacklevel=2,
        )

    df_scored = df_scored.with_columns(
        [
            pl.lit(0.0).alias("textbook_trend_score"),
            pl.lit(0.0).alias("textbook_kbar_score"),
            pl.lit(0.0).alias("textbook_volume_score"),
            pl.lit(0.0).alias("textbook_trigger_score"),
            pl.col("textbook_b1_score").alias("textbook_similarity_score"),
            pl.lit(0.0).alias("textbook_rule_score"),
            pl.lit(threshold).alias("textbook_b1_threshold"),
            (
                (pl.col("textbook_b1_score") >= pl.lit(threshold)) & hard_rule_expr
            ).alias("is_textbook_b1"),
        ]
    )
    if sim_col_names:
        df_scored = df_scored.drop(sim_col_names)
    return df_scored


def _apply_textbook_structure_labels(
    df: pl.DataFrame, *, score_version: str = "v1"
) -> pl.DataFrame:
    if score_version not in {"v1", "v3"}:
        raise ValueError(
            f"Unsupported textbook_score_version: {score_version!r}, expected 'v1' or 'v3'"
        )

    case_df = (
        pl.DataFrame(B1_TEXTBOOK_CASES)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .rename({"name": "textbook_case_name"})
        .with_columns(pl.lit(True).alias("is_textbook_case"))
    )
    df_with_case_flag = (
        df.join(case_df, on=["code", "date"], how="left")
        .with_columns(
            [
                pl.col("textbook_case_name").fill_null("").alias("textbook_case_name"),
                pl.col("is_textbook_case").fill_null(False).alias("is_textbook_case"),
            ]
        )
    )

    case_rows = df_with_case_flag.filter(pl.col("is_textbook_case"))
    if case_rows.is_empty():
        return df_with_case_flag.with_columns(_empty_textbook_columns())

    if score_version == "v3":
        return _score_textbook_v3(df_with_case_flag, case_rows)
    return _score_textbook_v1(df_with_case_flag, case_rows)


def _manual_regime_expr(loose_periods: Sequence[tuple[str, str]]) -> pl.Expr:
    expr = pl.lit(False)
    for start, end in loose_periods:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        expr = expr | pl.col("date").is_between(start_date, end_date, closed="both")
    return expr


def build_b1_research_frame(
    q_full: pl.LazyFrame,
    *,
    mv_min: float,
    mv_max: float,
    min_list_days: int,
    seed_j_max: float,
    loose_periods: Sequence[tuple[str, str]],
    label_horizon: int = 10,
    include_rotation_kbar_features: bool = False,
    textbook_score_version: str = "v1",
) -> pl.DataFrame:
    future_high_cols = [
        pl.col("high_adj").shift(-step).over("code").alias(f"_fwd_high_{step}")
        for step in range(1, label_horizon + 1)
    ]
    future_low_cols = [
        pl.col("low_adj").shift(-step).over("code").alias(f"_fwd_low_{step}")
        for step in range(1, label_horizon + 1)
    ]
    future_high_names = [f"_fwd_high_{step}" for step in range(1, label_horizon + 1)]
    future_low_names = [f"_fwd_low_{step}" for step in range(1, label_horizon + 1)]

    # ── Phase A: 在全量数据上计算前瞻标签 ────────────────────────────
    # 标签必须在 market_cap / _list_days 过滤之前计算，否则边界股票
    # 过滤后剩余行不连续，shift(-step) 会跳到多年之后，产生错误标签，
    # 且标签 null/non-null 状态随 END_DATE 变化，导致不可复现。
    df_with_labels = (
        calc_b1_factors_wmacd(q_full, {"MV_THRESHOLD": mv_min})
        .with_columns(pl.col("date").cum_count().over("code").alias("_list_days"))
        .with_columns(
            future_high_cols
            + future_low_cols
            + [
                (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
                (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
                (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
                (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
            ]
        )
        .with_columns(
            [
                (pl.max_horizontal(future_high_names) / pl.col("close_adj") - 1).alias("fwd_mfe_10d"),
                (pl.min_horizontal(future_low_names) / pl.col("close_adj") - 1).alias("fwd_mae_10d"),
                (pl.col("close_adj").shift(-10).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_10d"),
            ]
        )
        .with_columns(
            [
                (pl.col("fwd_mfe_10d") + pl.col("fwd_mae_10d")).alias("fwd_net_10d"),
                (pl.col("fwd_mfe_10d") / (1 + pl.col("fwd_mae_10d").abs())).alias("fwd_mfe_risk_adj_10d"),
            ]
        )
    )
    research_frame = df_with_labels
    if include_rotation_kbar_features:
        research_frame = calc_alpha158_factors(
            calc_rotation_factors(research_frame),
            use_kbar=True,
            price_fields=(),
            price_windows=(),
            volume_windows=(),
            rolling_windows=(),
        )
    optional_rotation_kbar_cols = (
        list(B1_ROTATION_CORE12_KBAR_FEATURE_COLS)
        if include_rotation_kbar_features
        else []
    )

    # ── Phase B: 过滤宇宙 + 计算特征 ─────────────────────────────────
    df_result = (
        research_frame
        .filter(
            (pl.col("_list_days") >= min_list_days)
            & (pl.col("market_cap_100m") >= mv_min)
            & (pl.col("market_cap_100m") <= mv_max)
        )
        .with_columns(_manual_regime_expr(loose_periods).alias("is_manual_bull"))
        .with_columns(
            [
                pl.col("close_adj").shift(1).over("code").alias("_prev_close_local"),
                pl.col("volume").shift(1).over("code").alias("_prev_volume_local"),
                pl.col("volume").rolling_mean(40).over("code").alias("_vol_ma40"),
                pl.when(pl.col("KEY_K"))
                .then(pl.col("date"))
                .otherwise(None)
                .forward_fill()
                .over("code")
                .alias("_last_key_k_date"),
                pl.col("volume").rolling_max(20).over("code").alias("_vol_max_20"),
                pl.col("volume").rolling_max(40).over("code").alias("_vol_max_40"),
                pl.when(pl.col("close_adj") >= pl.col("open_adj"))
                .then(pl.col("volume"))
                .otherwise(0.0)
                .rolling_sum(20)
                .over("code")
                .alias("_vol_yang_20"),
                pl.when(pl.col("close_adj") < pl.col("open_adj"))
                .then(pl.col("volume"))
                .otherwise(0.0)
                .rolling_sum(20)
                .over("code")
                .alias("_vol_yin_20"),
                pl.col("high_adj").rolling_max(20).over("code").alias("_peak_20"),
                pl.col("low_adj").rolling_min(20).over("code").alias("_trough_20"),
            ]
        )
        .with_columns(
            [
                ((pl.col("J") <= seed_j_max) & (pl.col("WL") > pl.col("YL"))).alias("seed_loose"),
                (
                    (pl.col("J") <= seed_j_max)
                    & (pl.col("WL") > pl.col("YL"))
                    & (pl.col("close_adj") > pl.col("YL"))
                ).alias("seed_mid"),
                (
                    (pl.col("J") <= seed_j_max)
                    & (pl.col("WL") > pl.col("YL"))
                    & (pl.col("close_adj") > pl.col("YL"))
                    & pl.col("TRIGGER")
                ).alias("seed_strict"),
                (
                    (pl.col("J") <= seed_j_max)
                    & (pl.col("WL") > pl.col("YL"))
                    & (pl.col("close_adj") > pl.col("YL"))
                    & pl.col("TRIGGER")
                    & pl.col("GOOD28")
                    & pl.col("MAX28_OK")
                ).alias("seed_strict_v2"),
                (pl.col("volume") / pl.max_horizontal(pl.col("_vol_max_20"), pl.lit(1.0))).alias("vol_shrink_20"),
                (pl.col("volume") / pl.max_horizontal(pl.col("_vol_max_40"), pl.lit(1.0))).alias("vol_shrink_40"),
                (pl.col("_vol_yang_20") / pl.max_horizontal(pl.col("_vol_yin_20"), pl.lit(1.0))).alias(
                    "red_green_ratio_20"
                ),
                # 教科书 B1 "前期放量启动" 序列识别 (z 哥 B1 完美图原版第 1 条规则的可计算化)
                # 过去 60 天内是否至少出现 1 根 "放量阳线" (vol/vol_ma40 >= 2 且 close>open)
                (
                    (
                        (pl.col("volume") / pl.max_horizontal(pl.col("_vol_ma40"), pl.lit(1.0)))
                        * pl.when(pl.col("close_adj") > pl.col("open_adj")).then(1.0).otherwise(0.0)
                    ).rolling_max(60).over("code") >= 2.0
                ).alias("prior_volume_surge_60d"),
                # 当日量 / 过去 60 天最大量 (排除当日, 用 shift(1)), 教科书 "对比前期高点放量极致缩量" 的代理
                (
                    pl.col("volume") / pl.max_horizontal(
                        pl.col("volume").shift(1).rolling_max(60).over("code"), pl.lit(1.0)
                    )
                ).alias("peak_vol_shrink_60d"),
                # 近 5 日均量 / 近 20 日均量, 越小越说明 "顶部/回调段持续缩量" (序列化)
                (
                    pl.col("volume").rolling_mean(5).over("code")
                    / pl.max_horizontal(pl.col("volume").rolling_mean(20).over("code"), pl.lit(1.0))
                ).alias("pullback_vol_shrink_5_20"),
                ((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.max_horizontal(pl.col("open_adj"), pl.lit(0.01))).alias(
                    "body_pct"
                ),
                (
                    (pl.col("high_adj") - pl.max_horizontal(pl.col("close_adj"), pl.col("open_adj")))
                    / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
                ).alias("upper_shadow_pct"),
                (
                    (pl.min_horizontal(pl.col("close_adj"), pl.col("open_adj")) - pl.col("low_adj"))
                    / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
                ).alias("lower_shadow_pct"),
                (
                    (pl.col("_peak_20") - pl.col("close_adj"))
                    / pl.max_horizontal(pl.col("_peak_20") - pl.col("_trough_20"), pl.lit(1e-8))
                ).alias("retrace_ratio_20"),
                (pl.col("date") - pl.col("_last_key_k_date")).dt.total_days().alias("days_since_key_k"),
            ]
        )
        .with_columns(
            [
                pl.col("TRIGGER").cast(pl.Float64).rolling_mean(10).over("code").alias("trigger_recent_10"),
                pl.col("KEY_K").cast(pl.Float64).rolling_mean(20).over("code").alias("key_k_recent_20"),
                pl.col("PLRY_CNT").cast(pl.Float64).rolling_mean(10).over("code").alias("plry_cluster_recent_10"),
                (1.0 / (1.0 + pl.col("days_since_key_k").fill_null(9999.0))).alias("days_since_key_k_inv"),
                ((pl.col("high_adj") - pl.col("low_adj")) / pl.max_horizontal(pl.col("close_adj"), pl.lit(0.01)) * 100).alias(
                    "range_pct"
                ),
                (
                    (pl.col("close_adj") - pl.col("open_adj")).abs()
                    / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
                ).alias("body_to_range"),
                (
                    (pl.col("close_adj") - pl.col("low_adj"))
                    / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
                ).alias("close_pos_in_bar"),
                (
                    (pl.col("open_adj") - pl.col("_prev_close_local"))
                    / pl.max_horizontal(pl.col("_prev_close_local"), pl.lit(0.01))
                    * 100
                ).alias("gap_from_prev_close_pct"),
                (pl.col("volume") / pl.max_horizontal(pl.col("_prev_volume_local"), pl.lit(1.0))).alias("vol_to_prev_vol"),
                (pl.col("volume") / pl.max_horizontal(pl.col("_vol_ma40"), pl.lit(1.0))).alias("vol_to_avg40"),
                (pl.col("vol_shrink_20") - pl.col("vol_shrink_20").shift(5).over("code")).alias("vol_shrink_20_delta_5"),
                (pl.col("red_green_ratio_20") - pl.col("red_green_ratio_20").shift(5).over("code")).alias(
                    "red_green_ratio_delta_5"
                ),
                (pl.col("rw_hist") - pl.col("rw_hist").shift(5).over("code")).alias("rw_hist_delta_5"),
                (pl.col("rm_hist") - pl.col("rm_hist").shift(5).over("code")).alias("rm_hist_delta_5"),
                (pl.col("Bias_WL_YL") - pl.col("Bias_WL_YL").shift(5).over("code")).alias("bias_wl_yl_delta_5"),
                (pl.col("close_adj").gt(pl.col("YL")).cast(pl.Float64).rolling_mean(5).over("code")).alias(
                    "close_above_yl_pct_5"
                ),
                (pl.col("close_adj").gt(pl.col("WL")).cast(pl.Float64).rolling_mean(5).over("code")).alias(
                    "close_above_wl_pct_5"
                ),
            ]
        )
        .select(
            list(
                dict.fromkeys(
                    [
                        "code",
                        "date",
                        "open_adj",
                        "high_adj",
                        "low_adj",
                        "close_adj",
                        "volume",
                        "amount",
                        "market_cap_100m",
                        "WL",
                        "YL",
                        "J",
                        "K",
                        "D",
                        "TRIGGER",
                        "KEY_K",
                        "KEY_K_EXIST",
                        "PLRY_CNT",
                        "GOOD28",
                        "MAX28_OK",
                        "YANGYIN_OK",
                        "Bias_C_WL",
                        "Bias_C_YL",
                        "Bias_WL_YL",
                        "rw_dif",
                        "rw_dif_pct",
                        "rw_hist",
                        "rm_hist",
                        "bad_k_count",
                        "seed_loose",
                        "seed_mid",
                        "seed_strict",
                        "seed_strict_v2",
                        "is_manual_bull",
                        "prior_volume_surge_60d",
                        "peak_vol_shrink_60d",
                        "pullback_vol_shrink_5_20",
                        *B1_MINING_FEATURE_COLS,
                        *optional_rotation_kbar_cols,
                        "fwd_ret_1d",
                        "fwd_ret_2d",
                        "fwd_ret_3d",
                        "fwd_ret_5d",
                        "fwd_ret_10d",
                        "fwd_mfe_10d",
                        "fwd_mae_10d",
                        "fwd_net_10d",
                        "fwd_mfe_risk_adj_10d",
                    ]
                )
            )
        )
        .collect()
    )
    return _apply_textbook_structure_labels(
        df_result, score_version=textbook_score_version
    )
