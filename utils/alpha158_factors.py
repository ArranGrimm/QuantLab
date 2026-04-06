"""
Qlib Alpha158 的 Polars 复刻版。

默认配置与 `qlib.contrib.data.handler.Alpha158` 对齐:
- kbar: 9 个
- price: OPEN/HIGH/LOW/VWAP, windows=[0]
- rolling: 29 类算子, windows=[5, 10, 20, 30, 60]

默认总特征数 = 158。
"""

from __future__ import annotations

import polars as pl

_EPS = 1e-12

ALPHA158_DEFAULT_PRICE_FIELDS = ("OPEN", "HIGH", "LOW", "VWAP")
ALPHA158_DEFAULT_PRICE_WINDOWS = (0,)
ALPHA158_DEFAULT_VOLUME_WINDOWS: tuple[int, ...] = ()
ALPHA158_DEFAULT_ROLLING_WINDOWS = (5, 10, 20, 30, 60)
ALPHA158_DEFAULT_ROLLING_OPS = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "LOW",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)

_KBAR_FACTOR_COLS = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
)

_PRICE_FIELD_TO_COL = {
    "OPEN": "open_adj",
    "HIGH": "high_adj",
    "LOW": "low_adj",
    "CLOSE": "close_adj",
    "VWAP": "vwap_adj",
}

_ROLLING_NAME_PREFIX = {
    "ROC": "ROC",
    "MA": "MA",
    "STD": "STD",
    "BETA": "BETA",
    "RSQR": "RSQR",
    "RESI": "RESI",
    "MAX": "MAX",
    "LOW": "MIN",
    "QTLU": "QTLU",
    "QTLD": "QTLD",
    "RANK": "RANK",
    "RSV": "RSV",
    "IMAX": "IMAX",
    "IMIN": "IMIN",
    "IMXD": "IMXD",
    "CORR": "CORR",
    "CORD": "CORD",
    "CNTP": "CNTP",
    "CNTN": "CNTN",
    "CNTD": "CNTD",
    "SUMP": "SUMP",
    "SUMN": "SUMN",
    "SUMD": "SUMD",
    "VMA": "VMA",
    "VSTD": "VSTD",
    "WVMA": "WVMA",
    "VSUMP": "VSUMP",
    "VSUMN": "VSUMN",
    "VSUMD": "VSUMD",
}

ALPHA158_FACTOR_GROUPS = {
    "kbar_shape": list(_KBAR_FACTOR_COLS),
    "price_level": ["OPEN0", "HIGH0", "LOW0", "VWAP0"],
    "price_trend": [
        *[f"ROC{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"MA{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"STD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "trend_regression": [
        *[f"BETA{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"RSQR{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"RESI{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "range_position": [
        *[f"MAX{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"MIN{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"QTLU{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"QTLD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"RSV{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "timing_position": [
        *[f"RANK{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"IMAX{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"IMIN{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"IMXD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "price_volume_corr": [
        *[f"CORR{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"CORD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "directionality": [
        *[f"CNTP{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"CNTN{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"CNTD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"SUMP{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"SUMN{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"SUMD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
    "volume_dynamics": [
        *[f"VMA{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"VSTD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"WVMA{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"VSUMP{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"VSUMN{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
        *[f"VSUMD{window}" for window in ALPHA158_DEFAULT_ROLLING_WINDOWS],
    ],
}

ALPHA158_FACTOR_GROUP_LABELS = {
    "kbar_shape": "K线形态",
    "price_level": "价格水平",
    "price_trend": "价格趋势",
    "trend_regression": "趋势回归",
    "range_position": "区间位置",
    "timing_position": "时点位置",
    "price_volume_corr": "量价相关",
    "directionality": "涨跌方向",
    "volume_dynamics": "成交量动态",
}

_ALPHA158_GROUP_ROLLING_OPS = {
    "price_trend": ("ROC", "MA", "STD"),
    "trend_regression": ("BETA", "RSQR", "RESI"),
    "range_position": ("MAX", "LOW", "QTLU", "QTLD", "RSV"),
    "timing_position": ("RANK", "IMAX", "IMIN", "IMXD"),
    "price_volume_corr": ("CORR", "CORD"),
    "directionality": ("CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD"),
    "volume_dynamics": ("VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"),
}


def get_alpha158_factor_group_map() -> dict[str, str]:
    factor_to_group: dict[str, str] = {}
    for group_key, factors in ALPHA158_FACTOR_GROUPS.items():
        for factor in factors:
            if factor in factor_to_group:
                raise ValueError(f"Alpha158 factor '{factor}' is assigned to multiple groups")
            factor_to_group[factor] = group_key

    missing = [factor for factor in ALPHA158_FACTOR_COLS if factor not in factor_to_group]
    extra = [factor for factor in factor_to_group if factor not in ALPHA158_FACTOR_COLS]
    if missing or extra:
        raise ValueError(
            f"Alpha158 factor grouping mismatch: missing={missing}, extra={extra}"
        )
    return factor_to_group


def _normalize_group_keys(
    group_mode: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if group_mode is None:
        return list(ALPHA158_FACTOR_GROUPS)

    if isinstance(group_mode, str):
        normalized = group_mode.strip().lower()
        if normalized in {"", "all"}:
            return list(ALPHA158_FACTOR_GROUPS)
        group_keys = [part.strip() for part in normalized.replace("+", ",").split(",") if part.strip()]
    else:
        group_keys = [str(part).strip().lower() for part in group_mode if str(part).strip()]
        if not group_keys:
            return list(ALPHA158_FACTOR_GROUPS)

    invalid = [group_key for group_key in group_keys if group_key not in ALPHA158_FACTOR_GROUPS]
    if invalid:
        raise ValueError(
            f"Unsupported Alpha158 group(s): {invalid}. "
            f"Expected one of: {', '.join(ALPHA158_FACTOR_GROUPS.keys())}, or 'all'"
        )
    return group_keys


def resolve_alpha158_group_config(
    group_mode: str | list[str] | tuple[str, ...] | None = "all",
) -> dict[str, object]:
    group_keys = _normalize_group_keys(group_mode)
    if set(group_keys) == set(ALPHA158_FACTOR_GROUPS):
        return {
            "group_keys": list(ALPHA158_FACTOR_GROUPS.keys()),
            "group_mode_label": "all",
            "use_kbar": True,
            "price_fields": ALPHA158_DEFAULT_PRICE_FIELDS,
            "include_ops": None,
            "factor_cols": list(ALPHA158_FACTOR_COLS),
        }

    selected = {factor for group_key in group_keys for factor in ALPHA158_FACTOR_GROUPS[group_key]}
    include_ops = []
    for group_key in group_keys:
        include_ops.extend(_ALPHA158_GROUP_ROLLING_OPS.get(group_key, ()))

    return {
        "group_keys": group_keys,
        "group_mode_label": ",".join(group_keys),
        "use_kbar": "kbar_shape" in group_keys,
        "price_fields": ALPHA158_DEFAULT_PRICE_FIELDS if "price_level" in group_keys else (),
        "include_ops": tuple(dict.fromkeys(include_ops)),
        "factor_cols": [factor for factor in ALPHA158_FACTOR_COLS if factor in selected],
    }

def build_alpha158_factor_cols(
    *,
    use_kbar: bool = True,
    price_fields: tuple[str, ...] = ALPHA158_DEFAULT_PRICE_FIELDS,
    price_windows: tuple[int, ...] = ALPHA158_DEFAULT_PRICE_WINDOWS,
    volume_windows: tuple[int, ...] = ALPHA158_DEFAULT_VOLUME_WINDOWS,
    rolling_windows: tuple[int, ...] = ALPHA158_DEFAULT_ROLLING_WINDOWS,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
) -> list[str]:
    cols: list[str] = []

    if use_kbar:
        cols.extend(_KBAR_FACTOR_COLS)

    for field in price_fields:
        field_upper = field.upper()
        cols.extend(f"{field_upper}{window}" for window in price_windows)

    if volume_windows:
        cols.extend(f"VOLUME{window}" for window in volume_windows)

    include_set = None if include is None else {op.upper() for op in include}
    exclude_set = {op.upper() for op in exclude}
    for op in ALPHA158_DEFAULT_ROLLING_OPS:
        if op in exclude_set:
            continue
        if include_set is not None and op not in include_set:
            continue
        prefix = _ROLLING_NAME_PREFIX[op]
        cols.extend(f"{prefix}{window}" for window in rolling_windows)

    return cols


ALPHA158_FACTOR_COLS = build_alpha158_factor_cols()

if len(ALPHA158_FACTOR_COLS) != 158:
    raise ValueError(f"Alpha158 factor count mismatch: expected 158, got {len(ALPHA158_FACTOR_COLS)}")


ALPHA158_FACTOR_TO_GROUP = get_alpha158_factor_group_map()


def _rolling_rank_expr(window: int) -> pl.Expr:
    return (
        pl.col("close_adj")
        .rolling_rank(window, method="average")
        .over("code")
        .truediv(float(window))
        .cast(pl.Float64)
    )


def _rolling_beta_expr(window: int) -> pl.Expr:
    return _safe_div(
        pl.rolling_cov(pl.col("_a_t"), pl.col("close_adj"), window_size=window, ddof=0).over("code"),
        pl.col("_a_t").rolling_var(window_size=window, ddof=0).over("code"),
    )


def _rolling_corr_expr(lhs: pl.Expr, rhs: pl.Expr, window: int) -> pl.Expr:
    lhs_var = lhs.rolling_var(window_size=window, ddof=0).over("code")
    rhs_var = rhs.rolling_var(window_size=window, ddof=0).over("code")
    return _safe_div(
        pl.rolling_cov(lhs, rhs, window_size=window, ddof=0).over("code"),
        (lhs_var * rhs_var).sqrt(),
    ).fill_nan(0.0)


def _rolling_rsqr_expr(window: int) -> pl.Expr:
    return _rolling_corr_expr(pl.col("_a_t"), pl.col("close_adj"), window).pow(2)


def _rolling_resi_expr(window: int, beta_expr: pl.Expr) -> pl.Expr:
    t_mean = pl.col("_a_t").rolling_mean(window).over("code")
    y_mean = pl.col("close_adj").rolling_mean(window).over("code")
    return pl.col("close_adj") - (y_mean + beta_expr * (pl.col("_a_t") - t_mean))


def _rolling_extreme_lag_expr(column_name: str, target_expr: pl.Expr, window: int) -> pl.Expr:
    lag_exprs = [
        pl.when(pl.col(column_name).shift(lag).over("code") == target_expr)
        .then(pl.lit(float(lag)))
        .otherwise(pl.lit(-1.0))
        for lag in range(window)
    ]
    return (
        pl.when(pl.col("_a_row_nr") >= window)
        .then(pl.max_horizontal(lag_exprs))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _safe_div(numerator: pl.Expr, denominator: pl.Expr, eps: float = _EPS) -> pl.Expr:
    return numerator / pl.max_horizontal(denominator, pl.lit(eps))


def calc_alpha158_factors(
    df: pl.LazyFrame,
    *,
    use_kbar: bool = True,
    price_fields: tuple[str, ...] = ALPHA158_DEFAULT_PRICE_FIELDS,
    price_windows: tuple[int, ...] = ALPHA158_DEFAULT_PRICE_WINDOWS,
    volume_windows: tuple[int, ...] = ALPHA158_DEFAULT_VOLUME_WINDOWS,
    rolling_windows: tuple[int, ...] = ALPHA158_DEFAULT_ROLLING_WINDOWS,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
) -> pl.LazyFrame:
    """
    在现有日线数据上追加 Qlib Alpha158 风格特征。

    要求输入至少包含:
    `code/date/open_adj/high_adj/low_adj/close_adj/vwap_adj/volume`
    """
    print("[Alpha158] 计算 Polars 复刻特征...")

    include_set = None if include is None else {op.upper() for op in include}
    exclude_set = {op.upper() for op in exclude}

    def use(op_name: str) -> bool:
        return op_name not in exclude_set and (include_set is None or op_name in include_set)

    close_den = pl.max_horizontal(pl.col("close_adj"), pl.lit(_EPS))
    open_den = pl.max_horizontal(pl.col("open_adj"), pl.lit(_EPS))
    range_den = pl.col("high_adj") - pl.col("low_adj") + _EPS

    result = (
        df.sort(["code", "date"])
        .with_columns([
            (pl.col("date").cum_count().over("code") - 1).cast(pl.Float64).alias("_a_t"),
            pl.col("date").cum_count().over("code").cast(pl.Int64).alias("_a_row_nr"),
            pl.col("close_adj").shift(1).over("code").alias("_a_prev_close"),
            pl.col("volume").shift(1).over("code").alias("_a_prev_volume"),
        ])
        .with_columns([
            _safe_div(pl.col("close_adj"), pl.col("_a_prev_close")).alias("_a_close_ratio"),
            (_safe_div(pl.col("volume"), pl.col("_a_prev_volume")) + 1.0).log().alias("_a_vol_ratio_log"),
            (pl.col("volume") + 1.0).log().alias("_a_log_volume"),
            (pl.col("close_adj") - pl.col("_a_prev_close")).alias("_a_close_delta"),
            (pl.col("volume") - pl.col("_a_prev_volume")).alias("_a_volume_delta"),
        ])
        .with_columns([
            pl.max_horizontal(pl.col("_a_close_delta"), pl.lit(0.0)).alias("_a_close_up"),
            pl.max_horizontal(-pl.col("_a_close_delta"), pl.lit(0.0)).alias("_a_close_down"),
            pl.col("_a_close_delta").abs().alias("_a_abs_close_delta"),
            pl.max_horizontal(pl.col("_a_volume_delta"), pl.lit(0.0)).alias("_a_volume_up"),
            pl.max_horizontal(-pl.col("_a_volume_delta"), pl.lit(0.0)).alias("_a_volume_down"),
            pl.col("_a_volume_delta").abs().alias("_a_abs_volume_delta"),
            (pl.col("close_adj") > pl.col("_a_prev_close")).cast(pl.Float64).alias("_a_up_day"),
            (pl.col("close_adj") < pl.col("_a_prev_close")).cast(pl.Float64).alias("_a_down_day"),
            ((pl.col("_a_close_ratio") - 1.0).abs() * pl.col("volume")).alias("_a_wv"),
        ])
    )

    exprs: list[pl.Expr] = []

    if use_kbar:
        exprs.extend([
            ((pl.col("close_adj") - pl.col("open_adj")) / open_den).alias("KMID"),
            ((pl.col("high_adj") - pl.col("low_adj")) / open_den).alias("KLEN"),
            ((pl.col("close_adj") - pl.col("open_adj")) / range_den).alias("KMID2"),
            ((pl.col("high_adj") - pl.max_horizontal(pl.col("open_adj"), pl.col("close_adj"))) / open_den).alias("KUP"),
            ((pl.col("high_adj") - pl.max_horizontal(pl.col("open_adj"), pl.col("close_adj"))) / range_den).alias("KUP2"),
            ((pl.min_horizontal(pl.col("open_adj"), pl.col("close_adj")) - pl.col("low_adj")) / open_den).alias("KLOW"),
            ((pl.min_horizontal(pl.col("open_adj"), pl.col("close_adj")) - pl.col("low_adj")) / range_den).alias("KLOW2"),
            ((2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / open_den).alias("KSFT"),
            ((2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / range_den).alias("KSFT2"),
        ])

    for field in price_fields:
        field_upper = field.upper()
        if field_upper not in _PRICE_FIELD_TO_COL:
            raise ValueError(f"Unsupported Alpha158 price field: {field_upper}")
        field_col = pl.col(_PRICE_FIELD_TO_COL[field_upper])
        for window in price_windows:
            base_expr = field_col.shift(window).over("code") if window != 0 else field_col
            exprs.append((base_expr / close_den).alias(f"{field_upper}{window}"))

    for window in volume_windows:
        base_expr = pl.col("volume").shift(window).over("code") if window != 0 else pl.col("volume")
        exprs.append(_safe_div(base_expr, pl.col("volume")).alias(f"VOLUME{window}"))

    for window in rolling_windows:
        high_roll_max = pl.col("high_adj").rolling_max(window).over("code")
        low_roll_min = pl.col("low_adj").rolling_min(window).over("code")
        beta_expr = _rolling_beta_expr(window)
        imax_expr = _rolling_extreme_lag_expr("high_adj", high_roll_max, window)
        imin_expr = _rolling_extreme_lag_expr("low_adj", low_roll_min, window)

        if use("ROC"):
            exprs.append((pl.col("close_adj").shift(window).over("code") / close_den).alias(f"ROC{window}"))
        if use("MA"):
            exprs.append((pl.col("close_adj").rolling_mean(window).over("code") / close_den).alias(f"MA{window}"))
        if use("STD"):
            exprs.append((pl.col("close_adj").rolling_std(window).over("code") / close_den).alias(f"STD{window}"))
        if use("BETA"):
            exprs.append((beta_expr / close_den).alias(f"BETA{window}"))
        if use("RSQR"):
            exprs.append(_rolling_rsqr_expr(window).alias(f"RSQR{window}"))
        if use("RESI"):
            exprs.append((_rolling_resi_expr(window, beta_expr) / close_den).alias(f"RESI{window}"))
        if use("MAX"):
            exprs.append((high_roll_max / close_den).alias(f"MAX{window}"))
        if use("LOW"):
            exprs.append((low_roll_min / close_den).alias(f"MIN{window}"))
        if use("QTLU"):
            exprs.append(
                (pl.col("close_adj").rolling_quantile(window_size=window, quantile=0.8, interpolation="linear").over("code") / close_den)
                .alias(f"QTLU{window}")
            )
        if use("QTLD"):
            exprs.append(
                (pl.col("close_adj").rolling_quantile(window_size=window, quantile=0.2, interpolation="linear").over("code") / close_den)
                .alias(f"QTLD{window}")
            )
        if use("RANK"):
            exprs.append(_rolling_rank_expr(window).alias(f"RANK{window}"))
        if use("RSV"):
            exprs.append(_safe_div(pl.col("close_adj") - low_roll_min, high_roll_max - low_roll_min).alias(f"RSV{window}"))
        if use("IMAX"):
            exprs.append((imax_expr / float(window)).alias(f"IMAX{window}"))
        if use("IMIN"):
            exprs.append((imin_expr / float(window)).alias(f"IMIN{window}"))
        if use("IMXD"):
            exprs.append(((imax_expr - imin_expr) / float(window)).alias(f"IMXD{window}"))
        if use("CORR"):
            exprs.append(
                _rolling_corr_expr(pl.col("close_adj"), pl.col("_a_log_volume"), window).alias(f"CORR{window}")
            )
        if use("CORD"):
            exprs.append(
                _rolling_corr_expr(pl.col("_a_close_ratio"), pl.col("_a_vol_ratio_log"), window).alias(f"CORD{window}")
            )
        if use("CNTP"):
            exprs.append(pl.col("_a_up_day").rolling_mean(window).over("code").alias(f"CNTP{window}"))
        if use("CNTN"):
            exprs.append(pl.col("_a_down_day").rolling_mean(window).over("code").alias(f"CNTN{window}"))
        if use("CNTD"):
            exprs.append((pl.col("_a_up_day").rolling_mean(window).over("code") - pl.col("_a_down_day").rolling_mean(window).over("code")).alias(f"CNTD{window}"))
        if use("SUMP"):
            exprs.append(
                _safe_div(
                    pl.col("_a_close_up").rolling_sum(window).over("code"),
                    pl.col("_a_abs_close_delta").rolling_sum(window).over("code"),
                ).alias(f"SUMP{window}")
            )
        if use("SUMN"):
            exprs.append(
                _safe_div(
                    pl.col("_a_close_down").rolling_sum(window).over("code"),
                    pl.col("_a_abs_close_delta").rolling_sum(window).over("code"),
                ).alias(f"SUMN{window}")
            )
        if use("SUMD"):
            exprs.append(
                _safe_div(
                    pl.col("_a_close_up").rolling_sum(window).over("code") - pl.col("_a_close_down").rolling_sum(window).over("code"),
                    pl.col("_a_abs_close_delta").rolling_sum(window).over("code"),
                ).alias(f"SUMD{window}")
            )
        if use("VMA"):
            exprs.append(_safe_div(pl.col("volume").rolling_mean(window).over("code"), pl.col("volume")).alias(f"VMA{window}"))
        if use("VSTD"):
            exprs.append(_safe_div(pl.col("volume").rolling_std(window).over("code"), pl.col("volume")).alias(f"VSTD{window}"))
        if use("WVMA"):
            exprs.append(
                _safe_div(
                    pl.col("_a_wv").rolling_std(window).over("code"),
                    pl.col("_a_wv").rolling_mean(window).over("code"),
                ).alias(f"WVMA{window}")
            )
        if use("VSUMP"):
            exprs.append(
                _safe_div(
                    pl.col("_a_volume_up").rolling_sum(window).over("code"),
                    pl.col("_a_abs_volume_delta").rolling_sum(window).over("code"),
                ).alias(f"VSUMP{window}")
            )
        if use("VSUMN"):
            exprs.append(
                _safe_div(
                    pl.col("_a_volume_down").rolling_sum(window).over("code"),
                    pl.col("_a_abs_volume_delta").rolling_sum(window).over("code"),
                ).alias(f"VSUMN{window}")
            )
        if use("VSUMD"):
            exprs.append(
                _safe_div(
                    pl.col("_a_volume_up").rolling_sum(window).over("code") - pl.col("_a_volume_down").rolling_sum(window).over("code"),
                    pl.col("_a_abs_volume_delta").rolling_sum(window).over("code"),
                ).alias(f"VSUMD{window}")
            )

    return result.with_columns(exprs)
