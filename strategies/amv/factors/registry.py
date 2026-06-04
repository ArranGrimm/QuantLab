"""AMV 因子注册表：按名查找，按需计算。

每个因子自描述——名字、Polars 表达式、依赖的中间列。
pipeline 收集 RankerSpec 所需的因子名 → 解析依赖 → 只计算被依赖的列。
"""

from __future__ import annotations

import polars as pl

_A_SHARE_LOT_SIZE = 100.0
_EPS = 1e-12

# ── intermediate helpers ────────────────────────────────────────────────

def _ensure_intermediates(frame: pl.LazyFrame, needed: set[str]) -> pl.LazyFrame:
    """Compute shared intermediate columns that any final factor depends on."""
    schema_names = frame.collect_schema().names()
    missing = needed - set(schema_names)
    if not missing:
        return frame

    exprs: list[pl.Expr] = []

    # --- tier 1: raw derivations ---
    tier1 = {
        "_pc", "_ret", "_tr", "turnover_rate",
        "ret_5d", "ret_20d", "_down_vol_sum_20", "_total_vol_sum_20",
    }
    if missing & tier1:
        if "_pc" not in schema_names:
            exprs.append(pl.col("close_adj").shift(1).over("code").alias("_pc"))
        if "_ret" not in schema_names:
            exprs.append((pl.col("close_adj") / pl.col("_pc") - 1).alias("_ret"))
        if "_tr" not in schema_names:
            exprs.append(
                pl.max_horizontal(
                    pl.col("high_adj") - pl.col("low_adj"),
                    (pl.col("high_adj") - pl.col("_pc")).abs(),
                    (pl.col("low_adj") - pl.col("_pc")).abs(),
                ).alias("_tr")
            )
        if "turnover_rate" not in schema_names:
            if "turnover" in schema_names:
                exprs.append((pl.col("turnover") * 100.0).fill_nan(0.0).alias("turnover_rate"))
            else:
                exprs.append(
                    ((pl.col("volume") * _A_SHARE_LOT_SIZE) / pl.col("circulating_capital").fill_null(1) * 100)
                    .fill_nan(0.0).alias("turnover_rate")
                )
        if "ret_5d" in missing:
            exprs.append((pl.col("close_adj") / pl.col("close_adj").shift(5).over("code") - 1).alias("ret_5d"))
        if "ret_20d" in missing:
            exprs.append((pl.col("close_adj") / pl.col("close_adj").shift(20).over("code") - 1).alias("ret_20d"))
        if "_down_vol_sum_20" in missing:
            exprs.append(
                pl.when(pl.col("_ret") < 0).then(pl.col("volume")).otherwise(0.0)
                .rolling_sum(20).over("code").alias("_down_vol_sum_20")
            )
        if "_total_vol_sum_20" in missing:
            exprs.append(pl.col("volume").rolling_sum(20).over("code").alias("_total_vol_sum_20"))

    if exprs:
        frame = frame.with_columns(exprs)

    # --- tier 2: rolling aggregates ---
    tier2 = {"panic_vol_ratio_20d", "_atr14", "_ma20", "_high_20d", "_c_min_20", "_c_max_20"}
    exprs2: list[pl.Expr] = []
    if missing & tier2:
        schema_names = set(frame.collect_schema().names())  # refresh after tier1
        if "panic_vol_ratio_20d" not in schema_names:
            exprs2.append(
                (pl.col("_down_vol_sum_20") / pl.max_horizontal(pl.col("_total_vol_sum_20"), pl.lit(1.0)))
                .alias("panic_vol_ratio_20d")
            )
        if "_atr14" not in schema_names:
            exprs2.append(pl.col("_tr").rolling_mean(14).over("code").alias("_atr14"))
        if "_ma20" not in schema_names:
            exprs2.append(pl.col("close_adj").rolling_mean(20).over("code").alias("_ma20"))
        if "_high_20d" not in schema_names:
            exprs2.append(pl.col("high_adj").rolling_max(20).over("code").alias("_high_20d"))
        if "_c_min_20" not in schema_names:
            exprs2.append(pl.col("close_adj").rolling_min(20).over("code").alias("_c_min_20"))
        if "_c_max_20" not in schema_names:
            exprs2.append(pl.col("close_adj").rolling_max(20).over("code").alias("_c_max_20"))
    if exprs2:
        frame = frame.with_columns(exprs2)

    return frame


# ── factor definitions ──────────────────────────────────────────────────

def _open_den() -> pl.Expr:
    return pl.max_horizontal(pl.col("open_adj"), pl.lit(_EPS))

def _range_den() -> pl.Expr:
    return pl.col("high_adj") - pl.col("low_adj") + _EPS

FACTOR_REGISTRY: dict[str, dict] = {
    # --- trend family ---
    "price_pos_20d": {
        "label": "20日价格位置",
        "requires": {"_c_min_20", "_c_max_20"},
        "expr": lambda: (pl.col("close_adj") - pl.col("_c_min_20"))
                        / pl.max_horizontal(pl.col("_c_max_20") - pl.col("_c_min_20"), pl.lit(0.01)),
    },
    "close_to_high_20d": {
        "label": "接近20日新高",
        "requires": {"_high_20d"},
        "expr": lambda: 1 - pl.col("close_adj") / pl.max_horizontal(pl.col("_high_20d"), pl.lit(0.01)),
    },
    "KLEN": {
        "label": "K线振幅收缩",
        "requires": set(),
        "expr": lambda: (pl.col("high_adj") - pl.col("low_adj")) / _open_den(),
    },
    "KMID2": {
        "label": "实体占比偏强",
        "requires": set(),
        "expr": lambda: (pl.col("close_adj") - pl.col("open_adj")) / _range_den(),
    },
    # --- momentum ---
    "ret_5d": {
        "label": "5日动量",
        "requires": set(),
        "expr": lambda: pl.col("ret_5d"),
    },
    "ret_20d": {
        "label": "20日动量",
        "requires": set(),
        "expr": lambda: pl.col("ret_20d"),
    },
    # --- pullback family ---
    "ma_bias_20": {
        "label": "20日均线偏离",
        "requires": {"_ma20"},
        "expr": lambda: (pl.col("close_adj") - pl.col("_ma20"))
                        / pl.max_horizontal(pl.col("_ma20"), pl.lit(0.01)) * 100,
    },
    "disp_bias_20": {
        "label": "20日成本偏离",
        "requires": set(),  # computes its own _ewm internally
        "expr": lambda: (lambda: (
            pl.when(pl.lit(True)).then(
                pl.col("close_adj")
                / pl.max_horizontal(
                    (pl.col("close_adj") + pl.col("high_adj") + pl.col("low_adj")) / 3
                    * pl.col("volume")
                ).alias("_tp_v_dummy"),
                pl.lit(0.01)
            ) - 1
        ))(),
        "_special": "disp_bias_20",  # marker for ewm computation
    },
    "intraday_pos": {
        "label": "日内收盘位置",
        "requires": set(),
        "expr": lambda: (pl.col("close_adj") - pl.col("low_adj"))
                        / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8)),
    },
    "KSFT": {
        "label": "K线实体重心",
        "requires": set(),
        "expr": lambda: (2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / _open_den(),
    },
    # --- risk ---
    "atr_14_pct": {
        "label": "ATR 风险",
        "requires": {"_atr14"},
        "expr": lambda: pl.col("_atr14") / pl.max_horizontal(pl.col("close_adj"), pl.lit(0.01)),
    },
    "panic_vol_ratio_20d": {
        "label": "恐慌量比",
        "requires": {"_down_vol_sum_20", "_total_vol_sum_20"},
        "expr": lambda: pl.col("panic_vol_ratio_20d"),
    },
}


def compute_required_factors(frame: pl.LazyFrame, factor_names: list[str]) -> pl.LazyFrame:
    """Return frame with ONLY the requested factor columns + shared intermediates added.

    To keep disp_bias_20 working, the ewm intermediate is computed as a special case.
    """
    if not factor_names:
        return frame

    # 1. collect all intermediates needed
    intermediates: set[str] = set()
    for name in factor_names:
        spec = FACTOR_REGISTRY.get(name)
        if spec is None:
            continue
        intermediates |= spec.get("requires", set())

    # 2. compute intermediates
    frame = frame.sort(["code", "date"])
    frame = _ensure_intermediates(frame, intermediates)

    # 3. handle disp_bias_20's special ewm computation
    if "disp_bias_20" in factor_names:
        frame = frame.with_columns([
            ((pl.col("close_adj") + pl.col("high_adj") + pl.col("low_adj")) / 3 * pl.col("volume")).alias("_tp_v"),
        ])
        frame = frame.with_columns([
            pl.col("_tp_v").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_pv_20"),
            pl.col("volume").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_v_20"),
        ])

    # 4. compute final factor expressions
    exprs: list[pl.Expr] = []
    for name in factor_names:
        spec = FACTOR_REGISTRY.get(name)
        if spec is None:
            continue
        if name == "disp_bias_20":
            exprs.append(
                (pl.col("close_adj")
                 / pl.max_horizontal(pl.col("_ewm_pv_20") / pl.max_horizontal(pl.col("_ewm_v_20"), pl.lit(1e-10)),
                                     pl.lit(0.01))
                 - 1).alias("disp_bias_20")
            )
        else:
            exprs.append(spec["expr"]().alias(name))

    if exprs:
        frame = frame.with_columns(exprs)

    return frame
