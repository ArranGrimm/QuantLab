"""因子注册表：按名查找，按需计算。

所有因子自描述——名字、Polars 表达式、依赖的中间列、状态。
strategy/ 和 research/ 共享同一套因子定义。
新增因子只需在 FACTOR_REGISTRY 加一行，compute_required_factors 零修改。
"""

from __future__ import annotations

import polars as pl
from typing import Literal

_A_SHARE_LOT_SIZE = 100.0
_EPS = 1e-12

FactorStatus = Literal["active", "experimental", "dead"]

# ── shared expression fragments ──

_open_den = pl.max_horizontal(pl.col("open_adj"), pl.lit(_EPS))
_range_den = pl.col("high_adj") - pl.col("low_adj") + _EPS


# ═══════════════════════════════════════════════════════════════════════════
# Intermediate column computation
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_intermediates(frame: pl.LazyFrame, needed: set[str]) -> pl.LazyFrame:
    """Compute shared intermediate columns.

    Split into tiers because Polars with_columns can't reference
    columns created in the same call — each tier only references
    columns from previous tiers or raw input.
    """
    schema_names = frame.collect_schema().names()
    missing = needed - set(schema_names)
    if not missing:
        return frame

    # ── resolve transitive deps (loop until stable) ──
    _TIER_DEPS: dict[str, set[str]] = {
        "_ret": {"_pc"}, "_tr": {"_pc"},
        "_atr14": {"_tr"},
        "_down_vol_sum_20": {"_ret"},
        "_ewm_pv_20": {"_tp_v"},
        "_ewm_v_20": {"_tp_v"},
        "panic_vol_ratio_20d": {"_down_vol_sum_20", "_total_vol_sum_20"},
        "_cgo_G": {"_cgo_g"},
        "_cgo_w": {"_cgo_G"},
        "_cgo_rp": {"_cgo_w"},
        "_qm_sigma_60d": {"_ret"},
        "_terr_mkt": {"_ret"},
        "_terr_sigma": {"_terr_mkt"},
        "_terr_weighted": {"_terr_sigma"},
        "_terr_avg_20d": {"_terr_weighted"},
        "_terr_std_20d": {"_terr_weighted"},
        "_stv_f": {"_ret"},
        "_stv_mkt": {"_stv_f"},
        "_stv_sigma": {"_stv_mkt"},
        "_stv_weighted": {"_stv_sigma"},
        "_stv_avg_20d": {"_stv_weighted"},
        "_stv_std_20d": {"_stv_weighted"},
    }
    changed = True
    while changed:
        changed = False
        for col in list(missing):
            for dep in _TIER_DEPS.get(col, set()):
                if dep not in schema_names and dep not in missing:
                    missing.add(dep)
                    changed = True

    # ── tier 1a: from raw columns only ──
    t1a_cols, exprs1a = _build_tier1a(schema_names, missing)
    if exprs1a:
        frame = frame.with_columns(exprs1a)

    # ── tier 1b: from tier 1a ──
    t1b = {"_ret", "_tr", "_cgo_G"}
    exprs1b: list[pl.Expr] = []
    if missing & t1b:
        if "_ret" in missing:
            exprs1b.append((pl.col("close_adj") / pl.col("_pc") - 1).alias("_ret"))
        if "_tr" in missing:
            exprs1b.append(
                pl.max_horizontal(
                    pl.col("high_adj") - pl.col("low_adj"),
                    (pl.col("high_adj") - pl.col("_pc")).abs(),
                    (pl.col("low_adj") - pl.col("_pc")).abs(),
                ).alias("_tr")
            )
        if "_cgo_G" in missing:
            exprs1b.append(pl.col("_cgo_g").cum_prod().over("code").alias("_cgo_G"))
    if exprs1b:
        frame = frame.with_columns(exprs1b)

    # ── tier 1c: from tier 1b ──
    t1c = {"_cgo_w"}
    exprs1c: list[pl.Expr] = []
    if missing & t1c:
        if "_cgo_w" in missing:
            t = (pl.col("turnover") / 100.0).clip(1e-7, 1.0 - 1e-7)
            exprs1c.append((t / pl.col("_cgo_G")).alias("_cgo_w"))
    if exprs1c:
        frame = frame.with_columns(exprs1c)

    # ── tier 2a: rolling / ewm from tier1 output ──
    t2a_cols, exprs2a = _build_tier2a(missing)
    if exprs2a:
        frame = frame.with_columns(exprs2a)

    # ── tier 2b: composites from tier2a ──
    exprs2b: list[pl.Expr] = []
    if "panic_vol_ratio_20d" in missing:
        exprs2b.append(
            (pl.col("_down_vol_sum_20")
             / pl.max_horizontal(pl.col("_total_vol_sum_20"), pl.lit(1.0)))
            .alias("panic_vol_ratio_20d")
        )
    if exprs2b:
        frame = frame.with_columns(exprs2b)

    # ── Terrified Score chain: _terr_mkt → _terr_sigma → _terr_weighted → rolling ──
    _terr_cols = {"_terr_mkt", "_terr_sigma", "_terr_weighted", "_terr_avg_20d", "_terr_std_20d"}
    if missing & _terr_cols:
        frame = _build_terrified_chain(frame, missing)

    # ── STV chain: _stv_f → _stv_mkt → _stv_sigma → _stv_weighted → rolling ──
    _stv_cols = {"_stv_f", "_stv_mkt", "_stv_sigma", "_stv_weighted", "_stv_avg_20d", "_stv_std_20d"}
    if missing & _stv_cols:
        frame = _build_stv_chain(frame, missing)

    return frame


def _build_terrified_chain(
    frame: pl.LazyFrame, missing: set[str]
) -> pl.LazyFrame:
    """Terrified Score intermediates, each in its own with_columns."""
    if "_terr_mkt" in missing:
        frame = frame.with_columns(
            pl.col("_ret").mean().over("date").alias("_terr_mkt")
        )
    if "_terr_sigma" in missing:
        frame = frame.with_columns(
            ((pl.col("_ret") - pl.col("_terr_mkt")).abs()
             / (pl.col("_ret").abs() + pl.col("_terr_mkt").abs() + 0.1))
            .alias("_terr_sigma")
        )
    if "_terr_weighted" in missing:
        frame = frame.with_columns(
            (pl.col("_terr_sigma") * pl.col("_ret")).alias("_terr_weighted")
        )
    if "_terr_avg_20d" in missing:
        frame = frame.with_columns(
            pl.col("_terr_weighted").rolling_mean(20).over("code").alias("_terr_avg_20d")
        )
    if "_terr_std_20d" in missing:
        frame = frame.with_columns(
            pl.col("_terr_weighted").rolling_std(20).over("code").alias("_terr_std_20d")
        )
    return frame


def _build_stv_chain(
    frame: pl.LazyFrame, missing: set[str]
) -> pl.LazyFrame:
    """STV intermediates, each in its own with_columns."""
    if "_stv_f" in missing:
        abs_ret = pl.col("_ret").abs()
        frame = frame.with_columns(
            pl.when(abs_ret >= 0.1).then(abs_ret * 100.0)
            .otherwise(pl.col("turnover")).alias("_stv_f")
        )
    if "_stv_mkt" in missing:
        frame = frame.with_columns(
            pl.col("_stv_f").mean().over("date").alias("_stv_mkt")
        )
    if "_stv_sigma" in missing:
        frame = frame.with_columns(
            ((pl.col("_stv_f") - pl.col("_stv_mkt")).abs()
             / (pl.col("_stv_f").abs() + pl.col("_stv_mkt").abs() + 0.1))
            .alias("_stv_sigma")
        )
    if "_stv_weighted" in missing:
        frame = frame.with_columns(
            (pl.col("_stv_sigma") * pl.col("_stv_f")).alias("_stv_weighted")
        )
    if "_stv_avg_20d" in missing:
        frame = frame.with_columns(
            pl.col("_stv_weighted").rolling_mean(20).over("code").alias("_stv_avg_20d")
        )
    if "_stv_std_20d" in missing:
        frame = frame.with_columns(
            pl.col("_stv_weighted").rolling_std(20).over("code").alias("_stv_std_20d")
        )
    return frame


def _build_tier1a(
    schema_names: frozenset | set, missing: set[str]
) -> tuple[set[str], list[pl.Expr]]:
    """Tier 1a: derivations from raw market columns only."""
    cols = {
        "_pc", "ret_5d", "ret_20d", "turnover_rate", "_tp_v", "_cgo_g",
        "_qm_ret_60d",
    }
    exprs: list[pl.Expr] = []
    if not (missing & cols):
        return cols, exprs

    if "_pc" in missing:
        exprs.append(pl.col("close_adj").shift(1).over("code").alias("_pc"))
    if "_tp_v" in missing:
        exprs.append(
            ((pl.col("close_adj") + pl.col("high_adj") + pl.col("low_adj"))
             / 3 * pl.col("volume")).alias("_tp_v")
        )
    if "ret_5d" in missing:
        exprs.append(
            (pl.col("close_adj") / pl.col("close_adj").shift(5).over("code") - 1)
            .alias("ret_5d")
        )
    if "ret_20d" in missing:
        exprs.append(
            (pl.col("close_adj") / pl.col("close_adj").shift(20).over("code") - 1)
            .alias("ret_20d")
        )
    if "turnover_rate" in missing:
        if "turnover" in schema_names:
            exprs.append(
                (pl.col("turnover") * 100.0).fill_nan(0.0).alias("turnover_rate")
            )
        else:
            exprs.append(
                ((pl.col("volume") * _A_SHARE_LOT_SIZE)
                 / pl.col("circulating_capital").fill_null(1) * 100)
                .fill_nan(0.0).alias("turnover_rate")
            )
    if "_cgo_g" in missing:
        t = (pl.col("turnover") / 100.0).clip(1e-7, 1.0 - 1e-7)
        exprs.append((1.0 - t).alias("_cgo_g"))
    if "_qm_ret_60d" in missing:
        exprs.append(
            (pl.col("close_adj") / pl.col("close_adj").shift(60).over("code") - 1.0)
            .alias("_qm_ret_60d")
        )
    return cols, exprs


def _build_tier2a(missing: set[str]) -> tuple[set[str], list[pl.Expr]]:
    """Tier 2a: rolling/ewm aggregates."""
    cols = {
        "_ma5", "_ma10", "_ma20", "_ma60",
        "_high_20d", "_c_min_20", "_c_max_20",
        "_atr14", "_down_vol_sum_20", "_total_vol_sum_20",
        "_ewm_pv_20", "_ewm_v_20", "_cgo_rp",
        "_qm_sigma_60d",
    }
    exprs: list[pl.Expr] = []
    if not (missing & cols):
        return cols, exprs

    if "_ma5" in missing:
        exprs.append(pl.col("close_adj").rolling_mean(5).over("code").alias("_ma5"))
    if "_ma10" in missing:
        exprs.append(pl.col("close_adj").rolling_mean(10).over("code").alias("_ma10"))
    if "_ma20" in missing:
        exprs.append(pl.col("close_adj").rolling_mean(20).over("code").alias("_ma20"))
    if "_ma60" in missing:
        exprs.append(pl.col("close_adj").rolling_mean(60).over("code").alias("_ma60"))
    if "_high_20d" in missing:
        exprs.append(pl.col("high_adj").rolling_max(20).over("code").alias("_high_20d"))
    if "_c_min_20" in missing:
        exprs.append(pl.col("close_adj").rolling_min(20).over("code").alias("_c_min_20"))
    if "_c_max_20" in missing:
        exprs.append(pl.col("close_adj").rolling_max(20).over("code").alias("_c_max_20"))
    if "_atr14" in missing:
        exprs.append(pl.col("_tr").rolling_mean(14).over("code").alias("_atr14"))
    if "_down_vol_sum_20" in missing:
        exprs.append(
            pl.when(pl.col("_ret") < 0).then(pl.col("volume")).otherwise(0.0)
            .rolling_sum(20).over("code").alias("_down_vol_sum_20")
        )
    if "_total_vol_sum_20" in missing:
        exprs.append(pl.col("volume").rolling_sum(20).over("code").alias("_total_vol_sum_20"))
    if "_ewm_pv_20" in missing:
        exprs.append(pl.col("_tp_v").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_pv_20"))
    if "_ewm_v_20" in missing:
        exprs.append(pl.col("volume").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_v_20"))
    if "_cgo_rp" in missing:
        vwap = pl.col("amount") / pl.col("volume")
        exprs.append(
            ((pl.col("_cgo_w") * vwap).rolling_sum(100).over("code")
             / pl.col("_cgo_w").rolling_sum(100).over("code")).alias("_cgo_rp")
        )
    if "_qm_sigma_60d" in missing:
        exprs.append(
            pl.col("_ret").rolling_std(60).over("code").alias("_qm_sigma_60d")
        )
    return cols, exprs


# ═══════════════════════════════════════════════════════════════════════════
# Factor registry
# ═══════════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY: dict[str, dict] = {
    # ── trend family (active) ──
    "price_pos_20d": {
        "label": "20日价格位置",
        "family": "trend",
        "status": "active",
        "requires": {"_c_min_20", "_c_max_20"},
        "expr": (
            (pl.col("close_adj") - pl.col("_c_min_20"))
            / pl.max_horizontal(pl.col("_c_max_20") - pl.col("_c_min_20"), pl.lit(0.01))
        ),
    },
    "close_to_high_20d": {
        "label": "接近20日新高",
        "family": "trend",
        "status": "active",
        "requires": {"_high_20d"},
        "expr": 1 - pl.col("close_adj") / pl.max_horizontal(pl.col("_high_20d"), pl.lit(0.01)),
    },
    "KLEN": {
        "label": "K线振幅收缩",
        "family": "trend",
        "status": "active",
        "requires": set(),
        "expr": (pl.col("high_adj") - pl.col("low_adj")) / _open_den,
    },
    "KMID2": {
        "label": "实体占比偏强",
        "family": "trend",
        "status": "active",
        "requires": set(),
        "expr": (pl.col("close_adj") - pl.col("open_adj")) / _range_den,
    },

    # ── momentum (active) ──
    "ret_5d": {
        "label": "5日动量",
        "family": "momentum",
        "status": "active",
        "requires": set(),  # ret_5d intermediate auto-computed
        "expr": pl.col("ret_5d"),
    },
    "ret_20d": {
        "label": "20日动量",
        "family": "momentum",
        "status": "active",
        "requires": set(),
        "expr": pl.col("ret_20d"),
    },

    # ── pullback family (active) ──
    "ma_bias_20": {
        "label": "20日均线偏离",
        "family": "pullback",
        "status": "active",
        "requires": {"_ma20"},
        "expr": (
            (pl.col("close_adj") - pl.col("_ma20"))
            / pl.max_horizontal(pl.col("_ma20"), pl.lit(0.01)) * 100
        ),
    },
    "disp_bias_20": {
        "label": "20日成本偏离",
        "family": "pullback",
        "status": "active",
        "requires": {"_ewm_pv_20", "_ewm_v_20"},
        "expr": (
            pl.col("close_adj")
            / pl.max_horizontal(
                pl.col("_ewm_pv_20") / pl.max_horizontal(pl.col("_ewm_v_20"), pl.lit(1e-10)),
                pl.lit(0.01),
            )
            - 1
        ),
    },
    "intraday_pos": {
        "label": "日内收盘位置",
        "family": "pullback",
        "status": "active",
        "requires": set(),
        "expr": (
            (pl.col("close_adj") - pl.col("low_adj"))
            / pl.max_horizontal(pl.col("high_adj") - pl.col("low_adj"), pl.lit(1e-8))
        ),
    },
    "KSFT": {
        "label": "K线实体重心",
        "family": "pullback",
        "status": "active",
        "requires": set(),
        "expr": (
            (2.0 * pl.col("close_adj") - pl.col("high_adj") - pl.col("low_adj")) / _open_den
        ),
    },

    # ── risk (active) ──
    "atr_14_pct": {
        "label": "ATR 风险",
        "family": "risk",
        "status": "active",
        "requires": {"_atr14"},
        "expr": pl.col("_atr14") / pl.max_horizontal(pl.col("close_adj"), pl.lit(0.01)),
    },
    "panic_vol_ratio_20d": {
        "label": "恐慌量比",
        "family": "risk",
        "status": "active",
        "requires": {"_down_vol_sum_20", "_total_vol_sum_20"},
        "expr": pl.col("panic_vol_ratio_20d"),
    },

    # ── experimental ──
    "cgo_100d": {
        "label": "CGO 处置效应 (100d RP)",
        "family": "behavioral",
        "status": "experimental",
        "requires": {"_cgo_rp"},
        "expr": pl.col("close_adj") / pl.col("_cgo_rp") - 1.0,
        "note": "IC=-0.052 IR=-0.43, 低CGO反转效应",
    },
    "terrified_score": {
        "label": "Terrified Score (原始版)",
        "family": "reversal",
        "status": "experimental",
        "requires": {"_terr_avg_20d", "_terr_std_20d"},
        "expr": (pl.col("_terr_avg_20d") + pl.col("_terr_std_20d")) * 0.5,
        "note": "IC=-0.085 IR=-0.64 ⭐ 最强IC",
    },
    "quality_momentum": {
        "label": "高质量动量 (r_60 - K×σ²)",
        "family": "momentum",
        "status": "experimental",
        "requires": {"_qm_ret_60d", "_qm_sigma_60d"},
        "expr": pl.col("_qm_ret_60d")
                - 3000.0 * pl.col("_qm_sigma_60d").pow(2),
        "note": "IC=0.064 IR=0.33, 风险调整动量，与市值相关仅0.085",
    },
    "ma_convergence_pcf": {
        "label": "MA 收敛 PCF (多周期均线收敛度)",
        "family": "trend",
        "status": "experimental",
        "requires": {"_ma5", "_ma10", "_ma20", "_ma60"},
        "expr": -(
            (pl.col("_ma5") - pl.col("_ma20")).abs()
            + (pl.col("_ma10") - pl.col("_ma20")).abs()
            + (pl.col("_ma60") - pl.col("_ma20")).abs()
        ) / 3.0,
        "note": "IC~0.05, 突破前兆识别，高PCF+上升趋势=蓄力阶段",
    },
    "stv_score_20d": {
        "label": "STV Terrified Score 量价变体",
        "family": "reversal",
        "status": "experimental",
        "requires": {"_stv_avg_20d", "_stv_std_20d"},
        "expr": (pl.col("_stv_avg_20d") + pl.col("_stv_std_20d")) * 0.5,
        "note": "IC=-0.067 IR=-0.40, 凸显反转",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def compute_required_factors(
    frame: pl.LazyFrame, factor_names: list[str]
) -> pl.LazyFrame:
    """Return frame with the requested factor columns + shared intermediates.

    Only the needed columns and their transitive dependencies are computed.
    New factors only need a FACTOR_REGISTRY entry — this function never changes.
    """
    if not factor_names:
        return frame

    # 1. collect all transitive intermediates
    intermediates: set[str] = set()
    for name in factor_names:
        spec = FACTOR_REGISTRY.get(name)
        if spec is None:
            continue
        intermediates |= spec.get("requires", set())

    # 2. compute shared intermediates (tiered, each tier only refs previous)
    frame = _ensure_intermediates(frame, intermediates)

    # 3. compute final factor expressions
    exprs: list[pl.Expr] = []
    for name in factor_names:
        spec = FACTOR_REGISTRY.get(name)
        if spec is None:
            continue
        exprs.append(spec["expr"].alias(name))

    if exprs:
        frame = frame.with_columns(exprs)

    return frame


def active_factors(family: str | None = None) -> list[str]:
    """Return names of all active factors, optionally filtered by family."""
    result = []
    for name, spec in FACTOR_REGISTRY.items():
        if spec.get("status") != "active":
            continue
        if family and spec.get("family") != family:
            continue
        result.append(name)
    return result


def experimental_factors() -> list[str]:
    """Return names of experimental factors (for research exploration)."""
    return [
        name for name, spec in FACTOR_REGISTRY.items()
        if spec.get("status") == "experimental"
    ]
