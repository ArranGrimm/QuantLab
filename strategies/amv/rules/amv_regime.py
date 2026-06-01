"""
AMV regime gate — 基于活跃市值内部阶段的风控规则。

规则来源
--------
AMV 牛市内部阶段诊断 (2026-05-26):
- 事后分析发现 P3 在 AMV late 阶段亏损 -221K，说明在牛市晚期避险有理论空间。
- 但前向 AMV 特征（age/maturity/progress）无法为 P3 构建正向 gating——P3 的 6d 静态 cadence
  与混沌期时间窗口不匹配。
- PB3 rolling 的 UNION 规则 (aged+非加速 OR 混沌期) 在 trade-level 跳过 258/1650 笔，
  net delta +23.2K，walk-forward 合计仍为正。

Rust 验证 (2026-05-26):
  raw PB3 rolling:  +99.62% (adj) / +80.55% (raw), MaxDD 20.70% (adj) / 11.75% (raw)
  gated:            +109.73% (adj) / +80.55% (raw), MaxDD 16.20% (adj) / 11.75% (raw)

稳健性 (2026-05-28):
  方向有效但不是年年稳定——核心贡献集中在 2022/2023 的 AMV 老化/混沌段，
  2025/2026 trade-level 为负贡献。不宜激进加码。

规则定义
--------
- aged_non_accel: AMV bull regime 已持续 16-30 天（aged），且 5 日涨速非加速（cruising/stalling/retreating）
- chaos: AMV 连续下跌 >= 3 天，且振幅 > 2.5%
- skip = aged_non_accel OR chaos

适用策略
--------
known_compatible: ["pullback-pb3"]
known_incompatible: ["trend-p2", "trend-p3", "trend-p3-enhanced"]  # 6d 静态 cadence 不匹配
"""

from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.regime import build_amv_regime_gate_frame
from strategies.amv.signals import build_backtest_signal_frame


def apply_amv_regime_gate(
    *,
    market: pl.DataFrame,
    signal_rows: pl.DataFrame,
    config: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """应用 AMV 风控 gate。由 workflows.py 在 strategy.rules 包含 'amv-regime-gate' 时调用。"""

    gate = build_amv_regime_gate_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    )
    before_rows = signal_rows.height
    before_days = signal_rows.select("signal_date").n_unique()
    gated_rows = signal_rows.join(gate, on="signal_date", how="left").with_columns(
        [
            pl.col("gate_skip").fill_null(False),
            pl.col("gate_aged_non_accel").fill_null(False),
            pl.col("gate_chaos").fill_null(False),
        ]
    )
    blocked = gated_rows.filter(pl.col("gate_skip"))
    kept = gated_rows.filter(~pl.col("gate_skip")).sort(["signal_date", "rank", "code"])
    export = build_backtest_signal_frame(market, kept)
    summary = {
        "amv_regime_gate": "aged_non_accel_or_chaos",
        "amv_regime_gate_applied": True,
        "gate_timing": "signal_date_close_before_t_plus_1_open",
        "gate_rows_before": before_rows,
        "gate_rows_after": kept.height,
        "gate_rows_blocked": blocked.height,
        "gate_days_before": before_days,
        "gate_days_after": kept.select("signal_date").n_unique(),
        "gate_days_blocked": blocked.select("signal_date").n_unique(),
        "gate_aged_non_accel_rows": int(blocked["gate_aged_non_accel"].sum()),
        "gate_chaos_rows": int(blocked["gate_chaos"].sum()),
    }
    return kept, export, summary
