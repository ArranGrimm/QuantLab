from __future__ import annotations

from typing import Any

import polars as pl

from strategies.amv.regime import build_pb3_regime_gate_frame
from strategies.amv.signals import build_backtest_signal_frame


def apply_pb3_regime_gate(
    *,
    market: pl.DataFrame,
    signal_rows: pl.DataFrame,
    config: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    if config.pb3_regime_gate == "none":
        export = build_backtest_signal_frame(market, signal_rows)
        return signal_rows, export, {
            "pb3_regime_gate": config.pb3_regime_gate,
            "pb3_regime_gate_applied": False,
        }
    if config.pb3_regime_gate != "aged_non_accel_or_chaos":
        raise ValueError(f"unknown PB3 regime gate: {config.pb3_regime_gate}")

    pb3_gate = build_pb3_regime_gate_frame(
        bull_trigger_pct=config.amv_bull_trigger_pct,
        bear_trigger_1d_pct=config.amv_bear_trigger_1d_pct,
        bull_lookback_days=config.amv_bull_lookback_days,
        effective_lag_days=config.amv_effective_lag_days,
    )
    before_rows = signal_rows.height
    before_days = signal_rows.select("signal_date").n_unique()
    gated_rows = signal_rows.join(pb3_gate, on="signal_date", how="left").with_columns(
        [
            pl.col("pb3_gate_skip").fill_null(False),
            pl.col("pb3_gate_aged_non_accel").fill_null(False),
            pl.col("pb3_gate_chaos").fill_null(False),
        ]
    )
    blocked = gated_rows.filter(pl.col("pb3_gate_skip"))
    kept = gated_rows.filter(~pl.col("pb3_gate_skip")).sort(["signal_date", "rank", "code"])
    export = build_backtest_signal_frame(market, kept)
    summary = {
        "pb3_regime_gate": config.pb3_regime_gate,
        "pb3_regime_gate_applied": True,
        "pb3_gate_timing": "signal_date_close_before_t_plus_1_open",
        "pb3_gate_rows_before": before_rows,
        "pb3_gate_rows_after": kept.height,
        "pb3_gate_rows_blocked": blocked.height,
        "pb3_gate_days_before": before_days,
        "pb3_gate_days_after": kept.select("signal_date").n_unique(),
        "pb3_gate_days_blocked": blocked.select("signal_date").n_unique(),
        "pb3_gate_aged_non_accel_rows": int(blocked["pb3_gate_aged_non_accel"].sum()),
        "pb3_gate_chaos_rows": int(blocked["pb3_gate_chaos"].sum()),
    }
    return kept, export, summary
