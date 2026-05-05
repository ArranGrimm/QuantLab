"""Mechanical regime helpers for Active Market Value (0AMV).

The 0AMV data source is maintained outside QMT. This module turns the daily
AMV OHLC series into a market-level bull/bear state that can be joined into
strategy signal exports, especially Rotation's `is_bull_regime` gate.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMV_DB = ROOT.parent / "QuantData" / "Ashare" / "active_market_value.duckdb"


def load_active_market_value_frame(
    db_path: str | Path = DEFAULT_AMV_DB,
) -> pl.DataFrame:
    """Load the canonical Active Market Value daily OHLC table."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute(
            """
            SELECT
                trade_date,
                amv_open,
                amv_high,
                amv_low,
                amv_close,
                chg_abs_pct,
                volume_100m,
                amount_100m,
                position_100m,
                turnover_pct,
                amplitude_pct,
                quality_flags
            FROM active_market_value
            ORDER BY trade_date
            """
        ).pl()
    finally:
        conn.close()


def build_active_market_value_regime_frame(
    db_path: str | Path = DEFAULT_AMV_DB,
    *,
    bull_trigger_pct: float = 4.0,
    bull_lookback_days: int = 2,
    bear_trigger_1d_pct: float = -2.3,
    effective_lag_days: int = 1,
    date_col: str = "date",
) -> pl.DataFrame:
    """Build a daily mechanical AMV regime frame.

    Current exploratory rule:
    - bull trigger: max(ret_1d...ret_Nd) >= `bull_trigger_pct`
    - bear trigger: ret_1d <= `bear_trigger_1d_pct`
    - effective regime: observed regime shifted by `effective_lag_days`

    The output contains one row per AMV trading day and is intended to be joined
    onto strategy data by date. Triggers are observed after the AMV close of the
    trigger day, so the tradable `is_bull_regime` must not become true on that
    same row.
    """
    if bull_lookback_days < 1:
        raise ValueError("bull_lookback_days must be >= 1")
    if effective_lag_days < 0:
        raise ValueError("effective_lag_days must be >= 0")

    df_amv = load_active_market_value_frame(db_path).with_columns(
        [
            ((pl.col("amv_close") / pl.col("amv_close").shift(1) - 1) * 100).alias("amv_ret_1d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(2) - 1) * 100).alias("amv_ret_2d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(3) - 1) * 100).alias("amv_ret_3d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(5) - 1) * 100).alias("amv_ret_5d"),
        ]
    )

    bull_ret_cols = [f"amv_ret_{i}d" for i in range(1, bull_lookback_days + 1)]
    df_regime = (
        df_amv.with_columns(
            pl.max_horizontal(*bull_ret_cols).alias("amv_bull_trigger_ret")
        )
        .with_columns(
            [
                (pl.col("amv_bull_trigger_ret") >= bull_trigger_pct)
                .fill_null(False)
                .alias("amv_bull_trigger"),
                (pl.col("amv_ret_1d") <= bear_trigger_1d_pct)
                .fill_null(False)
                .alias("amv_bear_trigger"),
            ]
        )
    )

    state = "neutral"
    observed_states: list[str] = []
    for row in df_regime.select(["amv_bull_trigger", "amv_bear_trigger"]).to_dicts():
        if row["amv_bear_trigger"]:
            state = "bear"
        if row["amv_bull_trigger"]:
            state = "bull"
        observed_states.append(state)

    effective_states = (
        ["neutral"] * effective_lag_days + observed_states[:-effective_lag_days]
        if effective_lag_days
        else observed_states
    )

    return (
        df_regime.with_columns(
            [
                pl.Series("amv_observed_regime", observed_states),
                (pl.Series("amv_observed_regime", observed_states) == "bull").alias(
                    "amv_observed_bull_regime"
                ),
                pl.Series("amv_mechanical_regime", effective_states),
                (pl.Series("amv_mechanical_regime", effective_states) == "bull").alias(
                    "is_bull_regime"
                ),
            ]
        )
        .rename({"trade_date": date_col})
        .select(
            [
                date_col,
                "is_bull_regime",
                "amv_mechanical_regime",
                "amv_observed_bull_regime",
                "amv_observed_regime",
                "amv_bull_trigger",
                "amv_bear_trigger",
                "amv_ret_1d",
                "amv_ret_2d",
                "amv_ret_3d",
                "amv_ret_5d",
                "amv_bull_trigger_ret",
                "amv_close",
            ]
        )
    )
