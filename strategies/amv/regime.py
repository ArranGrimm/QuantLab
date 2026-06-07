from __future__ import annotations

import polars as pl

from utils.active_market_value_regime import build_active_market_value_regime_frame, load_active_market_value_frame


FWD_DURATION_BUCKETS = {
    "fresh": (1, 5),
    "young": (6, 15),
    "aged": (16, 30),
    "old": (31, 999),
}

FWD_MOMENTUM_BUCKETS = {
    "accelerating": (1.5, 999),
    "cruising": (0, 1.5),
    "stalling": (-1.5, 0),
    "retreating": (-999, -1.5),
}


def build_amv_phase_frame(
    bull_trigger_pct: float = 4.0,
    bear_trigger_1d_pct: float = -2.3,
    bull_lookback_days: int = 2,
    effective_lag_days: int = 1,
) -> pl.DataFrame:
    df_regime = build_active_market_value_regime_frame(
        bull_trigger_pct=bull_trigger_pct,
        bear_trigger_1d_pct=bear_trigger_1d_pct,
        bull_lookback_days=bull_lookback_days,
        effective_lag_days=effective_lag_days,
    )

    df_amv = load_active_market_value_frame().select(
        [
            pl.col("trade_date").alias("date"),
            pl.col("amv_close"),
            pl.col("amv_high"),
            pl.col("amplitude_pct"),
            pl.col("turnover_pct"),
        ]
    )

    df = (
        df_regime.join(df_amv, on="date", how="left")
        .sort("date")
        .with_columns((pl.col("amv_close") / pl.col("amv_close").shift(1) - 1).alias("amv_ret_1d_raw"))
    )

    is_bull = df["is_bull_regime"].to_list()
    regime_ids: list[int | None] = []
    current_id = 0
    in_regime = False
    for is_bull_day in is_bull:
        if is_bull_day and not in_regime:
            current_id += 1
            in_regime = True
        elif not is_bull_day:
            in_regime = False
        regime_ids.append(current_id if is_bull_day else None)

    df = df.with_columns(pl.Series("regime_id", regime_ids))
    regime_durations = (
        df.filter(pl.col("regime_id").is_not_null()).group_by("regime_id").len(name="regime_total_days")
    )
    df = df.join(regime_durations, on="regime_id", how="left").sort("date")
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then(pl.col("regime_id").cum_count().over("regime_id"))
        .otherwise(None)
        .alias("regime_duration_days")
    )
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then(pl.col("regime_duration_days").cast(pl.Float64) / pl.col("regime_total_days").cast(pl.Float64))
        .otherwise(None)
        .alias("regime_progress")
    )
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_null())
        .then(pl.lit("non_bull"))
        .when(pl.col("regime_total_days") < 5)
        .then(pl.lit("pulse"))
        .when(pl.col("regime_progress") < 0.25)
        .then(pl.lit("early"))
        .when(pl.col("regime_progress") < 0.75)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("hindsight_phase")
    )

    df = df.with_columns(
        [
            pl.col("amv_ret_1d_raw").alias("amv_ret_1d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(5) - 1) * 100).alias("amv_slope_5d"),
            ((pl.col("amv_close") / pl.col("amv_close").shift(20) - 1) * 100).alias("amv_slope_20d"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then((pl.col("amv_ret_1d_raw") + 1).cum_prod().over("regime_id") - 1)
        .otherwise(None)
        .alias("amv_ret_from_start")
    )
    df = df.with_columns(
        [
            (pl.col("amv_slope_5d") - pl.col("amv_slope_20d")).alias("amv_acceleration"),
            pl.col("amv_ret_1d").rolling_mean(3).alias("amv_ret_ma3"),
            pl.col("amv_ret_1d").rolling_mean(5).alias("amv_ret_ma5"),
            pl.col("amplitude_pct").rolling_mean(3).alias("amplitude_ma3"),
        ]
    )

    neg_streak = [0]
    for idx in range(1, df.height):
        value = df["amv_ret_1d"][idx]
        if value is not None and value < 0:
            neg_streak.append(neg_streak[-1] + 1)
        else:
            neg_streak.append(0)
    neg_streak[0] = 1 if (df["amv_ret_1d"][0] is not None and df["amv_ret_1d"][0] < 0) else 0
    df = df.with_columns(pl.Series("amv_neg_streak", neg_streak))

    regime_cummax = (
        df.filter(pl.col("regime_id").is_not_null())
        .with_columns(pl.col("amv_high").cum_max().over("regime_id").alias("regime_amv_high_sofar"))
        .select(["date", "regime_amv_high_sofar"])
    )
    df = df.join(regime_cummax, on="date", how="left")
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then((pl.col("amv_close") / pl.col("regime_amv_high_sofar") - 1) * 100)
        .otherwise(None)
        .alias("amv_dd_from_high")
    )

    all_durations = (
        df.filter(pl.col("regime_id").is_not_null())
        .group_by("regime_id")
        .len(name="d")
        .select("d")
        .to_series()
        .sort()
        .to_list()
    )
    n_regimes = len(all_durations)

    def _maturity_pct(duration: int | None) -> float | None:
        if duration is None:
            return None
        shorter = sum(1 for item in all_durations if item < duration)
        return round(shorter / n_regimes * 100, 1)

    maturity_map = {duration: _maturity_pct(duration) for duration in range(1, 100)}
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then(pl.col("regime_duration_days").replace_strict(maturity_map, default=None))
        .otherwise(None)
        .alias("regime_maturity")
    )

    fwd_dur = pl.lit("unknown")
    for label, (lo, hi) in FWD_DURATION_BUCKETS.items():
        fwd_dur = (
            pl.when(pl.col("regime_id").is_not_null() & pl.col("regime_duration_days").is_between(lo, hi, closed="both"))
            .then(pl.lit(label))
            .otherwise(fwd_dur)
        )
    fwd_dur = pl.when(pl.col("regime_id").is_null()).then(pl.lit("non_bull")).otherwise(fwd_dur)
    df = df.with_columns(fwd_dur.alias("fwd_duration_bucket"))

    fwd_mom = pl.lit("unknown")
    for label, (lo, hi) in FWD_MOMENTUM_BUCKETS.items():
        fwd_mom = (
            pl.when(pl.col("regime_id").is_not_null() & pl.col("amv_slope_5d").is_between(lo, hi, closed="left"))
            .then(pl.lit(label))
            .otherwise(fwd_mom)
        )
    fwd_mom = pl.when(pl.col("regime_id").is_null()).then(pl.lit("non_bull")).otherwise(fwd_mom)
    df = df.with_columns(fwd_mom.alias("fwd_momentum_bucket"))
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_null())
        .then(pl.lit("non_bull"))
        .otherwise(pl.concat_str([pl.col("fwd_duration_bucket"), pl.col("fwd_momentum_bucket")], separator="_"))
        .alias("fwd_phase")
    )

    entry_close = (
        df.filter(pl.col("regime_duration_days") == 1)
        .select(["regime_id", pl.col("amv_close").alias("regime_entry_amv_close")])
    )
    df = df.join(entry_close, on="regime_id", how="left")

    for col_name in ["amv_slope_5d", "amv_slope_20d", "amv_acceleration", "amv_dd_from_high"]:
        df = df.with_columns(pl.when(pl.col(col_name).is_finite()).then(pl.col(col_name)).otherwise(None).alias(col_name))

    return df.select(
        [
            "date",
            "is_bull_regime",
            "amv_mechanical_regime",
            "amv_close",
            "regime_id",
            "regime_duration_days",
            "regime_total_days",
            "regime_progress",
            "hindsight_phase",
            "fwd_duration_bucket",
            "fwd_momentum_bucket",
            "fwd_phase",
            "amv_ret_from_start",
            "amv_ret_1d",
            "amv_slope_5d",
            "amv_slope_20d",
            "amv_acceleration",
            "amv_dd_from_high",
            "regime_maturity",
            "amv_ret_ma3",
            "amv_ret_ma5",
            "amplitude_ma3",
            "amv_neg_streak",
            "amplitude_pct",
            "turnover_pct",
            "regime_entry_amv_close",
        ]
    )


def build_amv_regime_gate_frame(
    *,
    bull_trigger_pct: float = 4.0,
    bear_trigger_1d_pct: float = -2.3,
    bull_lookback_days: int = 2,
    effective_lag_days: int = 1,
) -> pl.DataFrame:
    phase = build_amv_phase_frame(
        bull_trigger_pct=bull_trigger_pct,
        bear_trigger_1d_pct=bear_trigger_1d_pct,
        bull_lookback_days=bull_lookback_days,
        effective_lag_days=effective_lag_days,
    )
    aged_non_accel = (
        (pl.col("fwd_duration_bucket") == "aged")
        & pl.col("fwd_momentum_bucket").is_in(["cruising", "stalling", "retreating"])
    )
    chaos = (pl.col("amv_neg_streak") >= 3) & (pl.col("amplitude_pct") > 2.5)
    return (
        phase.select(
            [
                "date",
                "fwd_duration_bucket",
                "fwd_momentum_bucket",
                "fwd_phase",
                "amv_neg_streak",
                "amplitude_pct",
            ]
        )
        .with_columns(
            [
                aged_non_accel.alias("gate_aged_non_accel"),
                chaos.alias("gate_chaos"),
                (aged_non_accel | chaos).alias("gate_skip"),
            ]
        )
        .rename({"date": "signal_date"})
    )
