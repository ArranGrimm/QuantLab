"""AMV regime internal phase diagnostic – hindsight + forward-observable.

Two phase systems:
  (A) hindsight (for diagnosis only): early/mid/late based on regime_progress
      (uses regime_total_days – NOT available at entry time)
  (B) forward-observable (actionable): 2-D grid of
      duration (fresh/young/aged/old) × momentum (accelerating/cruising/stalling/retreating)
      using only data available at entry day.

Key question: can forward-observable features capture the "late-bull destruction"
pattern seen in the hindsight phases?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from utils.active_market_value_regime import (
    build_active_market_value_regime_frame,
    load_active_market_value_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "amv_regime_phase_diagnostic.json"

DEFAULT_SLEEVES: dict[str, dict[str, str]] = {
    "P3_static_strict": {
        "label": "P3/K0.5/R0 static strict Top3",
        "trades": "artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092208_801/trades.csv",
        "kind": "static",
    },
    "PB3_rolling_refill": {
        "label": "PB3/CP1/RV0 rolling21 refill Top10",
        "trades": "artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/backtests/6td_rolling21_refill_top10_no_stop_20260521_091007_830/trades.csv",
        "kind": "rolling",
    },
    "Ref_static_strict": {
        "label": "Ref/P2/K0.5/R0 static strict Top3",
        "trades": "artifacts/amv_static_sleeve_signals/20260520_092047_reference_p2_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092131_677/trades.csv",
        "kind": "static",
    },
}

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


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_amv_phase_frame(
    bull_trigger_pct: float = 4.0,
    bear_trigger_1d_pct: float = -2.3,
    bull_lookback_days: int = 2,
    effective_lag_days: int = 1,
) -> pl.DataFrame:
    """Build daily AMV regime frame with both hindsight and forward-observable phases."""
    df_regime = build_active_market_value_regime_frame(
        bull_trigger_pct=bull_trigger_pct,
        bear_trigger_1d_pct=bear_trigger_1d_pct,
        bull_lookback_days=bull_lookback_days,
        effective_lag_days=effective_lag_days,
    )

    df_amv = load_active_market_value_frame().select([
        pl.col("trade_date").alias("date"),
        pl.col("amv_close"),
        pl.col("amv_high"),
        pl.col("amplitude_pct"),
        pl.col("turnover_pct"),
    ])

    df = (
        df_regime.join(df_amv, on="date", how="left")
        .sort("date")
        .with_columns(
            (pl.col("amv_close") / pl.col("amv_close").shift(1) - 1).alias("amv_ret_1d_raw"),
        )
    )

    # ── contiguous regime ids ────────────────────────────────────────────
    is_bull = df["is_bull_regime"].to_list()
    regime_ids: list[int | None] = []
    current_id = 0
    in_regime = False
    for b in is_bull:
        if b and not in_regime:
            current_id += 1
            in_regime = True
        elif not b:
            in_regime = False
        regime_ids.append(current_id if b else None)

    df = df.with_columns(pl.Series("regime_id", regime_ids))

    regime_durations = (
        df.filter(pl.col("regime_id").is_not_null())
        .group_by("regime_id")
        .len(name="regime_total_days")
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
        .then(
            pl.col("regime_duration_days").cast(pl.Float64)
            / pl.col("regime_total_days").cast(pl.Float64)
        )
        .otherwise(None)
        .alias("regime_progress")
    )

    # ── (A) hindsight phase ──────────────────────────────────────────────
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

    # ── AMV return series ───────────────────────────────────────────────
    df = df.with_columns(
        pl.col("amv_ret_1d_raw").alias("amv_ret_1d"),
        ((pl.col("amv_close") / pl.col("amv_close").shift(5) - 1) * 100).alias("amv_slope_5d"),
        ((pl.col("amv_close") / pl.col("amv_close").shift(20) - 1) * 100).alias("amv_slope_20d"),
    )

    # cumulative return since regime start
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then((pl.col("amv_ret_1d_raw") + 1).cum_prod().over("regime_id") - 1)
        .otherwise(None)
        .alias("amv_ret_from_start")
    )

    # ── forward-observable features ────────────────────────────────────
    df = df.with_columns(
        (pl.col("amv_slope_5d") - pl.col("amv_slope_20d")).alias("amv_acceleration"),
        pl.col("amv_ret_1d").rolling_mean(3).alias("amv_ret_ma3"),
        pl.col("amv_ret_1d").rolling_mean(5).alias("amv_ret_ma5"),
        pl.col("amplitude_pct").rolling_mean(3).alias("amplitude_ma3"),
    )

    # consecutive negative ret_1d streak
    neg_streak = [0]
    for i in range(1, df.height):
        val = df["amv_ret_1d"][i]
        if val is not None and val < 0:
            neg_streak.append(neg_streak[-1] + 1)
        else:
            neg_streak.append(0)
    neg_streak[0] = 1 if (df["amv_ret_1d"][0] is not None and df["amv_ret_1d"][0] < 0) else 0
    df = df.with_columns(pl.Series("amv_neg_streak", neg_streak))

    # AMV drawdown from regime high (all-time regime high, known at entry)
    regime_cummax = (
        df.filter(pl.col("regime_id").is_not_null())
        .with_columns(
            pl.col("amv_high").cum_max().over("regime_id").alias("regime_amv_high_sofar")
        )
        .select(["date", "regime_amv_high_sofar"])
    )
    df = df.join(regime_cummax, on="date", how="left")
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then((pl.col("amv_close") / pl.col("regime_amv_high_sofar") - 1) * 100)
        .otherwise(None)
        .alias("amv_dd_from_high")
    )

    # regime maturity: empirical survival CDF (forward-observable)
    # "Of all historical bull regimes, what fraction die before reaching N days?"
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

    def _maturity_pct(d: int | None) -> float | None:
        if d is None:
            return None
        shorter = sum(1 for x in all_durations if x < d)
        return round(shorter / n_regimes * 100, 1)

    maturity_map = {d: _maturity_pct(d) for d in range(1, 100)}
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_not_null())
        .then(
            pl.col("regime_duration_days").replace_strict(maturity_map, default=None)
        )
        .otherwise(None)
        .alias("regime_maturity")
    )

    # (B) forward-observable duration bucket
    fwd_dur = pl.lit("unknown")
    for label, (lo, hi) in FWD_DURATION_BUCKETS.items():
        fwd_dur = (
            pl.when(
                pl.col("regime_id").is_not_null()
                & pl.col("regime_duration_days").is_between(lo, hi, closed="both")
            )
            .then(pl.lit(label))
            .otherwise(fwd_dur)
        )
    fwd_dur = pl.when(pl.col("regime_id").is_null()).then(pl.lit("non_bull")).otherwise(fwd_dur)
    df = df.with_columns(fwd_dur.alias("fwd_duration_bucket"))

    # ── (B) forward-observable momentum bucket ──────────────────────────
    fwd_mom = pl.lit("unknown")
    for label, (lo, hi) in FWD_MOMENTUM_BUCKETS.items():
        fwd_mom = (
            pl.when(
                pl.col("regime_id").is_not_null()
                & pl.col("amv_slope_5d").is_between(lo, hi, closed="left")
            )
            .then(pl.lit(label))
            .otherwise(fwd_mom)
        )
    fwd_mom = pl.when(pl.col("regime_id").is_null()).then(pl.lit("non_bull")).otherwise(fwd_mom)
    df = df.with_columns(fwd_mom.alias("fwd_momentum_bucket"))

    # ── (B) combined forward phase label ─────────────────────────────────
    df = df.with_columns(
        pl.when(pl.col("regime_id").is_null())
        .then(pl.lit("non_bull"))
        .otherwise(
            pl.concat_str(
                [pl.col("fwd_duration_bucket"), pl.col("fwd_momentum_bucket")],
                separator="_",
            )
        )
        .alias("fwd_phase")
    )

    # regime entry AMV close
    entry_close = (
        df.filter(pl.col("regime_duration_days") == 1)
        .select(["regime_id", pl.col("amv_close").alias("regime_entry_amv_close")])
    )
    df = df.join(entry_close, on="regime_id", how="left")

    # clean: NaN/Inf -> null for slope columns
    for col in ["amv_slope_5d", "amv_slope_20d", "amv_acceleration", "amv_dd_from_high"]:
        df = df.with_columns(
            pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None).alias(col)
        )

    return df.select([
        "date", "is_bull_regime", "amv_mechanical_regime", "amv_close",
        "regime_id", "regime_duration_days", "regime_total_days", "regime_progress",
        "hindsight_phase",
        "fwd_duration_bucket", "fwd_momentum_bucket", "fwd_phase",
        "amv_ret_from_start", "amv_ret_1d", "amv_slope_5d", "amv_slope_20d",
        "amv_acceleration", "amv_dd_from_high", "regime_maturity",
        "amv_ret_ma3", "amv_ret_ma5", "amplitude_ma3", "amv_neg_streak",
        "amplitude_pct", "turnover_pct", "regime_entry_amv_close",
    ])


def load_trades(trades_path: Path) -> pl.DataFrame:
    return pl.read_csv(
        str(trades_path),
        schema_overrides={
            "code": pl.Utf8, "entry_date": pl.Utf8, "exit_date": pl.Utf8,
            "entry_price": pl.Float64, "exit_price": pl.Float64,
            "shares": pl.Int64, "cost": pl.Float64, "exit_value": pl.Float64,
            "pnl": pl.Float64, "pnl_pct": pl.Float64,
            "hold_trading_days": pl.Int64, "exit_reason": pl.Utf8,
        },
    ).with_columns(
        pl.col("entry_date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("exit_date").str.strptime(pl.Date, "%Y-%m-%d"),
    )


def _phase_stats(phase_trades: pl.DataFrame, total_n: int) -> dict[str, Any]:
    n = phase_trades.height
    if n == 0:
        return {
            "trades": 0, "total_pnl": 0.0, "avg_pnl_pct": None,
            "win_rate": None, "avg_hold_days": None, "trade_share_pct": 0.0,
        }
    win_n = phase_trades.filter(pl.col("pnl") > 0).height
    return {
        "trades": n,
        "total_pnl": round(float(phase_trades["pnl"].sum()), 2),
        "avg_pnl_pct": round(float(phase_trades["pnl_pct"].mean()), 6),
        "win_rate": round(win_n / n, 4),
        "avg_hold_days": round(float(phase_trades["hold_trading_days"].mean()), 1),
        "trade_share_pct": round(n / total_n * 100, 1),
    }


def _context_stats(phase_trades: pl.DataFrame) -> dict[str, Any]:
    """AMV environment stats for trades in this phase (all forward-observable)."""
    return {
        "avg_amv_ret_from_start_pct": round(
            float(phase_trades["amv_ret_from_start"].mean() or 0) * 100, 2
        ),
        "avg_amv_slope_5d_pct": round(float(phase_trades["amv_slope_5d"].mean() or 0), 2),
        "avg_amv_slope_20d_pct": round(float(phase_trades["amv_slope_20d"].mean() or 0), 2),
        "avg_amv_acceleration_pct": round(float(phase_trades["amv_acceleration"].mean() or 0), 2),
        "avg_amv_dd_from_high_pct": round(float(phase_trades["amv_dd_from_high"].mean() or 0), 2),
        "avg_regime_duration_days": round(
            float(phase_trades["regime_duration_days"].mean() or 0), 1
        ),
        "avg_regime_maturity": round(
            float(phase_trades["regime_maturity"].mean() or 0), 1
        ),
    }


def compute_stats(
    trades: pl.DataFrame,
    amv_phase: pl.DataFrame,
    sleeve_name: str,
) -> dict[str, Any]:
    """Compute per-phase stats for both hindsight and forward phase systems."""

    join_cols = [
        "date", "hindsight_phase", "fwd_phase",
        "fwd_duration_bucket", "fwd_momentum_bucket",
        "regime_duration_days", "regime_total_days", "regime_progress",
        "amv_ret_from_start", "amv_ret_1d",
        "amv_slope_5d", "amv_slope_20d", "amv_acceleration", "amv_dd_from_high",
        "regime_maturity",
        "amv_ret_ma3", "amv_ret_ma5", "amplitude_ma3", "amv_neg_streak",
        "amplitude_pct", "turnover_pct",
    ]
    joined = trades.join(amv_phase.select(join_cols), left_on="entry_date", right_on="date", how="left")

    total_n = joined.height

    result: dict[str, Any] = {
        "sleeve_name": sleeve_name,
        "total_trades": total_n,
        "total_pnl": float(joined["pnl"].sum()),
    }

    # ── (A) hindsight ───────────────────────────────────────────────────
    result["hindsight"] = {}
    for phase in ["early", "mid", "late", "pulse", "non_bull"]:
        pt = joined.filter(pl.col("hindsight_phase") == phase)
        result["hindsight"][phase] = _phase_stats(pt, total_n)
        result["hindsight"][phase]["_context"] = _context_stats(pt)

    # ── (B) forward-observable: combined grid ────────────────────────────
    result["forward"] = {}
    fwd_phases = sorted(
        [p for p in joined["fwd_phase"].unique().to_list() if p is not None],
        key=lambda x: (x.split("_")[0], x.split("_")[1]),
    )
    for phase in fwd_phases:
        pt = joined.filter(pl.col("fwd_phase") == phase)
        result["forward"][phase] = _phase_stats(pt, total_n)
        result["forward"][phase]["_context"] = _context_stats(pt)

    # ── yearly breakdown (hindsight) ────────────────────────────────────
    yearly: dict[str, Any] = {}
    for year in sorted(joined["entry_date"].dt.year().unique().to_list()):
        yt = joined.filter(pl.col("entry_date").dt.year() == year)
        ydata = {"trades": yt.height, "total_pnl": round(float(yt["pnl"].sum()), 2)}
        for phase in ["early", "mid", "late", "pulse"]:
            pt = yt.filter(pl.col("hindsight_phase") == phase)
            ydata[phase] = {"trades": pt.height, "total_pnl": round(float(pt["pnl"].sum()), 2)}
        yearly[str(year)] = ydata
    result["yearly"] = yearly

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AMV regime phase diagnostic")
    parser.add_argument("--bull-trigger-pct", type=float, default=4.0)
    parser.add_argument("--bear-trigger-pct", type=float, default=-2.3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trades", type=Path, nargs="*", default=None)
    parser.add_argument("--dont-record-progress", action="store_true")
    args = parser.parse_args()

    # ── build AMV phase frame ──────────────────────────────────────────
    logger.info("Building AMV phase frame ...")
    amv_phase = build_amv_phase_frame(
        bull_trigger_pct=args.bull_trigger_pct,
        bear_trigger_1d_pct=args.bear_trigger_pct,
    )

    regime_summary = (
        amv_phase.filter(pl.col("regime_id").is_not_null())
        .group_by("regime_id")
        .agg([pl.col("date").min().alias("start"), pl.col("date").max().alias("end"), pl.len().alias("days")])
        .sort("regime_id")
    )
    logger.info(
        f"Found {regime_summary.height} bull regimes, "
        f"median {float(regime_summary['days'].median()):.0f}d, max {regime_summary['days'].max()}d"
    )

    # fwd phase distribution
    fwd_dist = amv_phase["fwd_phase"].value_counts().sort("count", descending=True)
    logger.info("Forward-observable phase distribution:")
    for row in fwd_dist.iter_rows():
        logger.info(f"  {row[0]}: {row[1]} days")

    # ── load sleeves ───────────────────────────────────────────────────
    sleeves = dict(DEFAULT_SLEEVES)
    if args.trades:
        keys = list(sleeves.keys())
        for i, path in enumerate(args.trades):
            if i < len(keys):
                sleeves[keys[i]]["trades"] = str(path)

    sleeves_data: dict[str, dict[str, Any]] = {}
    for name, cfg in sleeves.items():
        trades_path = ROOT / cfg["trades"]
        logger.info(f"Loading {name}: {trades_path}")
        trades = load_trades(trades_path)
        logger.info(f"  {trades.height} trades, total PnL {trades['pnl'].sum():,.0f}")
        sleeves_data[name] = compute_stats(trades, amv_phase, cfg["label"])

    # ── assemble output ────────────────────────────────────────────────
    output: dict[str, Any] = {
        "generated_at": timestamp_token(),
        "config": {"bull_trigger_pct": args.bull_trigger_pct, "bear_trigger_pct": args.bear_trigger_pct},
        "regime_inventory": {
            "total_regimes": regime_summary.height,
            "median_duration_days": float(regime_summary["days"].median()),
            "max_duration_days": int(regime_summary["days"].max()),
        },
        "sleeves": sleeves_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Report written to {args.output}")

    # ── print summaries ────────────────────────────────────────────────
    print_summaries(sleeves_data, amv_phase)

    # ── (D) reverse diagnostic: where do the hindsight-late trades land? ─
    print_reverse_diagnostic(amv_phase, sleeves)

    # ── (E) what-if gating simulation ──────────────────────────────────
    print_gating_simulation(amv_phase, sleeves)

    # ── progress.md ────────────────────────────────────────────────────
    if not args.dont_record_progress:
        record_progress(args, sleeves_data)


def print_gating_simulation(
    amv_phase: pl.DataFrame,
    sleeves: dict[str, dict[str, str]],
) -> None:
    """What-if: apply forward-observable gating rules to P3 static trades.

    Compares the raw trades.csv outcome against two skip rules:
      (old)  aged + non-accelerating
      (new)  amv_acceleration < 0 AND amv_dd_from_high < -2 AND regime_duration_days > 10
    """
    p3_cfg = sleeves.get("P3_static_strict") or sleeves.get(list(sleeves.keys())[0])
    trades = load_trades(ROOT / p3_cfg["trades"])
    join_cols = [
        "date", "hindsight_phase", "fwd_phase",
        "fwd_duration_bucket", "fwd_momentum_bucket",
        "regime_duration_days", "regime_total_days", "regime_maturity",
        "amv_ret_from_start", "amv_slope_5d", "amv_slope_20d",
        "amv_acceleration", "amv_dd_from_high",
        "amv_ret_ma3", "amv_ret_ma5", "amplitude_ma3", "amv_neg_streak",
        "amplitude_pct", "turnover_pct",
    ]
    joined = trades.join(
        amv_phase.select(join_cols), left_on="entry_date", right_on="date", how="left"
    )

    # ── rule definitions ──────────────────────────────────────────────
    rule_old = (
        (pl.col("fwd_duration_bucket") == "aged")
        & (pl.col("fwd_momentum_bucket").is_in(["cruising", "stalling", "retreating"]))
    )

    rule_new = (
        (pl.col("amv_acceleration") < 0)
        & (pl.col("amv_dd_from_high") < -2.0)
        & (pl.col("regime_duration_days") > 10)
    )

    rule_mat = (
        (pl.col("regime_maturity") > 50)
        & (pl.col("amv_acceleration") < -10)
    )

    # chaos-based rules: negative streak + elevated amplitude
    rule_chaos_a = (
        (pl.col("amv_neg_streak") >= 3)
        & (pl.col("amplitude_pct") > 2.5)
    )  # 3+ consecutive negative days + high amplitude

    rule_chaos_b = (
        (pl.col("amv_ret_ma3") < -0.3)
        & (pl.col("amplitude_pct") > 2.5)
    )  # 3d avg return negative < -0.3% + high amplitude

    rule_chaos_c = (
        (pl.col("amv_ret_ma5") < -0.2)
        & (pl.col("amplitude_ma3") > 2.5)
    )  # 5d avg return negative + high recent amplitude

    rule_chaos_d = (
        (pl.col("amv_ret_ma3") < 0)
        & (pl.col("amplitude_pct") > 2.5)
        & (pl.col("regime_duration_days") > 5)
    )  # 3d avg negative (any) + high amplitude + not fresh

    joined = joined.with_columns([
        rule_old.alias("_skip_old"),
        rule_new.alias("_skip_new"),
        rule_mat.alias("_skip_mat"),
        rule_chaos_a.alias("_skip_ca"),
        rule_chaos_b.alias("_skip_cb"),
        rule_chaos_c.alias("_skip_cc"),
        rule_chaos_d.alias("_skip_cd"),
    ])

    total_pnl = float(joined["pnl"].sum())
    total_n = joined.height

    # ── evaluate each rule ────────────────────────────────────────────
    for rule_name, skip_col in [
        ("OLD: aged + non-accel", "_skip_old"),
        ("NEW: accel<0 & dd<-2% & dur>10", "_skip_new"),
        ("MAT: maturity>50 & accel<-10%", "_skip_mat"),
        ("CHAOS-A: neg_streak>=3 & amp>2.5", "_skip_ca"),
        ("CHAOS-B: ret_ma3<-0.3 & amp>2.5", "_skip_cb"),
        ("CHAOS-C: ret_ma5<-0.2 & amp_ma3>2.5", "_skip_cc"),
        ("CHAOS-D: ret_ma3<0 & amp>2.5 & dur>5", "_skip_cd"),
    ]:
        skipped = joined.filter(pl.col(skip_col))
        kept = joined.filter(~pl.col(skip_col))
        skip_n = skipped.height
        skip_pnl = float(skipped["pnl"].sum())
        kept_pnl = float(kept["pnl"].sum())
        skip_win = skipped.filter(pl.col("pnl") > 0).height
        skip_wr = skip_win / skip_n if skip_n else 0

        print(f"\n--- Rule: {rule_name} ---")
        print(f"  Skipped: {skip_n}t / {total_n}t total")
        print(f"  Skipped PnL: {skip_pnl:+,.0f}")
        print(f"  Skipped WR:  {skip_wr:.1%}")
        print(f"  Kept PnL:    {kept_pnl:+,.0f}  (was {total_pnl:+,.0f}, delta {total_pnl - kept_pnl:+,.0f})")
        if kept_n := kept.height:
            kept_wr = kept.filter(pl.col("pnl") > 0).height / kept_n
            print(f"  Kept WR:     {kept_wr:.1%}  (was {joined.filter(pl.col('pnl')>0).height/total_n:.1%})")

        # what-if: P3 static has non-overlapping 6d cadence, so skip = cash for that round
        # show new effective PnL
        print(f"  Effective PnL after gating: {kept_pnl:+,.0f}")

        # check for mis-killed winners
        big_wins = skipped.filter(pl.col("pnl") > 20_000).sort("pnl", descending=True)
        if big_wins.height > 0:
            print(f"  WARNING: mis-killed big winners (PnL > 20K): {big_wins.height}")
            for row in big_wins.head(5).iter_rows(named=True):
                print(
                    f"    {row['code']:<12} {str(row['entry_date']):<12} "
                    f"PnL={row['pnl']:>10,.0f} {row['pnl_pct']:>7.1%} "
                    f"dur={row['regime_duration_days']:.0f}/{row['regime_total_days']:.0f}d "
                    f"mat={row['regime_maturity']:.0f}% "
                    f"accel={row['amv_acceleration']:.1f}% dd={row['amv_dd_from_high']:.1f}%"
                )
        else:
            print(f"  No mis-killed big winners (>20K).")

    # ── overlap between old and new rules ──────────────────────────────
    both = joined.filter(pl.col("_skip_old") & pl.col("_skip_new"))
    new_only = joined.filter(~pl.col("_skip_old") & pl.col("_skip_new"))
    old_only = joined.filter(pl.col("_skip_old") & ~pl.col("_skip_new"))

    print("\n--- Rule overlap ---")
    print(f"  Both skip:    {both.height}t, PnL={float(both['pnl'].sum()):+,.0f}")
    print(f"  New-only skip:{new_only.height}t, PnL={float(new_only['pnl'].sum()):+,.0f}")
    print(f"  Old-only skip:{old_only.height}t, PnL={float(old_only['pnl'].sum()):+,.0f}")

    if new_only.height > 0:
        all_new_skip = joined.filter(pl.col("_skip_new"))
        print(f"\n  All new-rule skipped trades ({all_new_skip.height}t, PnL={float(all_new_skip['pnl'].sum()):+,.0f}):")
        print(f"  {'code':<12} {'entry':<12} {'PnL':>10} {'PnL%':>7} {'dur':>5} {'mat':>5} {'accel':>7} {'dd':>7} {'fwd_phase':>28}")
        for row in all_new_skip.sort("pnl").iter_rows(named=True):
            print(
                f"  {row['code']:<12} {str(row['entry_date']):<12} "
                f"{row['pnl']:>10,.0f} {row['pnl_pct']:>6.1%} "
                f"{row['regime_duration_days']:>4.0f}d "
                f"{row['regime_maturity']:>4.0f}% "
                f"{row['amv_acceleration']:>6.1f}% {row['amv_dd_from_high']:>6.1f}% "
                f"{row['fwd_phase']:<28}"
            )


def print_reverse_diagnostic(
    amv_phase: pl.DataFrame,
    sleeves: dict[str, dict[str, str]],
) -> None:
    """For P3's hindsight-late trades (-221K), show what forward features looked like at entry."""
    # load P3 trades
    p3_cfg = sleeves.get("P3_static_strict") or sleeves.get(list(sleeves.keys())[0])
    trades = load_trades(ROOT / p3_cfg["trades"])
    join_cols = [
        "date", "hindsight_phase", "fwd_phase",
        "fwd_duration_bucket", "fwd_momentum_bucket",
        "regime_duration_days", "regime_total_days", "regime_progress", "regime_maturity",
        "amv_ret_from_start", "amv_ret_1d",
        "amv_slope_5d", "amv_slope_20d", "amv_acceleration", "amv_dd_from_high",
        "amv_ret_ma3", "amv_ret_ma5", "amplitude_ma3", "amv_neg_streak",
        "amplitude_pct", "turnover_pct",
    ]
    joined = trades.join(amv_phase.select(join_cols), left_on="entry_date", right_on="date", how="left")

    late_trades = joined.filter(pl.col("hindsight_phase") == "late")

    print("\n" + "=" * 70)
    print("(D) REVERSE DIAGNOSTIC: Hindsight-late trades in forward-observable space")
    print(f"    P3 static strict: {late_trades.height} late trades, total PnL {late_trades['pnl'].sum():,.0f}")
    print("=" * 70)

    # maturity curve reference
    all_maturities = (
        joined.filter(pl.col("regime_maturity").is_not_null())
        .select(["regime_duration_days", "regime_maturity"])
        .unique()
        .sort("regime_duration_days")
    )
    print("\nMaturity curve (empirical survival CDF):")
    curve_str = ""
    for row in all_maturities.iter_rows():
        d, m = int(row[0]), float(row[1])
        curve_str += f"  d{d:>2}: {m:>5.1f}%"
        if d % 10 == 0 or d == 1:
            curve_str += "\n"
    print(curve_str)

    # ── forward feature summary of late trades ──────────────────────────
    print(f"Late trades AMV environment at entry (avg):")
    print(f"  regime_duration_days:  {late_trades['regime_duration_days'].mean():.0f}d")
    print(f"  regime_maturity:       {late_trades['regime_maturity'].mean():.1f}%")
    print(f"  regime_total_days:     {late_trades['regime_total_days'].mean():.0f}d")
    print(f"  regime_progress:       {late_trades['regime_progress'].mean():.2%}")
    print(f"  amv_slope_5d:          {late_trades['amv_slope_5d'].mean():.2f}%")
    print(f"  amv_acceleration:      {late_trades['amv_acceleration'].mean():.2f}%")
    print(f"  amv_dd_from_high:      {late_trades['amv_dd_from_high'].mean():.2f}%")

    # ── late trades by forward duration bucket ──────────────────────────
    print("\n--- Late P3 trades by forward duration bucket ---")
    for dur in ["fresh", "young", "aged", "old"]:
        pt = late_trades.filter(pl.col("fwd_duration_bucket") == dur)
        if pt.height == 0:
            continue
        print(
            f"  {dur:<6}: {pt.height:>3}t, PnL={pt['pnl'].sum():>10,.0f}, "
            f"WR={float(pt.filter(pl.col('pnl')>0).height)/pt.height:.1%}, "
            f"avg_pnl_pct={float(pt['pnl_pct'].mean())*100:.2f}%"
        )

    # ── late trades by forward momentum bucket ─────────────────────────
    print("\n--- Late P3 trades by forward momentum bucket ---")
    for mom in ["accelerating", "cruising", "stalling", "retreating"]:
        pt = late_trades.filter(pl.col("fwd_momentum_bucket") == mom)
        if pt.height == 0:
            continue
        print(
            f"  {mom:<14}: {pt.height:>3}t, PnL={pt['pnl'].sum():>10,.0f}, "
            f"WR={float(pt.filter(pl.col('pnl')>0).height)/pt.height:.1%}, "
            f"avg_pnl_pct={float(pt['pnl_pct'].mean())*100:.2f}%"
        )

    # ── late trades by forward combined phase (top cells) ──────────────
    print("\n--- Late P3 trades by forward phase (sorted by PnL, top 15) ---")
    late_by_fwd = (
        late_trades.group_by("fwd_phase")
        .agg([
            pl.len().alias("trades"),
            pl.col("pnl").sum().alias("total_pnl"),
            (pl.col("pnl") > 0).sum().alias("wins"),
            pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
            pl.col("regime_duration_days").mean().alias("avg_dur"),
            pl.col("regime_total_days").mean().alias("avg_total_dur"),
            pl.col("amv_slope_5d").mean().alias("avg_slope_5d"),
            pl.col("amv_dd_from_high").mean().alias("avg_dd"),
            pl.col("regime_maturity").mean().alias("avg_maturity"),
        ])
        .sort("total_pnl")
    )
    for row in late_by_fwd.head(15).iter_rows():
        phase, n, pnl, wins, avg_pnl, avg_dur, avg_tot, avg_s5, avg_dd, avg_mat = row
        wr = wins / n if n else 0
        print(
            f"  {phase:<28}: {n:>3}t, PnL={pnl:>10,.0f}, WR={wr:.1%}, "
            f"avg_pnl_pct={avg_pnl*100:.2f}%, dur={avg_dur:.0f}/{avg_tot:.0f}d, "
            f"mat={avg_mat:.0f}%, slope5d={avg_s5:.1f}%, dd={avg_dd:.1f}%"
        )

    # ── key question: non-accelerating aged/old vs accelerating ────────
    print("\n--- Late trades: accelerating vs non-accelerating ---")
    accel = late_trades.filter(pl.col("fwd_momentum_bucket") == "accelerating")
    non_accel = late_trades.filter(pl.col("fwd_momentum_bucket") != "accelerating")
    print(
        f"  accelerating:    {accel.height}t, PnL={accel['pnl'].sum():,.0f}, "
        f"WR={float(accel.filter(pl.col('pnl')>0).height)/accel.height:.1%}" if accel.height else "  accelerating: 0 trades"
    )
    print(
        f"  non-accelerating: {non_accel.height}t, PnL={non_accel['pnl'].sum():,.0f}, "
        f"WR={float(non_accel.filter(pl.col('pnl')>0).height)/non_accel.height:.1%}" if non_accel.height else "  non-accelerating: 0 trades"
    )

    # ── trade-level list: worst 15 late trades with forward context ────
    print(f"  {'code':<12} {'entry':<12} {'PnL':>10} {'PnL%':>8} {'dur/tot':>9} {'mat':>5} {'slope5d':>8} {'dd_high':>7} {'phase':>28}")
    worst = late_trades.sort("pnl").head(15)
    for row in worst.iter_rows(named=True):
        print(
            f"  {row['code']:<12} {str(row['entry_date']):<12} "
            f"{row['pnl']:>10,.0f} {row['pnl_pct']:>7.1%} "
            f"{int(row['regime_duration_days']):>3}/{int(row['regime_total_days']):>3} "
            f"{row['regime_maturity']:>4.0f}% "
            f"{row['amv_slope_5d']:>7.1f}% {row['amv_dd_from_high']:>6.1f}% "
            f"{row['fwd_phase']:<28}"
        )

    # ── also show best 5 to contrast ───────────────────────────────────
    print(f"\n  Best 5 late P3 trades for contrast:")
    best = late_trades.sort("pnl", descending=True).head(5)
    for row in best.iter_rows(named=True):
        print(
            f"  {row['code']:<12} {str(row['entry_date']):<12} "
            f"{row['pnl']:>10,.0f} {row['pnl_pct']:>7.1%} "
            f"{int(row['regime_duration_days']):>3}/{int(row['regime_total_days']):>3} "
            f"{row['regime_maturity']:>4.0f}% "
            f"{row['amv_slope_5d']:>7.1f}% {row['amv_dd_from_high']:>6.1f}% "
            f"{row['fwd_phase']:<28}"
        )


def print_summaries(sleeves_data: dict[str, dict[str, Any]], amv_phase: pl.DataFrame) -> None:
    names = list(sleeves_data.keys())

    # ── (A) Hindsight summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("(A) HINDSIGHT PHASES  (regime_progress: early 0-25%, mid 25-75%, late 75-100%)")
    print("    WARNING: requires knowing total regime length – NOT tradable")
    print("=" * 70)
    phases_h = ["early", "mid", "late", "pulse"]

    header = f"{'Phase':<10}"
    for name in names:
        header += f" {name:>24}"
    print(header)
    print("-" * len(header))
    for phase in phases_h:
        line = f"{phase:<10}"
        for name in names:
            d = sleeves_data[name].get("hindsight", {}).get(phase, {})
            pnl = d.get("total_pnl", 0) or 0
            n = d.get("trades", 0) or 0
            line += f" {pnl:>12,.0f} ({n:>3}t)"
        print(line)

    print(f"\n{'Phase':<10} {'P3 avg_pnl_pct':>16} {'P3 WR':>8} {'PB3 avg_pnl_pct':>16} {'PB3 WR':>8}")
    print("-" * 60)
    for phase in phases_h:
        p3d = sleeves_data.get("P3_static_strict", {}).get("hindsight", {}).get(phase, {})
        pb3d = sleeves_data.get("PB3_rolling_refill", {}).get("hindsight", {}).get(phase, {})
        p3_pnl = p3d.get("avg_pnl_pct") or 0
        p3_wr = p3d.get("win_rate") or 0
        pb3_pnl = pb3d.get("avg_pnl_pct") or 0
        pb3_wr = pb3d.get("win_rate") or 0
        print(f"{phase:<10} {p3_pnl:>16.4%} {p3_wr:>7.1%} {pb3_pnl:>16.4%} {pb3_wr:>7.1%}")

    # ── (B) Forward-observable 2D grid ─────────────────────────────────
    print("\n" + "=" * 70)
    print("(B) FORWARD-OBSERVABLE PHASES   (duration × momentum, tradable at entry)")
    print("=" * 70)

    durations = ["fresh", "young", "aged", "old"]
    momentums = ["accelerating", "cruising", "stalling", "retreating"]

    for name in names:
        fwd = sleeves_data[name].get("forward", {})
        print(f"\n--- {name} ---")
        # header
        header = f"{'dur\\mom':<10}"
        for m in momentums:
            header += f" {m:>22}"
        print(header)
        print("-" * len(header))
        for dur in durations:
            line = f"{dur:<10}"
            for mom in momentums:
                key = f"{dur}_{mom}"
                d = fwd.get(key, {})
                pnl = d.get("total_pnl", 0) or 0
                n = d.get("trades", 0) or 0
                wr = d.get("win_rate") or 0
                wr_s = f"{wr:.0%}" if isinstance(wr, (int, float)) and n > 0 else "-"
                line += f" {pnl:>10,.0f} ({n:>3}t {wr_s:>4})"
            print(line)

    # ── (C) P3 vs PB3 delta grid (which sleeve wins in each cell) ──────
    if len(names) >= 2:
        print("\n" + "=" * 70)
        print("(C) P3 – PB3 DELTA GRID   (where does each sleeve dominate?)")
        print("=" * 70)
        p3_fwd = sleeves_data.get("P3_static_strict", {}).get("forward", {})
        pb3_fwd = sleeves_data.get("PB3_rolling_refill", {}).get("forward", {})

        header = f"{'dur\\mom':<10}"
        for m in momentums:
            header += f" {m:>22}"
        print(header)
        print("-" * len(header))
        for dur in durations:
            line = f"{dur:<10}"
            for mom in momentums:
                key = f"{dur}_{mom}"
                p3_pnl = p3_fwd.get(key, {}).get("total_pnl", 0) or 0
                pb3_pnl = pb3_fwd.get(key, {}).get("total_pnl", 0) or 0
                delta = p3_pnl - pb3_pnl
                p3_n = p3_fwd.get(key, {}).get("trades", 0) or 0
                pb3_n = pb3_fwd.get(key, {}).get("trades", 0) or 0
                winner = "P3" if delta > 0 else "PB" if delta < 0 else "--"
                line += f" {delta:>10,.0f} ({winner} {p3_n}t/{pb3_n}t)"
            print(line)


def record_progress(args: argparse.Namespace, sleeves_data: dict[str, dict[str, Any]]) -> None:
    from datetime import date
    today = date.today().isoformat()
    entry = (
        f"\n## {today}\n\n"
        f"### [AMV] Regime phase diagnostic – forward-observable edition\n\n"
        f"- 脚本: `scripts/amv_regime_phase_diagnostic.py`\n"
        f"- 产物: `{_rel_path(args.output)}`\n"
        f"- 双系统: (A) hindsight early/mid/late, (B) forward duration×momentum grid\n"
    )
    for name in sleeves_data:
        s = sleeves_data[name]
        entry += f"- {name} hindsight: "
        for phase in ["early", "mid", "late", "pulse"]:
            d = s.get("hindsight", {}).get(phase, {})
            entry += f"{phase}={d.get('total_pnl',0):,.0f}({d.get('trades',0)}t) "
        entry += "\n"
    progress_path = ROOT / "progress.md"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(entry)


def _rel_path(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


if __name__ == "__main__":
    main()
