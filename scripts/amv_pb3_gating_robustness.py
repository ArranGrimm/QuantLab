from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_regime_phase_diagnostic import build_amv_phase_frame


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_DIR = (
    ROOT
    / "artifacts"
    / "amv_static_sleeve_signals"
    / "20260521_090945_pullback_p0_k0_pb3_cp1_rv0"
)
DEFAULT_TRADES = (
    DEFAULT_SIGNAL_DIR
    / "backtests"
    / "6td_rolling21_refill_top10_no_stop_20260521_091007_830"
    / "trades.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "amv_pb3_gating_robustness.json"

NON_ACCEL_MOMENTUMS = ["cruising", "stalling", "retreating"]
BIG_WIN_PNL = 20_000.0


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    duration_mode: str
    use_chaos: bool
    neg_streak_min: int | None = None
    amplitude_min: float | None = None


def _parse_date_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides={
            "code": pl.Utf8,
            "entry_date": pl.Utf8,
            "exit_date": pl.Utf8,
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "shares": pl.Int64,
            "cost": pl.Float64,
            "exit_value": pl.Float64,
            "pnl": pl.Float64,
            "pnl_pct": pl.Float64,
            "hold_trading_days": pl.Int64,
            "exit_reason": pl.Utf8,
        },
    ).with_columns(
        pl.col("entry_date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("exit_date").str.strptime(pl.Date, "%Y-%m-%d"),
    )


def load_trade_context(signal_dir: Path, trades_path: Path) -> pl.DataFrame:
    trades = _parse_date_csv(trades_path).with_row_index("trade_id")
    signal_path = signal_dir / "signal.parquet"
    signals = (
        pl.scan_parquet(signal_path)
        .filter(pl.col("is_signal"))
        .select(
            [
                pl.col("date").alias("entry_date"),
                "code",
                "signal_date",
                "score",
                "rank",
            ]
        )
        .collect()
    )

    phase = build_amv_phase_frame().select(
        [
            pl.col("date").alias("signal_date"),
            "fwd_duration_bucket",
            "fwd_momentum_bucket",
            "fwd_phase",
            "regime_duration_days",
            "regime_maturity",
            "amv_slope_5d",
            "amv_slope_20d",
            "amv_acceleration",
            "amv_dd_from_high",
            "amv_ret_ma3",
            "amv_ret_ma5",
            "amplitude_ma3",
            "amv_neg_streak",
            "amplitude_pct",
        ]
    )

    joined = (
        trades.join(signals, on=["entry_date", "code"], how="left")
        .join(phase, on="signal_date", how="left")
        .with_columns(pl.col("entry_date").dt.year().alias("entry_year"))
    )
    missing = joined.filter(pl.col("signal_date").is_null()).height
    if missing:
        raise ValueError(f"{missing} trades could not be matched to signal rows")
    return joined


def duration_gate_expr(mode: str) -> pl.Expr:
    non_accel = pl.col("fwd_momentum_bucket").is_in(NON_ACCEL_MOMENTUMS)
    if mode == "none":
        return pl.lit(False)
    if mode == "aged":
        return (pl.col("fwd_duration_bucket") == "aged") & non_accel
    if mode == "aged_or_old":
        return pl.col("fwd_duration_bucket").is_in(["aged", "old"]) & non_accel
    if mode == "old":
        return (pl.col("fwd_duration_bucket") == "old") & non_accel
    raise ValueError(f"unknown duration mode: {mode}")


def rule_expr(rule: RuleSpec) -> pl.Expr:
    expr = duration_gate_expr(rule.duration_mode)
    if rule.use_chaos:
        if rule.neg_streak_min is None or rule.amplitude_min is None:
            raise ValueError(f"chaos rule missing thresholds: {rule}")
        chaos = (
            (pl.col("amv_neg_streak") >= rule.neg_streak_min)
            & (pl.col("amplitude_pct") > rule.amplitude_min)
        )
        expr = expr | chaos
    return expr.fill_null(False)


def make_rules() -> list[RuleSpec]:
    rules: list[RuleSpec] = []
    seen: set[str] = set()

    def add(rule: RuleSpec) -> None:
        if rule.rule_id not in seen:
            rules.append(rule)
            seen.add(rule.rule_id)

    for duration_mode in ["aged", "aged_or_old", "old"]:
        add(RuleSpec(f"{duration_mode}_nonaccel", duration_mode, False))

    for duration_mode in ["none", "aged", "aged_or_old", "old"]:
        for neg in [2, 3, 4]:
            for amp in [2.0, 2.5, 3.0, 3.5]:
                amp_token = str(amp).replace(".", "p")
                prefix = "chaos" if duration_mode == "none" else f"{duration_mode}_nonaccel_or_chaos"
                add(
                    RuleSpec(
                        f"{prefix}_n{neg}_amp{amp_token}",
                        duration_mode,
                        True,
                        neg,
                        amp,
                    )
                )
    return rules


def summarize_subset(df: pl.DataFrame, total_pnl: float, total_n: int) -> dict[str, Any]:
    n = df.height
    pnl = float(df["pnl"].sum()) if n else 0.0
    return {
        "trades": n,
        "trade_share_pct": round(n / total_n * 100, 2) if total_n else 0.0,
        "pnl": round(pnl, 2),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 6) if n else None,
        "win_rate": round(df.filter(pl.col("pnl") > 0).height / n, 4) if n else None,
        "delta_vs_raw_pnl": round(-pnl, 2),
        "delta_vs_raw_pnl_pct_of_raw": round((-pnl / total_pnl) * 100, 2) if total_pnl else None,
    }


def yearly_delta(skipped: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if skipped.height == 0:
        return {}
    yearly = (
        skipped.group_by("entry_year")
        .agg(
            [
                pl.len().alias("skipped_trades"),
                pl.col("pnl").sum().alias("skipped_pnl"),
                pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
                (pl.col("pnl") > 0).sum().alias("skipped_winners"),
            ]
        )
        .sort("entry_year")
    )
    return {
        str(row["entry_year"]): {
            "skipped_trades": row["skipped_trades"],
            "skipped_pnl": round(row["skipped_pnl"], 2),
            "delta_vs_raw_pnl": round(-row["skipped_pnl"], 2),
            "avg_pnl_pct": round(row["avg_pnl_pct"], 6) if row["avg_pnl_pct"] is not None else None,
            "skipped_win_rate": round(row["skipped_winners"] / row["skipped_trades"], 4)
            if row["skipped_trades"]
            else None,
        }
        for row in yearly.iter_rows(named=True)
    }


def evaluate_rule(df: pl.DataFrame, rule: RuleSpec) -> dict[str, Any]:
    total_pnl = float(df["pnl"].sum())
    total_n = df.height
    marked = df.with_columns(rule_expr(rule).alias("_skip"))
    skipped = marked.filter(pl.col("_skip"))
    kept = marked.filter(~pl.col("_skip"))
    yearly = yearly_delta(skipped)
    year_deltas = [item["delta_vs_raw_pnl"] for item in yearly.values()]
    positive_years = sum(1 for value in year_deltas if value > 0)
    negative_years = sum(1 for value in year_deltas if value < 0)
    return {
        "rule_id": rule.rule_id,
        "duration_mode": rule.duration_mode,
        "use_chaos": rule.use_chaos,
        "neg_streak_min": rule.neg_streak_min,
        "amplitude_min": rule.amplitude_min,
        "raw_pnl": round(total_pnl, 2),
        "kept_pnl": round(float(kept["pnl"].sum()), 2),
        "effective_trade_level_delta": round(-float(skipped["pnl"].sum()), 2),
        "effective_trade_level_delta_pct_of_raw": round(
            (-float(skipped["pnl"].sum()) / total_pnl) * 100, 2
        )
        if total_pnl
        else None,
        "skipped": summarize_subset(skipped, total_pnl, total_n),
        "kept": summarize_subset(kept, total_pnl, total_n),
        "mis_killed_big_winners": skipped.filter(pl.col("pnl") > BIG_WIN_PNL).height,
        "yearly": yearly,
        "positive_delta_years": positive_years,
        "negative_delta_years": negative_years,
        "min_year_delta": min(year_deltas) if year_deltas else 0.0,
        "max_year_delta": max(year_deltas) if year_deltas else 0.0,
    }


def walk_forward(df: pl.DataFrame, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    years = sorted(df["entry_year"].unique().to_list())
    by_rule = {item["rule_id"]: item for item in metrics}
    rules = {rule.rule_id: rule for rule in make_rules()}
    rows: list[dict[str, Any]] = []
    for test_year in years[1:]:
        train = df.filter(pl.col("entry_year") < test_year)
        test = df.filter(pl.col("entry_year") == test_year)
        if train.height == 0 or test.height == 0:
            continue
        train_metrics = [evaluate_rule(train, rule) for rule in rules.values()]
        train_metrics = [
            item
            for item in train_metrics
            if item["skipped"]["trade_share_pct"] <= 35.0 and item["skipped"]["trades"] >= 20
        ]
        if not train_metrics:
            continue
        chosen = sorted(
            train_metrics,
            key=lambda item: (
                item["effective_trade_level_delta"],
                item["positive_delta_years"],
                -item["mis_killed_big_winners"],
            ),
            reverse=True,
        )[0]
        test_result = evaluate_rule(test, rules[chosen["rule_id"]])
        current_test = evaluate_rule(test, rules["aged_nonaccel_or_chaos_n3_amp2p5"])
        rows.append(
            {
                "test_year": test_year,
                "chosen_rule_id": chosen["rule_id"],
                "train_delta": chosen["effective_trade_level_delta"],
                "test_delta": test_result["effective_trade_level_delta"],
                "test_skipped_trades": test_result["skipped"]["trades"],
                "current_rule_test_delta": current_test["effective_trade_level_delta"],
                "chosen_full_sample_rank_by_delta": sorted(
                    by_rule,
                    key=lambda rid: by_rule[rid]["effective_trade_level_delta"],
                    reverse=True,
                ).index(chosen["rule_id"])
                + 1
                if chosen["rule_id"] in by_rule
                else None,
            }
        )
    return rows


def summarize_walk_forward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    chosen_deltas = [row["test_delta"] for row in rows]
    current_deltas = [row["current_rule_test_delta"] for row in rows]
    return {
        "test_years": [row["test_year"] for row in rows],
        "chosen_total_test_delta": round(sum(chosen_deltas), 2),
        "chosen_positive_test_years": sum(1 for value in chosen_deltas if value > 0),
        "current_rule_total_test_delta": round(sum(current_deltas), 2),
        "current_rule_positive_test_years": sum(1 for value in current_deltas if value > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PB3 regime gating robustness diagnostic")
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_trade_context(args.signal_dir, args.trades)
    rules = make_rules()
    metrics = [evaluate_rule(df, rule) for rule in rules]
    top_by_delta = sorted(metrics, key=lambda item: item["effective_trade_level_delta"], reverse=True)[:20]
    positive_metrics = [item for item in metrics if item["effective_trade_level_delta"] > 0]
    top_stable = sorted(
        positive_metrics,
        key=lambda item: (
            item["positive_delta_years"],
            item["min_year_delta"],
            item["effective_trade_level_delta"],
        ),
        reverse=True,
    )[:20]
    current = next(item for item in metrics if item["rule_id"] == "aged_nonaccel_or_chaos_n3_amp2p5")
    wf_rows = walk_forward(df, metrics)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": {
            "signal_dir": str(args.signal_dir),
            "trades": str(args.trades),
            "trade_count": df.height,
            "raw_trade_pnl": round(float(df["pnl"].sum()), 2),
            "years": sorted(df["entry_year"].unique().to_list()),
            "note": "Trade-level skip approximation. Rust account-level exposure/cash effects must be rechecked for final candidates.",
        },
        "current_rule": current,
        "top_by_delta": top_by_delta,
        "top_stable": top_stable,
        "walk_forward_summary": summarize_walk_forward(wf_rows),
        "walk_forward": wf_rows,
        "all_rules": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "current_rule_delta": current["effective_trade_level_delta"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
