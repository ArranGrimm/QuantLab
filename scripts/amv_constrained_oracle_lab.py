from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from scripts.amv_horizon_aware_oracle_lab import DEFAULT_OUTPUT_ROOT as HORIZON_ORACLE_ROOT


DEFAULT_ORACLE_DIR = HORIZON_ORACLE_ROOT / "20260517_130248"
DEFAULT_OUTPUT_ROOT = HORIZON_ORACLE_ROOT.parent / "amv_constrained_oracle"
DEFAULT_ATTACK_CANDIDATES = [
    "ret_5d_6td",
    "ret_20d_6td",
    "ret_20d_2td",
    "kmid2_6td",
]
DEFAULT_MARGINS = [0.0, 0.01, 0.02, 0.03]


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must not be empty")
    return items


def parse_margins(value: str) -> list[float]:
    margins = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not margins:
        raise argparse.ArgumentTypeError("margins must not be empty")
    if any(margin < 0 for margin in margins):
        raise argparse.ArgumentTypeError("margins must be non-negative")
    return margins


def read_daily_candidates(path: Path) -> pl.DataFrame:
    if path.is_dir():
        path = path / "daily_candidate_sleeves.csv"
    if not path.exists():
        raise FileNotFoundError(f"daily_candidate_sleeves.csv not found: {path}")
    return pl.read_csv(path, try_parse_dates=True)


def _prefixed_select(prefix: str) -> list[pl.Expr]:
    return [
        pl.col("candidate_id").alias(f"{prefix}_candidate_id"),
        pl.col("sleeve_id").alias(f"{prefix}_sleeve_id"),
        pl.col("horizon").alias(f"{prefix}_horizon"),
        pl.col("top_ret").alias(f"{prefix}_ret"),
        pl.col("top_ret_dailyized").alias(f"{prefix}_dailyized_ret"),
        pl.col("top_mfe").alias(f"{prefix}_mfe"),
        pl.col("top_mae").alias(f"{prefix}_mae"),
        pl.col("top_hit15").alias(f"{prefix}_hit15"),
    ]


def build_best_attack(
    daily: pl.DataFrame,
    *,
    attack_candidates: list[str],
    metric_col: str,
) -> pl.DataFrame:
    return (
        daily.filter(pl.col("candidate_id").is_in(attack_candidates))
        .sort(["date", metric_col, "candidate_id"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(1)
        .select(["date", *_prefixed_select("attack")])
    )


def build_base(daily: pl.DataFrame, *, base_candidate: str) -> pl.DataFrame:
    return (
        daily.filter(pl.col("candidate_id") == base_candidate)
        .select(["date", "year", *_prefixed_select("base")])
        .sort("date")
    )


def choose_constrained(
    daily: pl.DataFrame,
    *,
    base_candidate: str,
    attack_candidates: list[str],
    metric_col: str,
    margin: float,
    allow_cash: bool,
) -> pl.DataFrame:
    if metric_col not in {"top_ret", "top_ret_dailyized"}:
        raise ValueError(f"unsupported metric_col: {metric_col}")
    metric_suffix = "ret" if metric_col == "top_ret" else "dailyized_ret"
    base_metric_col = f"base_{metric_suffix}"
    attack_metric_col = f"attack_{metric_suffix}"
    base = build_base(daily, base_candidate=base_candidate)
    attack = build_best_attack(daily, attack_candidates=attack_candidates, metric_col=metric_col)
    joined = base.join(attack, on="date", how="inner")

    attack_beats = pl.col(attack_metric_col) > (pl.col(base_metric_col) + margin)
    cash_condition = allow_cash & (pl.col("base_ret") < 0) & (pl.col("attack_ret") <= 0)
    choice_expr = (
        pl.when(cash_condition)
        .then(pl.lit("cash"))
        .when(attack_beats)
        .then(pl.col("attack_candidate_id"))
        .otherwise(pl.col("base_candidate_id"))
    )
    choice_type_expr = (
        pl.when(cash_condition)
        .then(pl.lit("cash"))
        .when(attack_beats)
        .then(pl.lit("attack"))
        .otherwise(pl.lit("base"))
    )
    return (
        joined.with_columns(
            [
                choice_expr.alias("chosen_candidate_id"),
                choice_type_expr.alias("choice_type"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("choice_type") == "cash")
                .then(0)
                .when(pl.col("choice_type") == "attack")
                .then(pl.col("attack_horizon"))
                .otherwise(pl.col("base_horizon"))
                .alias("chosen_horizon"),
                pl.when(pl.col("choice_type") == "cash")
                .then(0.0)
                .when(pl.col("choice_type") == "attack")
                .then(pl.col("attack_ret"))
                .otherwise(pl.col("base_ret"))
                .alias("chosen_ret"),
                pl.when(pl.col("choice_type") == "cash")
                .then(0.0)
                .when(pl.col("choice_type") == "attack")
                .then(pl.col("attack_dailyized_ret"))
                .otherwise(pl.col("base_dailyized_ret"))
                .alias("chosen_dailyized_ret"),
                pl.when(pl.col("choice_type") == "cash")
                .then(0.0)
                .when(pl.col("choice_type") == "attack")
                .then(pl.col("attack_mfe"))
                .otherwise(pl.col("base_mfe"))
                .alias("chosen_mfe"),
                pl.when(pl.col("choice_type") == "cash")
                .then(0.0)
                .when(pl.col("choice_type") == "attack")
                .then(pl.col("attack_mae"))
                .otherwise(pl.col("base_mae"))
                .alias("chosen_mae"),
            ]
        )
        .with_columns(
            [
                (pl.col("chosen_ret") - pl.col("base_ret")).alias("lift_vs_base"),
                (pl.col("chosen_dailyized_ret") - pl.col("base_dailyized_ret")).alias(
                    "dailyized_lift_vs_base"
                ),
                pl.lit(metric_col).alias("target_metric"),
                pl.lit(margin).alias("margin"),
                pl.lit(allow_cash).alias("allow_cash"),
            ]
        )
        .sort("date")
    )


def summarize_choice(daily: pl.DataFrame) -> dict[str, Any]:
    return {
        "target_metric": daily["target_metric"][0],
        "margin": float(daily["margin"][0]),
        "allow_cash": bool(daily["allow_cash"][0]),
        "days": daily.height,
        "mean_base_ret": float(daily["base_ret"].mean()),
        "mean_chosen_ret": float(daily["chosen_ret"].mean()),
        "mean_lift_vs_base": float(daily["lift_vs_base"].mean()),
        "mean_base_dailyized_ret": float(daily["base_dailyized_ret"].mean()),
        "mean_chosen_dailyized_ret": float(daily["chosen_dailyized_ret"].mean()),
        "mean_dailyized_lift_vs_base": float(daily["dailyized_lift_vs_base"].mean()),
        "positive_lift_ratio": float((daily["lift_vs_base"] > 0).mean()),
        "base_days": int((daily["choice_type"] == "base").sum()),
        "attack_days": int((daily["choice_type"] == "attack").sum()),
        "cash_days": int((daily["choice_type"] == "cash").sum()),
        "mean_chosen_mfe": float(daily["chosen_mfe"].mean()),
        "mean_chosen_mae": float(daily["chosen_mae"].mean()),
    }


def summarize_by_choice(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.group_by(["target_metric", "margin", "allow_cash", "chosen_candidate_id", "choice_type"])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("chosen_ret").mean().alias("mean_chosen_ret"),
                pl.col("chosen_dailyized_ret").mean().alias("mean_chosen_dailyized_ret"),
                pl.col("lift_vs_base").mean().alias("mean_lift_vs_base"),
            ]
        )
        .sort(["target_metric", "allow_cash", "margin", "days"], descending=[False, False, False, True])
    )


def summarize_by_year(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.group_by(["target_metric", "margin", "allow_cash", "year"])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("base_ret").mean().alias("mean_base_ret"),
                pl.col("chosen_ret").mean().alias("mean_chosen_ret"),
                pl.col("lift_vs_base").mean().alias("mean_lift_vs_base"),
                (pl.col("choice_type") == "attack").sum().alias("attack_days"),
                (pl.col("choice_type") == "cash").sum().alias("cash_days"),
            ]
        )
        .sort(["target_metric", "allow_cash", "margin", "year"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Constrained base+attack/cash oracle lab")
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-candidate", default="manual_p2_k0p5_r0_6td")
    parser.add_argument(
        "--attack-candidates",
        type=parse_csv_list,
        default=DEFAULT_ATTACK_CANDIDATES,
    )
    parser.add_argument("--margins", type=parse_margins, default=DEFAULT_MARGINS)
    parser.add_argument(
        "--target-metrics",
        type=parse_csv_list,
        default=["top_ret", "top_ret_dailyized"],
    )
    args = parser.parse_args()

    started_at = datetime.now()
    daily = read_daily_candidates(args.oracle_dir)
    available = set(daily["candidate_id"].unique().to_list())
    required = {args.base_candidate, *args.attack_candidates}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"missing candidates in daily panel: {', '.join(missing)}")

    choice_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for metric_col in args.target_metrics:
        for margin in args.margins:
            for allow_cash in [False, True]:
                choice_daily = choose_constrained(
                    daily,
                    base_candidate=args.base_candidate,
                    attack_candidates=args.attack_candidates,
                    metric_col=metric_col,
                    margin=margin,
                    allow_cash=allow_cash,
                )
                choice_frames.append(choice_daily)
                summary_rows.append(summarize_choice(choice_daily))

    all_choices = pl.concat(choice_frames, how="vertical")
    summary_df = pl.DataFrame(summary_rows).sort(["target_metric", "allow_cash", "margin"])
    choice_summary = summarize_by_choice(all_choices)
    year_summary = summarize_by_year(all_choices)

    output_dir = args.output_root / timestamp_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_choices.write_csv(output_dir / "daily_constrained_choices.csv")
    summary_df.write_csv(output_dir / "strategy_summary.csv")
    choice_summary.write_csv(output_dir / "choice_summary.csv")
    year_summary.write_csv(output_dir / "year_summary.csv")

    summary: dict[str, Any] = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "oracle_dir": str(args.oracle_dir),
            "base_candidate": args.base_candidate,
            "attack_candidates": args.attack_candidates,
            "margins": args.margins,
            "target_metrics": args.target_metrics,
        },
        "metrics": {
            "strategy_summary": summary_df.to_dicts(),
            "choice_summary": choice_summary.to_dicts(),
            "year_summary": year_summary.to_dicts(),
        },
        "files": {
            "daily_constrained_choices": "daily_constrained_choices.csv",
            "strategy_summary": "strategy_summary.csv",
            "choice_summary": "choice_summary.csv",
            "year_summary": "year_summary.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Saved: {output_dir / 'summary.json'}")
    print("Strategy summary:")
    for row in summary_rows:
        print(
            f"- metric={row['target_metric']} margin={row['margin']:.0%} "
            f"cash={row['allow_cash']} mean={row['mean_chosen_ret'] * 100:+.2f}% "
            f"lift={row['mean_lift_vs_base'] * 100:+.2f}pp "
            f"attack={row['attack_days']} cash_days={row['cash_days']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
