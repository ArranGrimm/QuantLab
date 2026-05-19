from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_CURRENT_RUN = Path("artifacts/amv_bull_pool_listwise_ranker/20260516_112948")
DEFAULT_PREVIOUS_RUN = Path("artifacts/amv_bull_pool_listwise_ranker/20260516_105316")


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("variants must not be empty")
    return variants


def load_selected(run_dir: Path, variants: list[str]) -> pl.DataFrame:
    path = run_dir / "ltr_topn_selected.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing LTR selected file: {path}")
    df = pl.read_csv(path, try_parse_dates=True)
    if "fwd_mae_6d" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("fwd_mae_6d"))
    return (
        df.filter(pl.col("feature_variant").is_in(variants))
        .with_columns(
            [
                pl.col("date").cast(pl.Date),
                pl.col("year").cast(pl.Int64),
            ]
        )
        .sort(["feature_variant", "date", "ltr_score"], descending=[False, False, True])
    )


def variant_year_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["feature_variant", "year"])
        .agg(
            [
                pl.len().alias("selected_rows"),
                pl.col("date").n_unique().alias("days"),
                pl.col("code").n_unique().alias("unique_codes"),
                pl.col("fwd_ret_6d").mean().alias("mean_ret"),
                pl.col("fwd_ret_6d").median().alias("median_ret"),
                (pl.col("fwd_ret_6d") > 0).mean().alias("win_rate"),
                (pl.col("fwd_mfe_6d") >= 0.15).mean().alias("hit15"),
                pl.col("fwd_mfe_6d").mean().alias("mean_mfe"),
                pl.col("fwd_mae_6d").mean().alias("mean_mae"),
            ]
        )
        .sort(["feature_variant", "year"])
    )


def variant_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        variant_year_summary(df)
        .group_by("feature_variant")
        .agg(
            [
                pl.col("selected_rows").sum().alias("selected_rows"),
                pl.col("days").sum().alias("days"),
                pl.col("unique_codes").mean().alias("avg_unique_codes_by_year"),
                pl.col("mean_ret").mean().alias("avg_year_mean_ret"),
                pl.col("win_rate").mean().alias("avg_year_win_rate"),
                pl.col("hit15").mean().alias("avg_year_hit15"),
                pl.col("mean_mfe").mean().alias("avg_year_mfe"),
                pl.col("mean_mae").mean().alias("avg_year_mae"),
            ]
        )
        .sort("avg_year_mean_ret", descending=True)
    )


def contribution_summary(df: pl.DataFrame, year: int, top_k: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in df["feature_variant"].unique().sort().to_list():
        df_variant = df.filter((pl.col("feature_variant") == variant) & (pl.col("year") == year))
        if df_variant.is_empty():
            continue
        ret_values = df_variant["fwd_ret_6d"].to_list()
        positive_sum = sum(value for value in ret_values if value > 0)
        net_sum = sum(ret_values)
        top = df_variant.sort("fwd_ret_6d", descending=True).head(top_k)
        worst = df_variant.sort("fwd_ret_6d").head(top_k)
        rows.append(
            {
                "feature_variant": variant,
                "year": year,
                "selected_rows": df_variant.height,
                "mean_ret": float(df_variant["fwd_ret_6d"].mean()),
                "median_ret": float(df_variant["fwd_ret_6d"].median()),
                "win_rate": float((df_variant["fwd_ret_6d"] > 0).mean()),
                "hit15": float((df_variant["fwd_mfe_6d"] >= 0.15).mean()),
                "mean_mfe": float(df_variant["fwd_mfe_6d"].mean()),
                "mean_mae": float(df_variant["fwd_mae_6d"].mean())
                if df_variant["fwd_mae_6d"].null_count() < df_variant.height
                else None,
                "positive_sum_ret": positive_sum,
                "net_sum_ret": net_sum,
                f"top{top_k}_sum_ret": float(top["fwd_ret_6d"].sum()),
                f"top{top_k}_share_of_positive": float(top["fwd_ret_6d"].sum() / positive_sum)
                if positive_sum != 0
                else None,
                f"top{top_k}_share_of_net": float(top["fwd_ret_6d"].sum() / net_sum)
                if net_sum != 0
                else None,
                f"worst{top_k}_sum_ret": float(worst["fwd_ret_6d"].sum()),
            }
        )
    return pl.DataFrame(rows).sort("mean_ret", descending=True)


def top_contributors(df: pl.DataFrame, year: int, top_k: int) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for variant in df["feature_variant"].unique().sort().to_list():
        df_variant = df.filter((pl.col("feature_variant") == variant) & (pl.col("year") == year))
        if df_variant.is_empty():
            continue
        frames.append(
            df_variant.sort("fwd_ret_6d", descending=True)
            .head(top_k)
            .with_columns(pl.lit("top").alias("bucket"))
        )
        frames.append(
            df_variant.sort("fwd_ret_6d")
            .head(top_k)
            .with_columns(pl.lit("worst").alias("bucket"))
        )
    return pl.concat(frames, how="vertical").sort(["feature_variant", "bucket", "fwd_ret_6d"])


def code_contributors(df: pl.DataFrame, year: int, top_k: int) -> pl.DataFrame:
    return (
        df.filter(pl.col("year") == year)
        .group_by(["feature_variant", "code"])
        .agg(
            [
                pl.len().alias("selected_count"),
                pl.col("fwd_ret_6d").sum().alias("sum_ret"),
                pl.col("fwd_ret_6d").mean().alias("mean_ret"),
                pl.col("fwd_mfe_6d").mean().alias("mean_mfe"),
                pl.col("fwd_mae_6d").mean().alias("mean_mae"),
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
            ]
        )
        .sort(["feature_variant", "sum_ret"], descending=[False, True])
        .group_by("feature_variant", maintain_order=True)
        .head(top_k)
    )


def robustness_summary(
    df: pl.DataFrame,
    *,
    year: int,
    max_code_repeats: int,
    return_cap: float,
    code_contribution_cap: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in df["feature_variant"].unique().sort().to_list():
        df_variant = df.filter((pl.col("feature_variant") == variant) & (pl.col("year") == year))
        if df_variant.is_empty():
            continue

        base_mean = float(df_variant["fwd_ret_6d"].mean())
        base_sum = float(df_variant["fwd_ret_6d"].sum())
        code_sums = (
            df_variant.group_by("code")
            .agg(
                [
                    pl.len().alias("selected_count"),
                    pl.col("fwd_ret_6d").sum().alias("sum_ret"),
                ]
            )
            .sort("sum_ret", descending=True)
        )
        top_code_row = code_sums.row(0, named=True)
        top_code = str(top_code_row["code"])
        top_code_sum = float(top_code_row["sum_ret"])

        remove_top = df_variant.filter(pl.col("code") != top_code)
        limited_repeats = (
            df_variant.sort(["feature_variant", "year", "code", "date"])
            .with_columns(
                pl.col("date")
                .rank(method="ordinal")
                .over(["feature_variant", "year", "code"])
                .alias("_code_pick_order")
            )
            .filter(pl.col("_code_pick_order") <= max_code_repeats)
        )
        capped_single = df_variant.with_columns(
            pl.when(pl.col("fwd_ret_6d") > return_cap)
            .then(pl.lit(return_cap))
            .otherwise(pl.col("fwd_ret_6d"))
            .alias("_capped_ret")
        )
        capped_code_sum = (
            code_sums.with_columns(
                pl.when(pl.col("sum_ret") > code_contribution_cap)
                .then(pl.lit(code_contribution_cap))
                .otherwise(pl.col("sum_ret"))
                .alias("_capped_code_sum")
            )["_capped_code_sum"]
            .sum()
        )
        remove_top_mean = float(remove_top["fwd_ret_6d"].mean())
        max_repeat_mean = float(limited_repeats["fwd_ret_6d"].mean())
        single_cap_mean = float(capped_single["_capped_ret"].mean())
        code_cap_mean = float(capped_code_sum / df_variant.height)

        rows.append(
            {
                "feature_variant": variant,
                "year": year,
                "selected_rows": df_variant.height,
                "base_mean_ret": base_mean,
                "base_net_sum_ret": base_sum,
                "top_code": top_code,
                "top_code_sum_ret": top_code_sum,
                "top_code_share_of_net": top_code_sum / base_sum if base_sum != 0 else None,
                "remove_top_code_rows": remove_top.height,
                "remove_top_code_mean_ret": remove_top_mean,
                "remove_top_code_delta": remove_top_mean - base_mean,
                "max_code_repeats": max_code_repeats,
                "max_repeat_rows": limited_repeats.height,
                "max_repeat_mean_ret": max_repeat_mean,
                "max_repeat_delta": max_repeat_mean - base_mean,
                "return_cap": return_cap,
                "single_pick_cap_mean_ret": single_cap_mean,
                "single_pick_cap_delta": single_cap_mean - base_mean,
                "code_contribution_cap": code_contribution_cap,
                "code_cap_mean_ret": code_cap_mean,
                "code_cap_delta": code_cap_mean - base_mean,
            }
        )
    return pl.DataFrame(rows).sort("base_mean_ret", descending=True)


def daily_sets(df: pl.DataFrame, variant: str) -> dict[Any, set[str]]:
    rows = (
        df.filter(pl.col("feature_variant") == variant)
        .group_by("date")
        .agg(pl.col("code").alias("codes"))
        .to_dicts()
    )
    return {row["date"]: set(row["codes"]) for row in rows}


def overlap_rows(left: pl.DataFrame, right: pl.DataFrame, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_variant, right_variant in pairs:
        left_sets = daily_sets(left, left_variant)
        right_sets = daily_sets(right, right_variant)
        for date in sorted(set(left_sets) & set(right_sets)):
            left_codes = left_sets[date]
            right_codes = right_sets[date]
            intersection = left_codes & right_codes
            union = left_codes | right_codes
            rows.append(
                {
                    "date": date,
                    "year": date.year,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "overlap_count": len(intersection),
                    "overlap_ratio": len(intersection) / len(left_codes) if left_codes else None,
                    "jaccard": len(intersection) / len(union) if union else None,
                }
            )
    return pl.DataFrame(rows).sort(["left_variant", "right_variant", "date"])


def overlap_summary(overlap: pl.DataFrame) -> pl.DataFrame:
    if overlap.is_empty():
        return overlap
    return (
        overlap.group_by(["left_variant", "right_variant", "year"])
        .agg(
            [
                pl.len().alias("days"),
                pl.col("overlap_count").mean().alias("avg_overlap_count"),
                pl.col("overlap_ratio").mean().alias("avg_overlap_ratio"),
                pl.col("jaccard").mean().alias("avg_jaccard"),
            ]
        )
        .sort(["left_variant", "right_variant", "year"])
    )


def daily_return_diff(
    current: pl.DataFrame,
    previous: pl.DataFrame,
    *,
    variant: str,
    previous_label: str,
    current_label: str,
) -> pl.DataFrame:
    def daily(df: pl.DataFrame, label: str) -> pl.DataFrame:
        return (
            df.filter(pl.col("feature_variant") == variant)
            .group_by("date")
            .agg(
                [
                    pl.col("fwd_ret_6d").mean().alias(f"{label}_mean_ret"),
                    pl.col("fwd_mfe_6d").mean().alias(f"{label}_mean_mfe"),
                ]
            )
        )

    return (
        daily(current, current_label)
        .join(daily(previous, previous_label), on="date", how="inner")
        .with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                (pl.col(f"{current_label}_mean_ret") - pl.col(f"{previous_label}_mean_ret")).alias(
                    "mean_ret_delta"
                ),
                (pl.col(f"{current_label}_mean_mfe") - pl.col(f"{previous_label}_mean_mfe")).alias(
                    "mean_mfe_delta"
                ),
            ]
        )
        .sort("date")
    )


def return_diff_summary(diff: pl.DataFrame) -> pl.DataFrame:
    return (
        diff.group_by("year")
        .agg(
            [
                pl.len().alias("days"),
                pl.col("mean_ret_delta").mean().alias("avg_mean_ret_delta"),
                pl.col("mean_mfe_delta").mean().alias("avg_mean_mfe_delta"),
                (pl.col("mean_ret_delta") > 0).mean().alias("improved_day_ratio"),
            ]
        )
        .sort("year")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AMV LTR topN selection attribution analysis")
    parser.add_argument("--current-run", type=Path, default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--previous-run", type=Path, default=DEFAULT_PREVIOUS_RUN)
    parser.add_argument("--variants", type=parse_variants, default=["no_risk", "kbar_momentum_state"])
    parser.add_argument("--compare-variant", default="no_risk")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-code-repeats", type=int, default=3)
    parser.add_argument("--return-cap", type=float, default=0.30)
    parser.add_argument("--code-contribution-cap", type=float, default=1.0)
    args = parser.parse_args()

    started_at = datetime.now()
    current = load_selected(args.current_run, args.variants)
    previous = load_selected(args.previous_run, [args.compare_variant])

    variant_pairs = list(combinations(args.variants, 2))
    current_overlap = overlap_rows(current, current, variant_pairs)
    before_after_overlap = overlap_rows(
        current,
        previous,
        [(args.compare_variant, args.compare_variant)],
    ).with_columns(
        [
            pl.lit(args.current_run.name).alias("left_run"),
            pl.lit(args.previous_run.name).alias("right_run"),
        ]
    )
    return_diff = daily_return_diff(
        current,
        previous,
        variant=args.compare_variant,
        previous_label="before_state_completion",
        current_label="after_state_completion",
    )

    output_dir = args.current_run / f"selection_analysis_{timestamp_token()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly = variant_year_summary(current)
    overall = variant_summary(current)
    contribution = contribution_summary(current, args.year, args.top_k)
    contributors = top_contributors(current, args.year, args.top_k)
    code_contribution = code_contributors(current, args.year, args.top_k)
    robustness = robustness_summary(
        current,
        year=args.year,
        max_code_repeats=args.max_code_repeats,
        return_cap=args.return_cap,
        code_contribution_cap=args.code_contribution_cap,
    )
    current_overlap_summary = overlap_summary(current_overlap)
    before_after_overlap_summary = overlap_summary(before_after_overlap)
    return_delta_summary = return_diff_summary(return_diff)

    overall.write_csv(output_dir / "variant_summary.csv")
    yearly.write_csv(output_dir / "variant_year_summary.csv")
    contribution.write_csv(output_dir / f"contribution_{args.year}.csv")
    contributors.write_csv(output_dir / f"top_worst_{args.year}.csv")
    code_contribution.write_csv(output_dir / f"top_code_contributors_{args.year}.csv")
    robustness.write_csv(output_dir / f"robustness_{args.year}.csv")
    current_overlap.write_csv(output_dir / "overlap_by_day.csv")
    current_overlap_summary.write_csv(output_dir / "overlap_summary.csv")
    before_after_overlap.write_csv(output_dir / "before_after_overlap_by_day.csv")
    before_after_overlap_summary.write_csv(output_dir / "before_after_overlap_summary.csv")
    return_diff.write_csv(output_dir / "before_after_daily_delta.csv")
    return_delta_summary.write_csv(output_dir / "before_after_delta_summary.csv")

    payload = {
        "generated_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (datetime.now() - started_at).total_seconds(),
        "config": {
            "current_run": str(args.current_run),
            "previous_run": str(args.previous_run),
            "variants": args.variants,
            "compare_variant": args.compare_variant,
            "year": args.year,
            "top_k": args.top_k,
            "max_code_repeats": args.max_code_repeats,
            "return_cap": args.return_cap,
            "code_contribution_cap": args.code_contribution_cap,
        },
        "files": {
            "variant_summary": "variant_summary.csv",
            "variant_year_summary": "variant_year_summary.csv",
            "contribution": f"contribution_{args.year}.csv",
            "top_worst": f"top_worst_{args.year}.csv",
            "top_code_contributors": f"top_code_contributors_{args.year}.csv",
            "robustness": f"robustness_{args.year}.csv",
            "overlap_by_day": "overlap_by_day.csv",
            "overlap_summary": "overlap_summary.csv",
            "before_after_overlap_by_day": "before_after_overlap_by_day.csv",
            "before_after_overlap_summary": "before_after_overlap_summary.csv",
            "before_after_daily_delta": "before_after_daily_delta.csv",
            "before_after_delta_summary": "before_after_delta_summary.csv",
        },
        "overall": overall.to_dicts(),
        "yearly": yearly.to_dicts(),
        "contribution": contribution.to_dicts(),
        "code_contribution": code_contribution.to_dicts(),
        "robustness": robustness.to_dicts(),
        "overlap_summary": current_overlap_summary.to_dicts(),
        "before_after_overlap_summary": before_after_overlap_summary.to_dicts(),
        "before_after_delta_summary": return_delta_summary.to_dicts(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Saved: {output_dir / 'summary.json'}")
    print("\n2026 contribution:")
    for row in contribution.to_dicts():
        print(
            f"- {row['feature_variant']}: mean={row['mean_ret'] * 100:+.2f}% "
            f"MFE={row['mean_mfe'] * 100:+.2f}% MAE={row['mean_mae'] * 100:+.2f}% "
            f"top{args.top_k}/positive={row[f'top{args.top_k}_share_of_positive']:.2f}"
        )
    print("\n2026 robustness:")
    for row in robustness.to_dicts():
        print(
            f"- {row['feature_variant']}: base={row['base_mean_ret'] * 100:+.2f}% "
            f"remove_top={row['remove_top_code_mean_ret'] * 100:+.2f}% "
            f"max_repeat={row['max_repeat_mean_ret'] * 100:+.2f}% "
            f"single_cap={row['single_pick_cap_mean_ret'] * 100:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
