"""Sector breadth diagnostic: can sector-level breadth distinguish healthy from narrow AMV bulls?

Key metric: daily fraction of sectors in bullish alignment (sector index > MA20 AND >40% of
stocks in sector above MA20). Joined with AMV regime phases and P3/PB3 trades.

Hypothesis: narrow AMV bulls (few sectors participating) explain P3 "aged+accelerating" losses,
while broad bulls explain the winners.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from scripts.amv_regime_phase_diagnostic import (
    build_amv_phase_frame,
    load_trades,
    DEFAULT_SLEEVES,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "QuantData" / "Ashare" / "qmt_data.duckdb"
DEFAULT_SECTOR_MAP = ROOT / "data" / "sector_map_em.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "amv_sector_breadth_diagnostic.json"


def load_sector_map(path: Path) -> pl.DataFrame:
    """Load sector map and convert code format to match DuckDB (sh.xxxxxx / sz.xxxxxx)."""
    df = pl.read_csv(str(path))
    # sector_map format: 600570_SH -> sh.600570
    df = df.with_columns(
        pl.when(pl.col("code").str.ends_with("_SH"))
        .then(pl.lit("sh.") + pl.col("code").str.replace("_SH", ""))
        .when(pl.col("code").str.ends_with("_SZ"))
        .then(pl.lit("sz.") + pl.col("code").str.replace("_SZ", ""))
        .otherwise(pl.col("code"))
        .alias("code_ddb")
    )
    return df.select(
        pl.col("code_ddb").alias("code"),
        pl.col("industry"),
    )


def load_daily_with_industry(
    db_path: Path,
    sector_map: pl.DataFrame,
    start_date: str = "2019-01-01",
) -> pl.DataFrame:
    """Load forward-adjusted daily close data with industry labels."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = conn.execute(
            f"""
            SELECT code, date, close
            FROM v_stock_daily_qfq_qmt
            WHERE date >= '{start_date}'
            ORDER BY code, date
            """
        ).pl()
    finally:
        conn.close()

    df = df.join(sector_map, on="code", how="left")
    null_industry = df.filter(pl.col("industry").is_null()).height
    if null_industry > 0:
        logger.warning(f"{null_industry} rows ({null_industry/df.height*100:.1f}%) have no industry mapping")
    return df.filter(pl.col("industry").is_not_null())


def compute_daily_breadth(df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily sector breadth: fraction of sectors in bullish alignment.

    A sector is "OK" when:
    - breadth (share of stocks > MA20) > 40%
    - sector index (equal-weight) > its MA20
    """
    # per-stock: is it above its 20-day MA?
    df = df.sort(["code", "date"]).with_columns(
        pl.col("close").rolling_mean(20).over("code").alias("ma20"),
        (pl.col("close") > pl.col("close").rolling_mean(20).over("code")).alias("is_strong"),
    )

    # per-sector per-day: breadth = share of strong stocks, sector_idx = cumprod of equal-weight returns
    sector_daily = (
        df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("code") - 1).alias("ret_1d"),
        )
        .group_by(["date", "industry"])
        .agg(
            [
                pl.col("is_strong").mean().alias("breadth"),
                pl.col("ret_1d").mean().alias("sector_ret"),
            ]
        )
        .sort(["industry", "date"])
        .with_columns(
            (1 + pl.col("sector_ret")).cum_prod().over("industry").alias("sector_idx"),
        )
        .with_columns(
            pl.col("sector_idx").rolling_mean(20).over("industry").alias("sector_idx_ma20"),
        )
        .with_columns(
            (
                (pl.col("breadth") > 0.4)
                & (pl.col("sector_idx") > pl.col("sector_idx_ma20"))
            ).alias("sector_ok")
        )
    )

    # aggregate to daily
    daily_breadth = (
        sector_daily.group_by("date")
        .agg(
            [
                pl.len().alias("total_sectors"),
                pl.col("sector_ok").sum().alias("sectors_ok"),
                pl.col("breadth").mean().alias("avg_breadth"),
                pl.col("breadth").median().alias("median_breadth"),
            ]
        )
        .with_columns(
            (pl.col("sectors_ok") / pl.col("total_sectors")).alias("breadth_ratio"),
        )
        .sort("date")
    )

    return daily_breadth


def main() -> None:
    parser = argparse.ArgumentParser(description="Sector breadth diagnostic for AMV regime")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dont-record-progress", action="store_true")
    args = parser.parse_args()

    # ── load sector data ───────────────────────────────────────────────
    logger.info("Loading sector map ...")
    sector_map = load_sector_map(args.sector_map)
    logger.info(f"  {sector_map.height} stocks, {sector_map['industry'].n_unique()} sectors")

    logger.info("Loading daily data with industry ...")
    df_daily = load_daily_with_industry(args.db, sector_map)
    logger.info(f"  {df_daily.height:,} rows, {df_daily['code'].n_unique()} stocks, "
                f"{df_daily['date'].min()} ~ {df_daily['date'].max()}")

    logger.info("Computing daily sector breadth ...")
    daily_breadth = compute_daily_breadth(df_daily)
    logger.info(f"  {daily_breadth.height} trading days")

    # ── load AMV phase frame and join ──────────────────────────────────
    logger.info("Building AMV phase frame and joining breadth ...")
    amv_phase = build_amv_phase_frame()
    joined = amv_phase.join(
        daily_breadth.select(["date", "total_sectors", "sectors_ok", "breadth_ratio",
                               "avg_breadth", "median_breadth"]),
        on="date", how="left",
    )

    # ── breadth summary stats ──────────────────────────────────────────
    bull_breadth = joined.filter(pl.col("is_bull_regime"))
    nonbull_breadth = joined.filter(~pl.col("is_bull_regime"))

    print("=" * 60)
    print("Sector Breadth Summary")
    print("=" * 60)
    print(f"  Bull days:  breadth_ratio mean={bull_breadth['breadth_ratio'].mean():.1%}, "
          f"median={bull_breadth['breadth_ratio'].median():.1%}")
    print(f"  Non-bull:   breadth_ratio mean={nonbull_breadth['breadth_ratio'].mean():.1%}, "
          f"median={nonbull_breadth['breadth_ratio'].median():.1%}")

    # ── breadth vs AMV regime phases ───────────────────────────────────
    print("\n--- Breadth by hindsight phase ---")
    for phase in ["early", "mid", "late", "pulse"]:
        pt = joined.filter(pl.col("hindsight_phase") == phase)
        print(f"  {phase:<6}: breadth_ratio={pt['breadth_ratio'].mean():.1%}, "
              f"avg_breadth={pt['avg_breadth'].mean():.1%}, n={pt.height}")

    # ── breadth vs forward phases ─────────────────────────────────────
    print("\n--- Breadth by forward duration-momentum ---")
    for dur in ["fresh", "young", "aged", "old"]:
        for mom in ["accelerating", "cruising", "stalling", "retreating"]:
            pt = joined.filter(
                (pl.col("fwd_duration_bucket") == dur)
                & (pl.col("fwd_momentum_bucket") == mom)
            )
            if pt.height > 5:
                print(f"  {dur:>5}_{mom:<14}: breadth_ratio={pt['breadth_ratio'].mean():.1%}, "
                      f"n={pt.height}")

    # ── KEY TEST: P3 trades by breadth ──────────────────────────────────
    print("\n" + "=" * 60)
    print("P3 Trade Performance by Sector Breadth at Entry")
    print("=" * 60)

    p3_cfg = DEFAULT_SLEEVES["P3_static_strict"]
    p3_trades = load_trades(ROOT / p3_cfg["trades"])
    join_cols = [
        "date", "hindsight_phase", "fwd_duration_bucket", "fwd_momentum_bucket",
        "fwd_phase", "regime_duration_days", "regime_maturity",
        "amv_slope_5d", "amv_acceleration", "amv_dd_from_high",
        "amv_neg_streak", "amv_ret_ma3", "amplitude_pct",
        "breadth_ratio", "avg_breadth", "median_breadth",
    ]
    p3_joined = p3_trades.join(
        joined.select(join_cols), left_on="entry_date", right_on="date", how="left"
    )

    # bread buckets
    p3_joined = p3_joined.with_columns(
        pl.when(pl.col("breadth_ratio").is_null())
        .then(pl.lit("no_data"))
        .when(pl.col("breadth_ratio") >= 0.6)
        .then(pl.lit("broad (>60%)"))
        .when(pl.col("breadth_ratio") >= 0.3)
        .then(pl.lit("moderate (30-60%)"))
        .otherwise(pl.lit("narrow (<30%)"))
        .alias("breadth_bucket")
    )

    total_pnl = float(p3_joined["pnl"].sum())
    total_n = p3_joined.height

    print(f"\nP3 total: {total_n}t, PnL={total_pnl:,.0f}")
    for bucket in ["broad (>60%)", "moderate (30-60%)", "narrow (<30%)", "no_data"]:
        pt = p3_joined.filter(pl.col("breadth_bucket") == bucket)
        if pt.height == 0:
            continue
        wr = pt.filter(pl.col("pnl") > 0).height / pt.height
        avg_pct = float(pt["pnl_pct"].mean())
        print(f"  {bucket:<20}: {pt.height:>3}t, PnL={float(pt['pnl'].sum()):>10,.0f}, "
              f"WR={wr:.1%}, avg_pnl_pct={avg_pct*100:.2f}%")

    # ── KEY: breadth × aged_forward ────────────────────────────────────
    print("\n--- P3 trades in 'aged' forward bucket × breadth ---")
    aged = p3_joined.filter(pl.col("fwd_duration_bucket") == "aged")
    for bucket in ["broad (>60%)", "moderate (30-60%)", "narrow (<30%)"]:
        pt = aged.filter(pl.col("breadth_bucket") == bucket)
        if pt.height == 0:
            continue
        wr = pt.filter(pl.col("pnl") > 0).height / pt.height
        print(f"  {bucket:<20}: {pt.height}t, PnL={float(pt['pnl'].sum()):>10,.0f}, WR={wr:.1%}")

    # ── KEY: breadth × aged_accelerating (the problematic cell) ───────
    print("\n--- P3 trades in 'aged_accelerating' × breadth (the problematic cell) ---")
    aged_accel = p3_joined.filter(pl.col("fwd_phase") == "aged_accelerating")
    for bucket in ["broad (>60%)", "moderate (30-60%)", "narrow (<30%)"]:
        pt = aged_accel.filter(pl.col("breadth_bucket") == bucket)
        if pt.height == 0:
            continue
        wr = pt.filter(pl.col("pnl") > 0).height / pt.height
        print(f"  {bucket:<20}: {pt.height}t, PnL={float(pt['pnl'].sum()):>10,.0f}, WR={wr:.1%}")

    # ── Breadth-based gating test ──────────────────────────────────────
    print("\n--- Breadth-based gating what-if ---")
    # rule: skip if breadth_ratio < 0.3 (narrow bull)
    rule_narrow = pl.col("breadth_ratio") < 0.3
    skipped = p3_joined.filter(rule_narrow)
    kept = p3_joined.filter(~rule_narrow)
    sn = skipped.height
    sp = float(skipped["pnl"].sum()) if sn else 0
    sk_wr = skipped.filter(pl.col("pnl") > 0).height / sn if sn else 0
    kp = float(kept["pnl"].sum())
    kwr = kept.filter(pl.col("pnl") > 0).height / kept.height
    big = skipped.filter(pl.col("pnl") > 20000).height
    print(f"  Skip narrow (<30%): {sn}t, PnL={sp:+,.0f}, WR={sk_wr:.1%} "
          f"-> kept PnL={kp:+,.0f} (net {kp-total_pnl:+,.0f}) big_killed={big}")

    # combinatorial: narrow AND aged
    rule_narrow_aged = (pl.col("breadth_ratio") < 0.3) & (pl.col("fwd_duration_bucket") == "aged")
    skipped2 = p3_joined.filter(rule_narrow_aged)
    kept2 = p3_joined.filter(~rule_narrow_aged)
    sn2 = skipped2.height
    sp2 = float(skipped2["pnl"].sum()) if sn2 else 0
    kp2 = float(kept2["pnl"].sum())
    big2 = skipped2.filter(pl.col("pnl") > 20000).height
    print(f"  Skip narrow & aged: {sn2}t, PnL={sp2:+,.0f} "
          f"-> kept PnL={kp2:+,.0f} (net {kp2-total_pnl:+,.0f}) big_killed={big2}")

    # ── also test on PB3 ──────────────────────────────────────────────
    print("\n--- PB3 gating with breadth ---")
    pb3_cfg = DEFAULT_SLEEVES["PB3_rolling_refill"]
    pb3_trades = load_trades(ROOT / pb3_cfg["trades"])
    pb3_joined = pb3_trades.join(
        joined.select(join_cols), left_on="entry_date", right_on="date", how="left"
    )
    pb3_total = float(pb3_joined["pnl"].sum())
    pb3_n = pb3_joined.height

    # PB3: narrow breadth skip
    pb3_narrow = pb3_joined.filter(pl.col("breadth_ratio") < 0.3)
    pb3_kept = pb3_joined.filter(pl.col("breadth_ratio") >= 0.3)
    print(f"  Narrow skip: {pb3_narrow.height}t, PnL={float(pb3_narrow['pnl'].sum()):+,.0f} "
          f"-> kept PnL={float(pb3_kept['pnl'].sum()):+,.0f} (net {float(pb3_kept['pnl'].sum())-pb3_total:+,.0f})")

    # ── write report ──────────────────────────────────────────────────
    output: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "breadth_summary": {
            "bull_mean": round(float(bull_breadth["breadth_ratio"].mean()), 4),
            "bull_median": round(float(bull_breadth["breadth_ratio"].median()), 4),
            "nonbull_mean": round(float(nonbull_breadth["breadth_ratio"].mean()), 4),
            "nonbull_median": round(float(nonbull_breadth["breadth_ratio"].median()), 4),
        },
        "p3_by_breadth": {
            bucket: {
                "trades": int(pt.height),
                "total_pnl": round(float(pt["pnl"].sum()), 2),
            }
            for bucket, pt in [
                ("broad", p3_joined.filter(pl.col("breadth_bucket") == "broad (>60%)")),
                ("moderate", p3_joined.filter(pl.col("breadth_bucket") == "moderate (30-60%)")),
                ("narrow", p3_joined.filter(pl.col("breadth_bucket") == "narrow (<30%)")),
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Report written to {args.output}")

    # ── progress.md ────────────────────────────────────────────────────
    if not args.dont_record_progress:
        from datetime import date
        today = date.today().isoformat()
        entry = (
            f"\n## {today}\n\n"
            f"### [AMV] Sector breadth diagnostic\n\n"
            f"- 目标: 板块宽度是否能解释 P3 在 AMV 牛市不同阶段的收益差异\n"
            f"- 脚本: `scripts/amv_sector_breadth_diagnostic.py`\n"
            f"- 产物: `reports/amv_sector_breadth_diagnostic.json`\n"
            f"- 行业数据: 东方财富 `sector_map_em.csv` (5553只), DuckDB `stock_daily` + `v_stock_daily_qfq_qmt` (2018-2026)\n"
            f"- 板块OK定义: 板块指数>MA20 且 板块内强势股占比(站上MA20)>40%\n"
        )
        progress_path = ROOT / "progress.md"
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Appended to progress.md")


if __name__ == "__main__":
    main()
