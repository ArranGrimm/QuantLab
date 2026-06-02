from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger


def format_stock_code(raw_code: Any) -> str:
    code = str(raw_code).strip()
    if code.startswith(("sh.", "sz.", "bj.")):
        return code
    if code.endswith("_SH"):
        return f"sh.{code[:-3]}"
    if code.endswith("_SZ"):
        return f"sz.{code[:-3]}"
    if code.endswith("_BJ"):
        return f"bj.{code[:-3]}"
    code = code.zfill(6)
    if code.startswith("6"):
        return f"sh.{code}"
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    if code.startswith(("4", "8", "92")):
        return f"bj.{code}"
    return code


def refresh_em_sector_map(path: Path, *, request_sleep: float = 0.0) -> None:
    """通过 Baostock 拉取申万行业分类并写入 CSV。"""

    from utils.baostock_utils import get_stock_industry

    logger.info("Fetching industry classification via Baostock ...")
    df = get_stock_industry()

    if df.height == 0:
        raise RuntimeError("No industry data from Baostock")

    out = (
        df.select(
            pl.col("code").map_elements(format_stock_code, return_dtype=pl.Utf8).alias("code"),
            pl.col("code_name").alias("name"),
            pl.col("industry"),
        )
        .with_columns(pl.lit("").alias("industry_code"))
        .unique(subset=["code"], keep="first")
        .sort("code")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path)
    logger.info(f"Wrote {out.height:,} stock industry mappings to {path}")


def load_sector_map(path: Path, *, refresh: bool, request_sleep: float) -> pl.DataFrame:
    if refresh or not path.exists():
        refresh_em_sector_map(path, request_sleep=request_sleep)

    if not path.exists():
        raise FileNotFoundError(f"Sector map does not exist: {path}")

    df = pl.read_csv(path)
    required = {"code", "industry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sector map missing columns: {sorted(missing)}")

    return (
        df.select(
            pl.col("code").map_elements(format_stock_code, return_dtype=pl.Utf8).alias("code"),
            pl.col("industry").cast(pl.Utf8),
        )
        .filter(pl.col("industry").is_not_null())
        .unique(subset=["code"], keep="first")
    )


def load_daily_with_industry(db_path: Path, sector_map: pl.DataFrame, start_date: str) -> pl.DataFrame:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        daily = conn.execute(
            """
            SELECT code, date, open, high, low, close, volume, amount
            FROM v_stock_daily_qfq_qmt
            WHERE date >= ?
            ORDER BY code, date
            """,
            [start_date],
        ).pl()
    finally:
        conn.close()

    joined = daily.join(sector_map, on="code", how="left")
    missing_rows = joined.filter(pl.col("industry").is_null()).height
    if missing_rows:
        logger.warning(f"{missing_rows:,}/{joined.height:,} daily rows have no industry mapping")

    return joined.filter(pl.col("industry").is_not_null()).sort(["code", "date"])


def build_sector_tailwind_features(daily: pl.DataFrame) -> pl.DataFrame:
    stock = (
        daily.sort(["code", "date"])
        .with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1).over("code") - 1.0).alias("ret_1d"),
                (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("stock_ret_5d"),
                (pl.col("close") / pl.col("close").shift(10).over("code") - 1.0).alias("stock_ret_10d"),
                (pl.col("close") / pl.col("close").shift(20).over("code") - 1.0).alias("stock_ret_20d"),
                pl.col("close").rolling_mean(20).over("code").alias("stock_ma20"),
                pl.col("close").rolling_max(20).over("code").alias("stock_high_20"),
                pl.col("close").rolling_max(60).over("code").alias("stock_high_60"),
                pl.col("amount").rolling_mean(20).over("code").alias("stock_amount_ma20"),
            ]
        )
        .with_columns(
            [
                (pl.col("close") > pl.col("stock_ma20")).alias("stock_above_ma20"),
                (pl.col("close") >= pl.col("stock_high_20")).alias("stock_new_high_20"),
                (pl.col("close") >= pl.col("stock_high_60")).alias("stock_new_high_60"),
                (pl.col("amount") / pl.col("stock_amount_ma20")).alias("stock_amount_ratio_20"),
            ]
        )
    )

    sector = (
        stock.group_by(["date", "industry"])
        .agg(
            [
                pl.len().alias("sector_stock_count"),
                pl.col("ret_1d").mean().alias("sector_ret_1d"),
                pl.col("stock_above_ma20").mean().alias("sector_breadth_ma20"),
                pl.col("stock_new_high_20").mean().alias("sector_new_high_20"),
                pl.col("stock_new_high_60").mean().alias("sector_new_high_60"),
                pl.col("stock_amount_ratio_20").median().alias("sector_amount_ratio_20"),
            ]
        )
        .sort(["industry", "date"])
        .with_columns((1.0 + pl.col("sector_ret_1d")).cum_prod().over("industry").alias("sector_idx"))
        .with_columns(
            [
                (pl.col("sector_idx") / pl.col("sector_idx").shift(5).over("industry") - 1.0).alias(
                    "sector_ret_5d"
                ),
                (pl.col("sector_idx") / pl.col("sector_idx").shift(10).over("industry") - 1.0).alias(
                    "sector_ret_10d"
                ),
                (pl.col("sector_idx") / pl.col("sector_idx").shift(20).over("industry") - 1.0).alias(
                    "sector_ret_20d"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("sector_ret_5d").rank("average").over("date") / pl.len().over("date")).alias(
                    "sector_ret_5d_rank_pct"
                ),
                (pl.col("sector_ret_10d").rank("average").over("date") / pl.len().over("date")).alias(
                    "sector_ret_10d_rank_pct"
                ),
                (pl.col("sector_ret_20d").rank("average").over("date") / pl.len().over("date")).alias(
                    "sector_ret_20d_rank_pct"
                ),
            ]
        )
        .with_columns(
            (
                (pl.col("sector_ret_10d_rank_pct") >= 0.65)
                & (pl.col("sector_breadth_ma20") >= 0.45)
                & (pl.col("sector_amount_ratio_20") >= 0.90)
            ).alias("sector_tailwind_ok")
        )
    )

    return (
        stock.join(sector, on=["date", "industry"], how="left")
        .with_columns(
            [
                (pl.col("stock_ret_5d") - pl.col("sector_ret_5d")).alias("stock_rel_sector_ret_5d"),
                (pl.col("stock_ret_10d") - pl.col("sector_ret_10d")).alias("stock_rel_sector_ret_10d"),
                (pl.col("stock_ret_20d") - pl.col("sector_ret_20d")).alias("stock_rel_sector_ret_20d"),
            ]
        )
        .select(
            [
                "date",
                "code",
                "industry",
                "stock_ret_5d",
                "stock_ret_10d",
                "stock_ret_20d",
                "stock_rel_sector_ret_5d",
                "stock_rel_sector_ret_10d",
                "stock_rel_sector_ret_20d",
                "sector_stock_count",
                "sector_ret_5d",
                "sector_ret_10d",
                "sector_ret_20d",
                "sector_ret_5d_rank_pct",
                "sector_ret_10d_rank_pct",
                "sector_ret_20d_rank_pct",
                "sector_breadth_ma20",
                "sector_new_high_20",
                "sector_new_high_60",
                "sector_amount_ratio_20",
                "sector_tailwind_ok",
            ]
        )
    )


def build_sector_features(args: argparse.Namespace) -> pl.DataFrame:
    sector_map = load_sector_map(
        args.sector_map,
        refresh=args.refresh_sector_map,
        request_sleep=args.sector_map_request_sleep,
    )
    daily = load_daily_with_industry(args.qmt_db, sector_map, args.sector_start_date)
    return build_sector_tailwind_features(daily).select(
        [
            "date",
            "code",
            "industry",
            "sector_ret_5d_rank_pct",
            "sector_ret_10d_rank_pct",
            "sector_ret_20d_rank_pct",
            "stock_rel_sector_ret_5d",
            "stock_rel_sector_ret_10d",
            "stock_rel_sector_ret_20d",
            "sector_breadth_ma20",
            "sector_amount_ratio_20",
            "sector_tailwind_ok",
        ]
    )


def sector_rank_expr(args: argparse.Namespace) -> pl.Expr:
    if args.rank_source == "5d":
        return pl.col("sector_ret_5d_rank_pct")
    if args.rank_source == "10d":
        return pl.col("sector_ret_10d_rank_pct")
    if args.rank_source == "20d":
        return pl.col("sector_ret_20d_rank_pct")
    if args.rank_source == "mix_10_20":
        return (pl.col("sector_ret_10d_rank_pct") + pl.col("sector_ret_20d_rank_pct")) / 2.0
    raise ValueError(f"unknown rank source: {args.rank_source}")


def relative_confirm_expr(args: argparse.Namespace) -> pl.Expr:
    if args.relative_confirm == "none":
        return pl.lit(True)
    if args.relative_confirm == "rel5_under0":
        return pl.col("stock_rel_sector_ret_5d").fill_null(0.0) < 0.0
    if args.relative_confirm == "rel10_under0":
        return pl.col("stock_rel_sector_ret_10d").fill_null(0.0) < 0.0
    if args.relative_confirm == "rel20_under0":
        return pl.col("stock_rel_sector_ret_20d").fill_null(0.0) < 0.0
    raise ValueError(f"unknown relative confirm: {args.relative_confirm}")


def rank_source_token(value: str) -> str:
    return value.replace("_", "")


def threshold_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")
