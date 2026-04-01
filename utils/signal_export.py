"""
Signal Export Tool - Export stock signals for Rust backtesting

支持两种策略:
1. B1 超跌反转 — 事件驱动信号 (b1_signal)
2. 截面轮动模型 — 日频截面打分 (score + rank)

设计理念：
- 导出完整数据, Rust 端负责回测逻辑
- B1: 标记信号日, Rust 做买入/止损/止盈
- 截面模型: 导出每日全 universe 打分, Rust 做 Top-N 选股/持仓管理
"""
import polars as pl
from pathlib import Path
from datetime import datetime


def export_for_rust(
    df_full: pl.LazyFrame,
    output_path: str = "data/signals/market_data.parquet",
    loose_periods: list = None,
    start_date: str = None,
    extra_sort_cols: list = None,
) -> str:
    """
    Export complete market data with B1 signals for Rust backtesting

    Args:
        df_full: LazyFrame from calc_b1_factors_opt (contains ALL rows, not just signals)
        output_path: Output parquet file path
        loose_periods: Active period list, e.g. [("2025-04-09", "2025-09-04"), ...]
        stop_loss_pct: Stop loss percentage for calculating stop price
        start_date: Only export data >= this date, e.g. "2025-01-01"
        extra_sort_cols: Extra columns to export for sorting, e.g. ["Bias_C_WL", "J"]

    Returns:
        Exported file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Filter by start_date if provided
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        df_full = df_full.filter(pl.col("date") >= pl.lit(start))
        print(f"Filtering data >= {start_date}")

    # Build is_loose marker
    if loose_periods:
        loose_exprs = []
        for s_str, e_str in loose_periods:
            s = datetime.strptime(s_str, "%Y-%m-%d").date()
            e = datetime.strptime(e_str, "%Y-%m-%d").date()
            loose_exprs.append(pl.col("date").is_between(pl.lit(s), pl.lit(e)))
        is_loose_expr = pl.any_horizontal(*loose_exprs)
    else:
        is_loose_expr = pl.lit(True)

    print("Processing data...")
    df_export = (
        df_full.sort(["code", "date"])
        .with_columns([
            pl.col("volume").rolling_mean(40).over("code").alias("vol_40_mean"),
        ])
        .with_columns(
            [
                # 标记：昨天收盘价
                pl.col("close_adj").shift(1).over("code").fill_null(pl.col("close_adj")).alias("pre_close_adj"),
                # 标记：昨天是否是信号 (T日信号 → T+1日买入)
                pl.col("b1_signal").shift(1).over("code").fill_null(False).alias("pre_b1_signal"),
                # 标记：是否在活跃期
                is_loose_expr.alias("is_loose"),
                # 计算 vol_ratio (用于排序的默认选项)
                (pl.col("volume") / pl.col("vol_40_mean")).alias("vol_ratio")
            ]
        )
    )

    # Select columns needed by Rust
    required_cols = [
        "code",
        "date",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "pre_close_adj",
        "volume",
        "WL",
        "YL",
        "J",
        "b1_signal",
        "pre_b1_signal",
        "is_loose",
        "vol_ratio"
    ]
    
    # Add extra sort columns if specified
    if extra_sort_cols:
        for col in extra_sort_cols:
            if col not in required_cols:
                required_cols.append(col)

    # Collect and select
    df_collected = df_export.collect()
    available_cols = [c for c in required_cols if c in df_collected.columns]

    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        print(f"Warning: Missing columns: {missing}")

    df_final = df_collected.select(available_cols)

    # Write to parquet
    df_final.write_parquet(output_file)

    # Print summary
    total_rows = df_final.height
    signal_rows = df_final.filter(pl.col("b1_signal")).height
    loose_signal_rows = df_final.filter(pl.col("b1_signal") & pl.col("is_loose")).height
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    print(f"\n=== Export Summary ===")
    print(f"File: {output_file}")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"B1 signals: {signal_rows}")
    print(f"B1 signals in loose periods: {loose_signal_rows}")

    return str(output_file)


def export_rotation_scores(
    df_scores: pl.DataFrame,
    output_path: str = "data/signals/rotation_scores.parquet",
    top_n: int = 20,
) -> str:
    """
    Export cross-section rotation model scores for Rust backtesting.

    Python 端只负责打分, Rust 端负责:
      - 每日读取 Top-N 候选
      - 买入/卖出决策 (止损/止盈/排名退出)
      - 仓位管理、交易成本

    Args:
        df_scores: DataFrame, 必须包含:
            date, code, score, open_adj, high_adj, low_adj, close_adj
            可选: volume, market_cap_100m
        output_path: 输出 Parquet 路径
        top_n: 每日 Top-N 标记 (is_top_n 列, 供 Rust 参考)

    Returns:
        Exported file path

    Output schema:
        date, code, score, rank, is_top_n,
        open_adj, high_adj, low_adj, close_adj, pre_close_adj,
        [volume, market_cap_100m]
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    required = ["date", "code", "score", "open_adj", "high_adj", "low_adj", "close_adj"]
    missing = [c for c in required if c not in df_scores.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Processing rotation scores...")

    df_valid = df_scores.filter(
        pl.col("score").is_not_null() & pl.col("score").is_not_nan()
    )

    df_export = (
        df_valid
        .sort(["code", "date"])
        .with_columns(
            pl.col("close_adj").shift(1).over("code")
                .fill_null(pl.col("close_adj"))
                .alias("pre_close_adj"),
        )
        .with_columns(
            pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt16)
                .alias("rank"),
        )
        .with_columns(
            (pl.col("rank") <= top_n).alias("is_top_n"),
        )
    )

    out_cols = [
        "date", "code", "score", "rank", "is_top_n",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    ]
    for opt_col in ["volume", "market_cap_100m"]:
        if opt_col in df_scores.columns:
            out_cols.append(opt_col)

    df_final = df_export.select(out_cols)
    df_final.write_parquet(output_file)

    total_rows = df_final.height
    unique_dates = df_final.select(pl.col("date").n_unique()).item()
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    top_n_rows = df_final.filter(pl.col("is_top_n")).height
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    print(f"\n=== Rotation Scores Export ===")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Total rows: {total_rows:,}")
    print(f"Trading days: {unique_dates}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"Top-{top_n} signals: {top_n_rows:,} ({top_n_rows/unique_dates:.1f}/day avg)")

    return str(output_file)


def export_renko_scores(
    df_scores: pl.DataFrame,
    output_path: str = "data/signals/renko_scores.parquet",
    top_n: int = 20,
) -> str:
    """
    Export full-universe Renko model scores for T+1 open execution.

    时钟:
      - T 日收盘后得到 score/rank
      - Rust 在 T+1 日开盘使用 pre_score/pre_rank 买入
      - T+1 日收盘使用当日 rank 做持仓管理

    Required columns:
        date, code, score, open_adj, high_adj, low_adj, close_adj

    Optional columns:
        volume, market_cap_100m, renko_signal / is_renko
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    required = ["date", "code", "score", "open_adj", "high_adj", "low_adj", "close_adj"]
    missing = [c for c in required if c not in df_scores.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Processing renko scores...")

    df_export = (
        df_scores
        .sort(["code", "date"])
        .with_columns(
            pl.col("close_adj").shift(1).over("code")
                .fill_null(pl.col("close_adj"))
                .alias("pre_close_adj"),
        )
        .with_columns(
            pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("date")
                .cast(pl.UInt16)
                .alias("rank"),
        )
        .with_columns(
            (pl.col("rank") <= top_n).alias("is_top_n"),
        )
        .with_columns([
            pl.col("score").shift(1).over("code").fill_null(-999.0).alias("pre_score"),
            pl.col("rank").shift(1).over("code").fill_null(9999).cast(pl.UInt16).alias("pre_rank"),
            pl.col("is_top_n").shift(1).over("code").fill_null(False).alias("pre_is_top_n"),
        ])
    )

    out_cols = [
        "date", "code",
        "score", "rank", "is_top_n",
        "pre_score", "pre_rank", "pre_is_top_n",
        "open_adj", "high_adj", "low_adj", "close_adj", "pre_close_adj",
    ]
    for opt_col in ["volume", "market_cap_100m", "is_renko", "renko_signal"]:
        if opt_col in df_scores.columns:
            out_cols.append(opt_col)

    df_final = df_export.select(out_cols)
    df_final.write_parquet(output_file)

    total_rows = df_final.height
    unique_dates = df_final.select(pl.col("date").n_unique()).item()
    unique_codes = df_final.select(pl.col("code").n_unique()).item()
    top_n_rows = df_final.filter(pl.col("is_top_n")).height
    pre_top_n_rows = df_final.filter(pl.col("pre_is_top_n")).height
    date_range = (
        df_final.select(pl.col("date").min()).item(),
        df_final.select(pl.col("date").max()).item(),
    )

    print(f"\n=== Renko Scores Export ===")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Total rows: {total_rows:,}")
    print(f"Trading days: {unique_dates}")
    print(f"Unique stocks: {unique_codes}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"Top-{top_n} signals: {top_n_rows:,} ({top_n_rows/unique_dates:.1f}/day avg)")
    print(f"Pre Top-{top_n} signals: {pre_top_n_rows:,} ({pre_top_n_rows/unique_dates:.1f}/day avg)")

    return str(output_file)


def validate_export(filepath: str) -> dict:
    """Validate exported parquet file"""
    df = pl.read_parquet(filepath)

    result = {
        "filepath": filepath,
        "rows": df.height,
        "columns": df.columns,
        "date_range": (
            df.select(pl.col("date").min()).item(),
            df.select(pl.col("date").max()).item(),
        ),
        "unique_codes": df.select(pl.col("code").n_unique()).item(),
        "signal_count": df.filter(pl.col("b1_signal")).height,
        "loose_signal_count": df.filter(pl.col("b1_signal") & pl.col("is_loose")).height,
    }

    print(f"\n=== Validation ===")
    print(f"Rows: {result['rows']:,}")
    print(f"Date Range: {result['date_range']}")
    print(f"Unique Codes: {result['unique_codes']}")
    print(f"B1 Signals: {result['signal_count']}")
    print(f"Loose B1 Signals: {result['loose_signal_count']}")

    return result
