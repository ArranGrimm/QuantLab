"""
Signal Export Tool - Export complete stock data with B1 signals for Rust backtesting

设计理念：
- 导出完整数据（所有股票所有日期），不做任何 filter
- 只做标记（b1_signal, pre_b1_signal, is_loose）
- Rust 端根据标记进行回测逻辑判断
"""
import polars as pl
from pathlib import Path
from datetime import datetime


def export_for_rust(
    df_full: pl.LazyFrame,
    output_path: str = "data/signals/market_data.parquet",
    loose_periods: list = None,
    stop_loss_pct: float = 0.03,
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
                # 标记：昨天是否是信号 (T日信号 → T+1日买入)
                pl.col("b1_signal").shift(1).over("code").fill_null(False).alias("pre_b1_signal"),
                # 标记：是否在活跃期
                is_loose_expr.alias("is_loose"),
                # 计算 vol_ratio (用于排序的默认选项)
                (pl.col("volume") / pl.col("vol_40_mean")).alias("vol_ratio"),
                # 止损价 = 当日最低价 * (1 - stop_loss_pct)
                (pl.col("low_adj") * (1 - stop_loss_pct)).alias("stop_price"),
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
        "volume",
        "WL",
        "YL",
        "J",
        "b1_signal",
        "pre_b1_signal",
        "is_loose",
        "vol_ratio",
        "stop_price",
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
