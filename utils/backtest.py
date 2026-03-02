"""
回测引擎模块
提供通用的策略回测功能，支持动态止损、择时过滤、排序选股等
"""
import polars as pl
from datetime import datetime
from typing import List, Tuple


def run_backtest(
    df_signals: pl.LazyFrame,
    return_days: List[int],
    loose_periods: List[Tuple[str, str]],
    top_n: int = 3,
    stop_loss_pct: float = 0.02,
    rank_by: str = "vol_ratio",
    rank_ascending: bool = True,
) -> pl.DataFrame:
    """
    实战回测引擎 (动态技术止损版)
    
    Args:
        df_signals: 信号 LazyFrame，需包含 b1_signal, code, date, open_adj, low_adj, close_adj, volume 等
        return_days: 持仓天数列表，如 [5, 10, 15, 20, 25, 30]
        loose_periods: 择时区间列表，如 [("2024-01-01", "2024-03-31"), ...]
        top_n: 每日选股数量上限
        stop_loss_pct: 止损幅度 (0.02 = 下浮2%)
        rank_by: 排序依据列名
        rank_ascending: 排序方向 (True=升序，值小优先)
    
    Returns:
        回测结果 DataFrame
    """
    print(f"🛠️ [Backtest] 启动实战回测：T+1开盘买入 + 动态止损({stop_loss_pct*100:.1f}%)...")

    # 1. 构建择时条件 (纯表达式，无需 collect)
    loose_conditions = []
    for s_str, e_str in loose_periods:
        try:
            s = datetime.strptime(s_str, "%Y-%m-%d").date()
            e = datetime.strptime(e_str, "%Y-%m-%d").date()
            loose_conditions.append(pl.col("date").is_between(s, e))
        except Exception:
            pass
    
    is_loose_expr = pl.any_horizontal(loose_conditions) if loose_conditions else pl.lit(False)

    # 2. 构建计算表达式
    expr_list = [
        # 择时标记
        is_loose_expr.cast(pl.Int32).alias("is_loose"),
        # T+1 开盘买入
        pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
        pl.col("low_adj").shift(-1).over("code").alias("buy_price_low"),
        # 冷却期标记
        (pl.col("b1_signal").cast(pl.Int32).shift(1).rolling_max(10).over("code").fill_null(0) == 0).alias("is_cool"),
        # 计算缩量比 (用于排序)
        (pl.col("volume") / pl.col("vol_40_mean")).alias("vol_ratio")
    ]

    # 动态生成未来收益列
    for rd in return_days:
        expr_list.append(
            pl.col("close_adj").shift(-rd).over("code").alias(f"close_{rd}d")
        )
        # 用收盘价判断止损 (而不是最低价，避免过度止损)
        expr_list.append(
            pl.col("close_adj").rolling_min(rd).shift(-rd).over("code").alias(f"low_min_{rd}d")
        )

    # 3. 收益计算表达式
    return_expr_list = []
    for rd in return_days:
        # 带止损收益
        return_expr_list.append(
            pl.when(pl.col(f"low_min_{rd}d") <= pl.col("stop_price_tech"))
              .then(pl.col("risk_pct"))
              .otherwise((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1)
              .alias(f"ret_{rd}d")
        )
        # 死拿对照组
        return_expr_list.append(
            ((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1).alias(f"ret_{rd}d_raw")
        )

    # 4. 核心回测逻辑 (单次 collect)
    return (
        df_signals
        .sort(["code", "date"])
        .with_columns([
            pl.col("volume").rolling_mean(40).over("code").alias("vol_40_mean"),
        ])
        .with_columns(expr_list)
        .with_columns(
            # 动态止损价
            (pl.col("buy_price_low") * (1 - stop_loss_pct)).alias("stop_price_tech"),
        )
        # 过滤
        .filter(pl.col("b1_signal"))
        .filter(pl.col("is_loose") == 1)
        # 排序选股
        .with_columns([
            pl.col(rank_by).rank("ordinal", descending=not rank_ascending).over("date").alias("daily_rank")
        ])
        .filter(pl.col("daily_rank") <= top_n)
        # 收益结算
        .with_columns([
            ((pl.col("stop_price_tech") / pl.col("buy_price")) - 1).alias("risk_pct")
        ])
        .with_columns(return_expr_list)
        .collect()
    )


def print_backtest_report(df_result: pl.DataFrame, return_days: List[int]) -> None:
    """
    打印回测报告
    
    Args:
        df_result: 回测结果 DataFrame
        return_days: 持仓天数列表
    """
    total_trades = df_result.height
    print("\n====== ⚔️ 实战回测报告 ======")
    print(f"✅ 交易信号总数: {total_trades}")

    if total_trades == 0:
        print("⚠️ 没有交易信号")
        return

    print("-" * 100)
    print(f"{'策略模式':<12} | {'胜率':<8} | {'均值':<8} | {'盈亏比(Odds)':<10} | {'期望值(Exp)':<10}")
    print("-" * 100)

    def print_metric(name: str, col_name: str):
        df_valid = df_result.filter(pl.col(col_name).is_not_null())
        cnt = df_valid.height
        if cnt == 0:
            return

        win_cnt = df_valid.filter(pl.col(col_name) > 0).height
        win_rate = win_cnt / cnt
        avg_ret = df_valid.select(pl.col(col_name).mean()).item()
        avg_win = df_valid.filter(pl.col(col_name) > 0).select(pl.col(col_name).mean()).item()
        avg_loss = df_valid.filter(pl.col(col_name) <= 0).select(pl.col(col_name).mean()).item()

        if avg_loss == 0 or avg_loss is None:
            odds = 99.9
        else:
            odds = abs(avg_win / avg_loss) if avg_win else 0

        if avg_win is None:
            avg_win = 0
        if avg_loss is None:
            avg_loss = 0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        print(f"{name:<12} | {win_rate*100:>6.1f}% | {avg_ret*100:>6.2f}% | {odds:>10.2f}x  | {expectancy*100:>8.2f}%")

    for rd in return_days:
        print_metric(f"持仓{rd}天", f"ret_{rd}d")
    print("-" * 100)
    for rd in return_days:
        print_metric(f"死拿{rd}天(对照)", f"ret_{rd}d_raw")
    print("-" * 100)


def run_backtest_short(
    df_signals: pl.LazyFrame,
    signal_col: str = "renko_signal",
    return_days: List[int] = None,
    loose_periods: List[Tuple[str, str]] = None,
    top_n: int = 5,
    stop_loss_pct: float = 0.03,
    rank_by: str = "vol_ratio",
    rank_ascending: bool = True,
    cooldown_days: int = 5,
) -> pl.DataFrame:
    """
    短线策略回测引擎 (通用版)

    与 run_backtest 的区别:
    - signal_col 可配置，不限于 b1_signal
    - 默认持仓更短 (3/5/7/10 天)
    - 冷却期更短 (5 天 vs 10 天)
    - 默认选股更多 (top_n=5)

    Args:
        df_signals: 信号 LazyFrame，需包含 signal_col, code, date, open_adj, low_adj, close_adj, volume
        signal_col: 信号列名
        return_days: 持仓天数列表，默认 [3, 5, 7, 10]
        loose_periods: 择时区间列表，为 None 则不做择时过滤
        top_n: 每日选股数量上限
        stop_loss_pct: 止损幅度 (0.03 = 3%)
        rank_by: 排序依据列名
        rank_ascending: 排序方向
        cooldown_days: 冷却期天数 (避免对同一只票连续开仓)

    Returns:
        回测结果 DataFrame
    """
    if return_days is None:
        return_days = [3, 5, 7, 10]

    print(f"🛠️ [Backtest] 启动短线回测：signal={signal_col}, 止损={stop_loss_pct*100:.1f}%...")

    # 1. 构建择时条件
    if loose_periods:
        loose_conditions = []
        for s_str, e_str in loose_periods:
            try:
                s = datetime.strptime(s_str, "%Y-%m-%d").date()
                e = datetime.strptime(e_str, "%Y-%m-%d").date()
                loose_conditions.append(pl.col("date").is_between(s, e))
            except Exception:
                pass
        is_loose_expr = pl.any_horizontal(loose_conditions) if loose_conditions else pl.lit(True)
    else:
        is_loose_expr = pl.lit(True)

    # 2. 构建计算表达式
    expr_list = [
        is_loose_expr.cast(pl.Int32).alias("is_loose"),
        pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
        pl.col("low_adj").shift(-1).over("code").alias("buy_price_low"),
        (pl.col(signal_col).cast(pl.Int32).shift(1).rolling_max(cooldown_days).over("code").fill_null(0) == 0).alias("is_cool"),
        (pl.col("volume") / pl.col("vol_40_mean")).alias("vol_ratio"),
    ]

    for rd in return_days:
        expr_list.append(
            pl.col("close_adj").shift(-rd).over("code").alias(f"close_{rd}d")
        )
        expr_list.append(
            pl.col("close_adj").rolling_min(rd).shift(-rd).over("code").alias(f"low_min_{rd}d")
        )

    # 3. 收益计算表达式
    return_expr_list = []
    for rd in return_days:
        return_expr_list.append(
            pl.when(pl.col(f"low_min_{rd}d") <= pl.col("stop_price_tech"))
              .then(pl.col("risk_pct"))
              .otherwise((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1)
              .alias(f"ret_{rd}d")
        )
        return_expr_list.append(
            ((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1).alias(f"ret_{rd}d_raw")
        )

    # 4. 核心回测逻辑
    return (
        df_signals
        .sort(["code", "date"])
        .with_columns(
            pl.col("volume").rolling_mean(40).over("code").alias("vol_40_mean"),
        )
        .with_columns(expr_list)
        .with_columns(
            (pl.col("buy_price_low") * (1 - stop_loss_pct)).alias("stop_price_tech"),
        )
        .filter(pl.col(signal_col))
        .filter(pl.col("is_loose") == 1)
        .with_columns(
            pl.col(rank_by).rank("ordinal", descending=not rank_ascending).over("date").alias("daily_rank")
        )
        .filter(pl.col("daily_rank") <= top_n)
        .with_columns(
            ((pl.col("stop_price_tech") / pl.col("buy_price")) - 1).alias("risk_pct")
        )
        .with_columns(return_expr_list)
        .collect()
    )


def analyze_yearly_intensity(df_result: pl.DataFrame, target_year: int) -> None:
    """
    年度交易强度分析
    
    Args:
        df_result: 回测结果 DataFrame
        target_year: 目标年份
    """
    print(f"\n====== 📊 {target_year} 年度交易强度分析 ======")

    try:
        df_year = df_result.filter(pl.col("date").dt.year() == target_year)
    except Exception:
        df_year = df_result.filter(pl.col("date").str.slice(0, 4) == str(target_year))

    total_signals = df_year.height

    if total_signals == 0:
        print(f"⚠️ {target_year} 年没有交易信号")
        return

    df_daily_counts = (
        df_year
        .group_by("date")
        .agg(pl.len().alias("trade_count"))
        .sort("trade_count", descending=True)
    )

    active_days = df_daily_counts.height
    avg_trades = df_daily_counts.select(pl.col("trade_count").mean()).item()
    median_trades = df_daily_counts.select(pl.col("trade_count").median()).item()
    max_trades = df_daily_counts.select(pl.col("trade_count").max()).item()

    print(f"📅 交易天数: {active_days} 天")
    print(f"🔫 总开枪数: {total_signals} 次")
    print("-" * 40)
    print(f"📉 平均每天: {avg_trades:.1f} 只")
    print(f"⚖️ 中位每天: {median_trades:.1f} 只")
    print(f"🔥 爆发极值: {max_trades} 只")
    print("-" * 40)

    print("🥵 最忙碌的 3 天:")
    for row in df_daily_counts.head(3).iter_rows(named=True):
        print(f"   {row['date']}: {row['trade_count']} 只")
