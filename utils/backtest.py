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
        (pl.col("volume") / pl.col("avg40")).alias("vol_ratio")
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


# ==============================================================================
# 非重叠回测 (动态卖出版)
# ==============================================================================
def run_backtest_realistic(
    df_signals: pl.LazyFrame,
    loose_periods: List[Tuple[str, str]],
    stop_loss_pct: float = 0.03,
    max_hold_days: int = 30,
) -> pl.DataFrame:
    """
    非重叠回测 (动态卖出版)
    
    特点：
    1. 每天只选 1 只 (top_n=1)
    2. 持仓期间不开新仓 (非重叠)
    3. 动态卖出条件：止损 或 跌破WL 或 最大持有期
    
    Args:
        df_signals: 信号 LazyFrame，需包含 b1_signal, WL, close_adj, avg40 等
        loose_periods: 择时区间列表
        stop_loss_pct: 止损幅度 (0.03 = 3%)
        max_hold_days: 最大持有天数
    
    Returns:
        回测结果 DataFrame，包含每笔交易的详细信息
    """
    print(f"🛠️ [Realistic Backtest] 启动非重叠回测：止损{stop_loss_pct*100:.1f}% + 跌破WL卖出...")

    # 1. 构建择时条件
    loose_conditions = []
    for s_str, e_str in loose_periods:
        try:
            s = datetime.strptime(s_str, "%Y-%m-%d").date()
            e = datetime.strptime(e_str, "%Y-%m-%d").date()
            loose_conditions.append(pl.col("date").is_between(s, e))
        except Exception:
            pass
    
    is_loose_expr = pl.any_horizontal(loose_conditions) if loose_conditions else pl.lit(False)

    # 2. 准备信号数据 (top_n=1)
    df_candidates = (
        df_signals
        .with_columns([
            is_loose_expr.alias("is_loose"),
            pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
            (pl.col("volume") / pl.col("avg40")).alias("vol_ratio"),
        ])
        .filter(pl.col("b1_signal") & pl.col("is_loose"))
        .filter(pl.col("buy_price").is_not_null() & (pl.col("buy_price") > 0))
        .with_columns([
            pl.col("vol_ratio").rank("ordinal", descending=False).over("date").alias("daily_rank")
        ])
        .filter(pl.col("daily_rank") == 1)  # 每天只选 1 只
        .select(["code", "date", "buy_price"])
        .sort("date")
        .collect()
    )

    # 3. 获取全量价格数据 (用于卖出判断)
    df_prices = (
        df_signals
        .select(["code", "date", "close_adj", "WL"])
        .collect()
    )

    print(f"📊 候选信号数: {df_candidates.height}")

    # 4. 非重叠回测主循环
    results = []
    current_exit_date = None  # 当前持仓的卖出日期

    for row in df_candidates.iter_rows(named=True):
        signal_date = row["date"]
        code = row["code"]
        buy_price = row["buy_price"]

        # 检查是否在持仓期间 (非重叠)
        if current_exit_date is not None and signal_date <= current_exit_date:
            continue  # 还在持仓，跳过此信号

        # 计算止损价
        stop_price = buy_price * (1 - stop_loss_pct)

        # 获取未来 max_hold_days 天的数据
        future = (
            df_prices
            .filter((pl.col("code") == code) & (pl.col("date") > signal_date))
            .sort("date")
            .head(max_hold_days)
        )

        if future.height == 0:
            continue  # 没有未来数据

        # 找卖出点
        exit_date = None
        exit_price = None
        exit_reason = None

        for future_row in future.iter_rows(named=True):
            close = future_row["close_adj"]
            wl = future_row["WL"]

            # 止损判断
            if close < stop_price:
                exit_date = future_row["date"]
                exit_price = close
                exit_reason = "止损"
                break
            # 跌破 WL 判断
            elif wl is not None and close < wl:
                exit_date = future_row["date"]
                exit_price = close
                exit_reason = "跌破WL"
                break

        # 如果没有触发条件，最后一天强制卖出
        if exit_date is None and future.height > 0:
            last_row = future.tail(1).row(0, named=True)
            exit_date = last_row["date"]
            exit_price = last_row["close_adj"]
            exit_reason = "最大持有期"

        if exit_date and exit_price:
            ret = (exit_price - buy_price) / buy_price
            hold_days = (exit_date - signal_date).days

            results.append({
                "signal_date": signal_date,
                "code": code,
                "buy_price": buy_price,
                "stop_price": stop_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_days": hold_days,
                "return": ret,
            })

            # 更新当前持仓的卖出日期
            current_exit_date = exit_date

    print(f"✅ 有效交易数: {len(results)}")
    return pl.DataFrame(results)


def print_realistic_report(df_result: pl.DataFrame) -> None:
    """
    打印非重叠回测报告
    
    Args:
        df_result: run_backtest_realistic 的返回结果
    """
    total = df_result.height
    if total == 0:
        print("⚠️ 没有交易记录")
        return

    print("\n====== 📈 非重叠回测报告 ======")
    print(f"✅ 总交易笔数: {total}")

    # 整体统计
    win_cnt = df_result.filter(pl.col("return") > 0).height
    win_rate = win_cnt / total
    avg_ret = df_result.select(pl.col("return").mean()).item()
    avg_hold = df_result.select(pl.col("hold_days").mean()).item()

    avg_win = df_result.filter(pl.col("return") > 0).select(pl.col("return").mean()).item()
    avg_loss = df_result.filter(pl.col("return") <= 0).select(pl.col("return").mean()).item()

    if avg_loss is None or avg_loss == 0:
        odds = 99.9
    else:
        odds = abs(avg_win / avg_loss) if avg_win else 0

    print("-" * 60)
    print(f"📊 胜率: {win_rate*100:.1f}%")
    print(f"📊 平均收益: {avg_ret*100:.2f}%")
    print(f"📊 平均持仓: {avg_hold:.1f} 天")
    print(f"📊 盈亏比: {odds:.2f}x")
    print("-" * 60)

    # 按卖出原因统计
    print("\n📋 卖出原因分布:")
    reason_stats = (
        df_result
        .group_by("exit_reason")
        .agg([
            pl.len().alias("count"),
            pl.col("return").mean().alias("avg_return"),
            pl.col("hold_days").mean().alias("avg_hold"),
        ])
        .sort("count", descending=True)
    )

    for row in reason_stats.iter_rows(named=True):
        reason = row["exit_reason"]
        count = row["count"]
        pct = count / total * 100
        avg_r = row["avg_return"] * 100
        avg_h = row["avg_hold"]
        print(f"   {reason}: {count}笔 ({pct:.1f}%) | 均收益{avg_r:+.2f}% | 均持仓{avg_h:.1f}天")
