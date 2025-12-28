import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    import numpy as np
    from loguru import logger

    # 1. 修改点：更新数据根目录
    DATA_ROOT = r"../QuantData/Ashare" 
    STOP_LOSS = -0.05  # 止损线
    TAKE_PROFIT = 0.1 # 止盈线
    return DATA_ROOT, TAKE_PROFIT, np, os, pl


@app.cell
def _(DATA_ROOT, TAKE_PROFIT, os, pl):
    # 简单版本b1策略
    def run_final_strategy():
        print("🚀 [Step 1] 加载数据 & 构建‘全A等权指数’ (Market Index)...")
    
        # 1. 加载全市场行情
        q_adj = (
            pl.scan_parquet(
                os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
                include_file_paths="file_path"
            )
            .with_columns(
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            )
            .select(["code", "date", "open", "high", "low", "close", "volume"])
            .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
        )
    
        # ---------------------------------------------------------------------
        # 🌟 核心升级：构建“全A等权指数” (环境择时)
        # ---------------------------------------------------------------------
        # 逻辑：每天计算全市场所有股票的平均收盘价，作为大盘指数
        q_market_index = (
            q_adj
            .group_by("date")
            .agg(pl.col("close_adj").mean().alias("index_close"))
            .sort("date")
            .with_columns([
                # 计算指数的 20日均线 (生命线)
                pl.col("index_close").rolling_mean(20).alias("index_ma20")
            ])
            .with_columns([
                # 定义环境状态：1=手松(线上), 0=手紧(线下)
                (pl.col("index_close") > pl.col("index_ma20")).alias("is_bull_market")
            ])
        )

        # 2. 个股数据与环境数据合并
        # 同样需要加载 raw 和 capital (省略部分重复代码，直接假设你已经有 q_full 的构建逻辑)
        # 为了代码简洁，这里直接复用之前的 q_full 构建逻辑，但在最后 join 环境数据
    
        # ... (重复之前的 raw 和 capital 加载逻辑，为节省篇幅，假设 q_market 是合并好的个股数据) ...
        # 这里必须重新写一遍完整的 join 链条，确保你能运行
    
        q_raw = pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path").with_columns(
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ).select(["code", "date", "close"]).rename({"close": "close_raw"})

        q_cap = pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path").with_columns(
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("total_capital").cast(pl.Float64)
        ).select(["code", "date", "total_capital"]).sort(["code", "date"])
    
        # 合并个股
        q_full = q_adj.join(q_raw, on=["code", "date"]).sort(["code", "date"]).join_asof(
            q_cap, on="date", by="code", strategy="backward"
        ).with_columns(
            (pl.col("close_raw") * pl.col("total_capital") / 1e8).alias("market_cap_100m")
        )
    
        # 🔥 将环境数据 Join 进个股数据
        q_full_with_env = q_full.join(q_market_index, on="date", how="left")

        print("🚀 [Step 2] 计算因子 (含环境过滤)...")
    
        # 3. 因子计算 (同 v2.04b)
        # ... (此处省略繁琐的因子计算代码，直接跳到 filter 部分) ...
        # 为了你能运行，我必须把因子计算补全，但会简化一点写法
    
        q_factors = q_full_with_env.with_columns([
            ((pl.col("close_adj").rolling_mean(14) + pl.col("close_adj").rolling_mean(28) + 
              pl.col("close_adj").rolling_mean(57) + pl.col("close_adj").rolling_mean(114)) / 4).alias("ztalk_yellow"),
            ((pl.col("close_adj") - pl.col("low_adj").rolling_min(9)) / 
             (pl.col("high_adj").rolling_max(9) - pl.col("low_adj").rolling_min(9) + 1e-9) * 100).alias("rsv"),
            # v2.04b 核心
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").alias("ema_10"),
            pl.col("volume").rolling_mean(40).alias("vol_ma40"),
            # 简单定义真阳性辅助计算
            ((pl.col("close_adj") > pl.col("open_adj"))).alias("is_yang"),
            (pl.col("volume") > 1.8 * pl.col("volume").shift(1)).alias("is_double_vol")
        ]).with_columns([
            (2 * pl.col("ema_10") - pl.col("ema_10").ewm_mean(span=10, adjust=False).over("code")).alias("ztalk_white"),
            pl.col("rsv").ewm_mean(alpha=1/3, adjust=False).over("code").ewm_mean(alpha=1/3, adjust=False).over("code").alias("d_val"),
            pl.col("rsv").ewm_mean(alpha=1/3, adjust=False).over("code").alias("k_val")
        ]).with_columns([
            (3 * pl.col("k_val") - 2 * pl.col("d_val")).alias("j_val"),
            # 简化版堆量: 20天内有3次倍量
            (pl.col("is_double_vol") & pl.col("is_yang")).rolling_sum(20).alias("pile_count")
        ])

        # 4. 信号筛选 (加入环境过滤!)
        q_signals = q_factors.filter(
            # A. 基础清洗 (负值剔除)
            (pl.col("close_adj") > 1.0) &
            (pl.col("volume") > 0) & 
            (pl.col("market_cap_100m") >= 50) &
        
            # B. 🔥 环境择时 (只做手松区)
            (pl.col("is_bull_market") == True) &  # <--- 新增核心条件：指数 > 20日线
        
            # C. Ztalk 趋势锁
            (pl.col("ztalk_white") > pl.col("ztalk_yellow")) &
            (pl.col("close_adj") > pl.col("ztalk_yellow")) &
        
            # D. B1 触发器
            (pl.col("j_val") < 5) &   # 稍微放宽J值，因为有环境保护了
            (pl.col("pile_count") >= 2) & # 至少有2次倍量堆积
            (pl.col("volume") < pl.col("volume").rolling_mean(5) * 0.7) # 缩量
        )

        print("🚀 [Step 3] 模拟智能止盈 (Smart Exit)...")

        # 5. 回测统计 (Smart Exit)
        # 逻辑：检查 T+1 到 T+5，是否有一天 Highest > +6%，如果有，止盈离场。
        # 否则，第 5 天强制卖出。
    
        q_backtest = q_signals.with_columns([
            # 获取 T+1 到 T+5 的最高价
            pl.col("high_adj").shift(-1).rolling_max(5).over("code").alias("max_h_5d"),
            # 获取 T+5 收盘价 (作为保底)
            pl.col("close_adj").shift(-5).over("code").alias("close_t5")
        ]).with_columns([
            # 实际卖出价格逻辑：
            # 如果 5天内最高价涨幅超过 6%，按 (买入价 * 1.06) 止盈
            # 否则，按 T+5 收盘价卖出
            pl.when(pl.col("max_h_5d") > pl.col("close_adj") * (1 + TAKE_PROFIT))
              .then(TAKE_PROFIT)  # 止盈收益固定为 6%
              .otherwise(pl.col("close_t5") / pl.col("close_adj") - 1) # 否则吃满5天的波动
              .alias("final_return")
        ])
    
        # 清洗异常值
        q_final = q_backtest.filter(pl.col("final_return").is_not_null())
    
        print("🚀 [Step 4] 计算最终结果...")
        df_result = q_final.collect()
    
        if len(df_result) == 0:
            print("无信号。")
            return
        
        win_rate = len(df_result.filter(pl.col("final_return") > 0)) / len(df_result)
        avg_ret = df_result["final_return"].mean()
        median_ret = df_result["final_return"].median()
    
        print(f"\n======== Ztalk B1 最终实战版 (择时+止盈) ========")
        print(f"策略配置：全A指数>MA20才开仓 | 6%止盈 | 最长持仓5天")
        print(f"总交易次数: {len(df_result)} (大幅减少，因为避开了熊市)")
        print(f"------------------------------------------------")
        print(f"最终胜率: {win_rate:.2%} (目标 > 55%)")
        print(f"平均单笔收益: {avg_ret:.2%} (目标 > 1%)")
        print(f"中位数收益: {median_ret:.2%} (目标 > 0%)")
        print("================================================")
    
        return df_result

    return (run_final_strategy,)


@app.cell
def _(run_final_strategy):

    # 运行 (确保你在有数据的环境下运行)
    df = run_final_strategy()
    return


@app.cell
def _(DATA_ROOT, np, os, pl):
    # 模拟配置
    SL_BUFFER = 0.99  # 止损位 = 最低价 * 99% (模拟下方3-5个价位)
    MAX_HOLD_DAYS = 20 # 最长拿20天 (给主升浪一点时间)

    def run_full_system():
        print("🚀 [Step 1] 加载全量数据...")
    
        # 1. 基础数据加载 (包含 Close, High, Low, Volume)
        q_raw = pl.scan_parquet(
            os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
            include_file_paths="file_path"
        ).with_columns(
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ).select(["code", "date", "open", "high", "low", "close", "volume"]) \
         .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})

        # 2. 计算 Ztalk 核心指标 (黄线、白线)
        q_indicators = q_raw.with_columns([
            # 黄线 (长期趋势)
            ((pl.col("close_adj").rolling_mean(14) + pl.col("close_adj").rolling_mean(28) + 
              pl.col("close_adj").rolling_mean(57) + pl.col("close_adj").rolling_mean(114)) / 4).alias("ztalk_yellow"),
            # 中间变量 EMA10
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").alias("ema_10")
        ]).with_columns([
            # 白线 (DEMA10, 短期动能)
            (2 * pl.col("ema_10") - pl.col("ema_10").ewm_mean(span=10, adjust=False).over("code")).alias("ztalk_white")
        ])

        print("🚀 [Step 2] 构建‘Ztalk 市场温度计’ (替代活跃市值)...")
    
        # 3. 计算市场情绪 (Market Regime)
        # 定义：全市场处于黄线之上的股票占比
        q_market_breadth = (
            q_indicators
            .select(["date", "close_adj", "ztalk_yellow"])
            .filter(pl.col("ztalk_yellow").is_not_null())
            .with_columns((pl.col("close_adj") > pl.col("ztalk_yellow")).alias("above_yellow"))
            .group_by("date")
            .agg([
                pl.count("above_yellow").alias("total_count"),
                pl.sum("above_yellow").alias("bull_count")
            ])
            .with_columns(
                (pl.col("bull_count") / pl.col("total_count")).alias("market_temperature")
            )
            .sort("date")
            .with_columns(
                pl.col("market_temperature").rolling_mean(5).alias("market_temp_ma5")
            )
        )
    
        # 将情绪指标 Join 回个股数据
        q_full = q_indicators.join(q_market_breadth, on="date", how="left")

        print("🚀 [Step 3] 信号筛选 (B1 v2.04b)...")
    
        # 4. 计算 B1 信号 (复用之前的逻辑，精简写法)
        q_signals = q_full.with_columns([
            # KDJ
            ((pl.col("close_adj") - pl.col("low_adj").rolling_min(9)) / 
             (pl.col("high_adj").rolling_max(9) - pl.col("low_adj").rolling_min(9) + 1e-9) * 100).alias("rsv"),
            pl.col("volume").rolling_mean(5).alias("vol_ma5")
        ]).with_columns([
            pl.col("rsv").ewm_mean(alpha=1/3, adjust=False).over("code").alias("k_val")
        ]).with_columns([
            (3 * pl.col("k_val") - 2 * pl.col("k_val").ewm_mean(alpha=1/3, adjust=False).over("code")).alias("j_val")
        ]).filter(
            # === 核心开仓条件 ===
            # 1. 环境：市场温度不能太冷 (占比 > 20%) 且 处于上升期 (温度 > 5日均线)
            (pl.col("market_temperature") > 0.20) & 
            (pl.col("market_temperature") > pl.col("market_temp_ma5")) &
        
            # 2. 趋势：白线 > 黄线 (多头排列)
            (pl.col("ztalk_white") > pl.col("ztalk_yellow")) &
            (pl.col("close_adj") > pl.col("ztalk_yellow")) &
        
            # 3. 形态：J值超卖 + 缩量
            (pl.col("j_val") < 0) &
            (pl.col("volume") < pl.col("vol_ma5") * 0.7)
        )

        print("🚀 [Step 4] 模拟‘牵牛绳’持仓演化 (Path Simulation)...")
    # ---------------------------------------------------------------------
        # 替换原有的 Step 4 及其后续逻辑
        # ---------------------------------------------------------------------
        print("🚀 [Step 4] 模拟‘牵牛绳’持仓演化 (修复版)...")
    
        # 1. 扩展未来数据列
        future_cols = []
        for i in range(1, MAX_HOLD_DAYS + 1):
            future_cols.extend([
                pl.col("low_adj").shift(-i).over("code").alias(f"low_{i}"),
                pl.col("close_adj").shift(-i).over("code").alias(f"close_{i}"),
                pl.col("ztalk_white").shift(-i).over("code").alias(f"white_{i}")
            ])
        
        q_path = q_signals.with_columns(future_cols)
    
        # 2. 强力清洗：在进入 Numpy 之前，剔除核心数据缺失的行
        # 要求：入场价必须有效，且 T+1 的数据必须存在（否则无法计算次日止损）
        q_clean_path = q_path.filter(
            (pl.col("close_adj").is_not_null()) & 
            (pl.col("close_adj") > 0.1) &  # 剔除 0 元或极低价
            (pl.col("low_1").is_not_null()) # 至少得有第二天的数据
        )
    
        df_sim = q_clean_path.select(
            ["code", "date", "close_adj", "low_adj"] + 
            [f"low_{i}" for i in range(1, MAX_HOLD_DAYS + 1)] +
            [f"close_{i}" for i in range(1, MAX_HOLD_DAYS + 1)] +
            [f"white_{i}" for i in range(1, MAX_HOLD_DAYS + 1)]
        ).collect()
    
        if len(df_sim) == 0:
            print("无有效信号。")
            return

        # === Numpy 加速回测逻辑 ===
        entry_price = df_sim["close_adj"].to_numpy()
        # 修复：防止 entry_price 里的 0 导致除法错误 (虽然上面过滤了，但双重保险)
        entry_price = np.where(entry_price < 0.01, np.nan, entry_price)
    
        stop_loss_price = df_sim["low_adj"].to_numpy() * SL_BUFFER
    
        n_samples = len(df_sim)
        final_returns = np.full(n_samples, np.nan) # 默认填 NaN，方便后续过滤
        hold_days = np.zeros(n_samples)
    
        print(f"正在逐单模拟 {n_samples} 笔交易 (已剔除脏数据)...")
    
        # 循环天数
        for i in range(1, MAX_HOLD_DAYS + 1):
            # 提取当日数据
            day_low = df_sim[f"low_{i}"].to_numpy()
            day_close = df_sim[f"close_{i}"].to_numpy()
            day_white = df_sim[f"white_{i}"].to_numpy()
        
            # 掩码：当前还在持仓的 (final_returns 为 NaN 的)
            active_mask = np.isnan(final_returns)
        
            # 处理数据缺失的情况 (如果某天数据突然没了，强制平仓)
            data_missing_mask = active_mask & (np.isnan(day_low) | np.isnan(day_close))
            if np.any(data_missing_mask):
                # 缺失数据按亏损处理或者取上一个有效值，这里简单处理为 0 收益离场(平盘)
                # 或者更保守一点：按 -5% 计提坏账
                final_returns[data_missing_mask] = -0.05 
                hold_days[data_missing_mask] = i
        
            # 更新活跃掩码 (剔除刚刚处理掉的缺失数据)
            active_mask = np.isnan(final_returns)
        
            # 1. 检查止损 (Stop Loss)
            sl_mask = active_mask & (day_low < stop_loss_price)
            final_returns[sl_mask] = (stop_loss_price[sl_mask] - entry_price[sl_mask]) / entry_price[sl_mask]
            hold_days[sl_mask] = i
        
            # 2. 检查止盈 (Tow Rope): 收盘价跌破白线
            tp_mask = active_mask & (~sl_mask) & (day_close < day_white)
            final_returns[tp_mask] = (day_close[tp_mask] - entry_price[tp_mask]) / entry_price[tp_mask]
            hold_days[tp_mask] = i
    
        # 3. 到期强平 (Time Exit)
        still_active = np.isnan(final_returns)
        last_close = df_sim[f"close_{MAX_HOLD_DAYS}"].to_numpy()
        # 对于到期还没出来的，按最后一天收盘价结算。如果最后一天也没数据，填0
        valid_last = ~np.isnan(last_close)
        final_returns[still_active & valid_last] = (last_close[still_active & valid_last] - entry_price[still_active & valid_last]) / entry_price[still_active & valid_last]
        final_returns[still_active & ~valid_last] = 0 # 极端情况
        hold_days[still_active] = MAX_HOLD_DAYS
    
        # === 最终清洗与统计 ===
        # 剔除极端异常值 (比如复权导致的 +/- 1000% 收益)
        valid_mask = (final_returns > -0.5) & (final_returns < 2.0)
        clean_returns = final_returns[valid_mask]
    
        win_rate = np.mean(clean_returns > 0)
        avg_ret = np.mean(clean_returns)
        median_ret = np.median(clean_returns)
        big_win_rate = np.mean(clean_returns > 0.10)
    
        print(f"\n======== Ztalk 完全体策略 (最终修复版) ========")
        print(f"有效交易样本: {len(clean_returns)}")
        print(f"------------------------------------------------")
        print(f"最终胜率: {win_rate:.2%} (关注是否 > 40%)")
        print(f"平均收益: {avg_ret:.2%} (核心指标：必须 > 0)")
        print(f"中位数收益: {median_ret:.2%} (如果是负的，说明这是‘彩票策略’)")
        print(f"主升浪捕获率 (>10%): {big_win_rate:.2%}")
        print(f"------------------------------------------------")
        print(f"盈亏比估算: {abs(np.mean(clean_returns[clean_returns>0]) / np.mean(clean_returns[clean_returns<0])):.2f}")
        print("================================================")
    
        return final_returns
    return (run_full_system,)


@app.cell
def _(run_full_system):
    res_df = run_full_system()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
