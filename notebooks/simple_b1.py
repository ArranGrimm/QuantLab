import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    from loguru import logger
    from datetime import datetime

    # ==============================================================================
    # 1. 配置与数据加载 (Clean Version)
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    # ==============================================================================
    # Ztalk 体系核心：只在“活跃市值”强势期开仓
    MANUAL_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),  # 春季躁动
        ("2019-12-16", "2020-03-02"),  # 疫情反弹
        ("2020-06-19", "2020-07-15"),  # 证券带头的疯牛
        ("2020-12-24", "2021-01-25"),  # 新能源抱团主升
        ("2021-04-16", "2021-09-14"),  # 锂电光伏大主升
        ("2022-04-27", "2022-07-05"),  # 427大反弹
        ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO 狂潮
        ("2024-02-06", "2024-03-20"),  # 救市后AI反弹
        ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
        ("2025-06-24", "2025-09-04"),  # 2025年慢牛行情
    ]


    logger.info("🚀 [Step 1] 加载原始行情数据...")

    # (A) 加载复前权行情
    q_adj = (
        pl.scan_parquet(
            os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
            include_file_paths="file_path"
        )
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ])
        .select(["code", "date", "open", "high", "low", "close", "volume", "amount"])
        .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
        .filter(pl.col("volume") > 0)
    )

    # (B) 加载 Raw (不复权) 和 Capital (股本)
    q_raw = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ])
        .select(["code", "date", "close"]).rename({"close": "close_raw"})
    )

    q_cap = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("total_capital").cast(pl.Float64)
        ])
        .select(["code", "date", "total_capital"]).sort(["code", "date"])
    )

    # (C) 合并数据 (移除了所有市场指数相关代码)
    logger.info("🔗 [Step 2] 合并基础数据...")
    q_full = (
        q_adj
        .join(q_raw, on=["code", "date"])
        .sort(["code", "date"])
        .join_asof(q_cap, on="date", by="code", strategy="backward")
        .with_columns([
            (pl.col("close_raw") * pl.col("total_capital") / 1e8).alias("market_cap_100m")
        ])
        # 🔥 已移除 .join(q_market_index)
    )

    # ==============================================================================
    # 2. 因子计算逻辑 (Original Tongdaxin Logic)
    # ==============================================================================

    def calc_b1_factors(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        B1 选股公式 v2.04b (Original Clean Version)
        已移除 LEFT_STRONG 和 VOL_SHRINK 等自定义过滤器
        """
        return df.sort(["code", "date"]).with_columns([
            # 0. 基础衍生数据
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),
            pl.col("volume").shift(1).over("code").alias("prev_vol"),

        ]).with_columns([
            # 1. KDJ 基础
            (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),

            # 2. 阴阳线
            ((pl.col("close_adj") > pl.col("open_adj")) & ~(pl.col("close_adj") < pl.col("prev_close"))).alias("real_yang"),
            ((pl.col("close_adj") < pl.col("open_adj")) & ~(pl.col("close_adj") > pl.col("prev_close"))).alias("real_yin"),

            # 3. 均线
            pl.col("amount").rolling_mean(28).over("code").alias("ma_amount_28"),
            pl.col("volume").rolling_mean(40).over("code").alias("ma_vol_40"),

            # 4. 黄白线 (Trend Lines)
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            # 5. 辅助逻辑
            (pl.col("open_adj").rolling_min(28).over("code") + 
             0.925 * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
            pl.col("volume").rolling_max(28).over("code").alias("max_vol_28"),
            pl.col("prev_vol").rolling_mean(40).over("code").alias("v40p"),
            (pl.col("close_adj").rolling_min(40).over("code") + 
             0.55 * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        ]).with_columns([
            # KDJ 计算
            pl.when(pl.col("kdj_den") == 0).then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),

            # 阴阳量
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(21).over("code").alias("vol_yang_21"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(21).over("code").alias("vol_yin_21"),
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(14).over("code").alias("vol_yang_14"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(14).over("code").alias("vol_yin_14"),

            # 选股基础条件
            ((pl.col("ma_amount_28") / 1e8) >= 0.005).alias("LQ"),
            (pl.col("market_cap_100m") >= 50).alias("MVOK"),
            (pl.col("open_adj") >= pl.col("O85")).alias("TOP15O"),
            ((pl.col("close_adj") < pl.col("prev_close")) & (pl.col("close_adj") <= pl.col("open_adj")) & (pl.col("volume") >= 1.15 * pl.col("prev_vol"))).alias("FD15"),
            ((pl.col("volume") == pl.col("max_vol_28")) & pl.col("real_yin")).alias("is_max_yin"),
            ((pl.col("volume") > 1.8 * pl.col("prev_vol")) & (pl.col("close_adj") > pl.col("open_adj")) & (pl.col("volume") > pl.col("ma_vol_40"))).alias("PLRY"),
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))).alias("BD"),
            (pl.col("volume") > 1.75 * pl.col("v40p")).alias("BIGV"),
            (pl.col("close_adj") > pl.col("R55")).alias("POSOK"),

        ]).with_columns([
            # K, D, J
            pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),
            ((pl.col("TOP15O") & pl.col("FD15")).cast(pl.Int32).rolling_sum(28).over("code") == 0).alias("GOOD28"),
            (pl.col("is_max_yin").cast(pl.Int32).rolling_sum(28).over("code") == 0).alias("MAX28_OK"),
            ((pl.col("vol_yang_21") > 1.5 * pl.col("vol_yin_21")) | (pl.col("vol_yang_14") > 1.5 * pl.col("vol_yin_14"))).alias("YANGYIN_OK"),
            (pl.col("PLRY").cast(pl.Int32).rolling_sum(28).over("code") >= 3).alias("PLRY_CNT"),

        ]).with_columns([
            pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),
        ]).with_columns([
            (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
            (pl.col("PLRY_CNT") | (pl.col("BD") & pl.col("BIGV") & pl.col("POSOK"))).alias("TRIGGER"),
        ]).with_columns([
            (pl.col("J") <= 13).alias("J_OK"),
        ]).with_columns([
            # 🔥 回归原始 XG 逻辑 (无 Left Strong / Vol Shrink)
            (pl.col("TRIGGER") & 
             pl.col("J_OK") & 
             pl.col("LQ") & 
             pl.col("MVOK") & 
             pl.col("GOOD28") & 
             pl.col("MAX28_OK") & 
             pl.col("YANGYIN_OK")
            ).alias("XG"),

        ]).with_columns([
            # B1 最终信号
            (pl.col("XG") & 
             (pl.col("WL") > pl.col("YL")) & 
             (pl.col("close_adj") > pl.col("YL"))).alias("b1_signal")
        ])
    return MANUAL_LOOSE_PERIODS, calc_b1_factors, datetime, logger, pl, q_full


@app.cell
def _(calc_b1_factors, logger, q_full):
    # 3. 执行计算
    logger.info("⏳ 计算原始 B1 信号...")
    df_signals = calc_b1_factors(q_full)
    return (df_signals,)


@app.cell
def _(MANUAL_LOOSE_PERIODS, datetime, df_signals, logger, pl):
    # ==============================================================================
    # 4. 回测引擎：集成“人工择时” + “黄线低吸” (Strategy B)
    # ==============================================================================
    def run_strategy_b_with_manual_regime(df_signals: pl.LazyFrame) -> pl.DataFrame:
        logger.info("🛠️ [Step 4] 构建‘人工择时’日历 & 执行回测...")

        # 1. 构建日期过滤器 (Is Loose Day?)
        # 获取数据中的所有交易日
        all_dates = df_signals.select("date").unique().collect()["date"].to_list()
        df_dates = pl.DataFrame({"date": all_dates}).with_columns(pl.lit(0).alias("is_loose"))

        # 映射区间：将你在 MANUAL_LOOSE_PERIODS 定义的日期标记为 1
        loose_date_set = set()
        for s_str, e_str in MANUAL_LOOSE_PERIODS:
            try:
                s = datetime.strptime(s_str, "%Y-%m-%d").date()
                e = datetime.strptime(e_str, "%Y-%m-%d").date()
                # 找到在区间内的交易日
                loose_date_set.update([d for d in all_dates if s <= d <= e])
            except Exception as e:
                logger.warning(f"日期解析错误: {s_str} - {e_str}, 请检查格式")

        # 生成择时日历 DataFrame
        df_regime = df_dates.with_columns(
            pl.col("date").is_in(list(loose_date_set)).cast(pl.Int32).alias("is_loose")
        )

        logger.info(f"📅 择时覆盖统计: 共有 {len(loose_date_set)} 个手松交易日")

        # 2. 核心回测逻辑
        return (
            df_signals
            .join(df_regime.lazy(), on="date", how="left") # 关联择时状态
            .sort(["code", "date"])
            .with_columns([
                # 预计算 T+1 日数据 (用于判断能否买入)
                pl.col("open_adj").shift(-1).over("code").alias("t1_open"),
                pl.col("low_adj").shift(-1).over("code").alias("t1_low"),
                pl.col("high_adj").shift(-1).over("code").alias("t1_high"),
                pl.col("close_adj").shift(-1).over("code").alias("t1_close"),

                # 预计算 T+N 日卖出价格 (模拟波段持有)
                pl.col("close_adj").shift(-4).over("code").alias("sell_price_3d"),  # 持有3天
                pl.col("close_adj").shift(-6).over("code").alias("sell_price_5d"),  # 持有5天
                pl.col("close_adj").shift(-11).over("code").alias("sell_price_10d"), # 持有10天
            ])

            # --- 核心过滤层 ---
            .filter(pl.col("b1_signal"))        # 1. 必须触发 B1 信号
            .filter(pl.col("is_loose") == 1)    # 2. 🔥 必须在“手松”周期内 (看天)
            .filter(pl.col("t1_open") > 3.0)    # 3. 剔除低价垃圾股 (防止计算异常)

            # --- 交易执行层 (Strategy B: 黄线低吸) --- ## 太理想化了
            # 逻辑：挂单价 = YL (前复权黄线)
            # 成交判定：T+1 最低价 <= YL * 1.01 (给予 1% 的偏差容忍度，模拟能买进去)
            .with_columns([
                (pl.col("t1_low") <= pl.col("YL") * 1.01).alias("is_filled"), 
            ])
            .filter(pl.col("is_filled")) # 只保留成交的单子

            # --- 成本计算 ---
            # 细节优化：如果 T+1 直接低开在黄线之下 (Open < YL)，我们按更便宜的 Open 价买入
            # 如果 Open > YL，则按 YL 挂单成交
            .with_columns([
                pl.when(pl.col("t1_open") < pl.col("YL"))
                  .then(pl.col("t1_open"))
                  .otherwise(pl.col("YL"))
                  .alias("buy_price")
            ])

            # --- 交易执行层 (修改版: T+1 开盘粗暴买入) ---
            # 逻辑：完全放弃“黄线挂单”的优势，信号次日开盘直接成交
            # 目的：测试在最差执行情况下的策略底线

            # 1. 移除 is_filled 判定 (不再等待回踩，假设流动性充足均可买入)
            # .with_columns([(pl.col("t1_low") <= pl.col("YL") * 1.01).alias("is_filled")]) # 删除
            # .filter(pl.col("is_filled")) # 删除

            # 2. 直接设定成本
            # .with_columns([
            #     (pl.col("t1_open") * 1.01).alias("buy_price")
            # ])

            # --- 收益率计算 ---
            .with_columns([
                ((pl.col("sell_price_3d") / pl.col("buy_price") - 1) * 100).alias("ret_3d"),
                ((pl.col("sell_price_5d") / pl.col("buy_price") - 1) * 100).alias("ret_5d"),
                ((pl.col("sell_price_10d") / pl.col("buy_price") - 1) * 100).alias("ret_10d"),
            ])
            .collect()
        )

    # ==============================================================================
    # 5. 运行与报告输出
    # ==============================================================================
    df_result = run_strategy_b_with_manual_regime(df_signals)

    print(f"\n====== 🏆 Ztalk B1 终极回测报告 (人工择时) ======")
    total_trades = df_result.height
    print(f"✅ 交易总数: {total_trades}")

    if total_trades > 0:
        print("-" * 115)
        print(f"{'周期':<8} | {'胜率':<8} | {'均值(Avg)':<10} | {'中位数(Med)':<12} | {'最大(Max)':<10} | {'最小(Min)':<10} | {'盈亏比':<8}")
        print("-" * 115)

        for m in ["ret_3d", "ret_5d", "ret_10d"]:
            win_rate = (df_result.filter(pl.col(m) > 0).height / total_trades) * 100
            avg_ret = df_result.select(pl.col(m).mean()).item()
            med_ret = df_result.select(pl.col(m).median()).item()
            max_ret = df_result.select(pl.col(m).max()).item()
            min_ret = df_result.select(pl.col(m).min()).item()

            avg_win = df_result.filter(pl.col(m) > 0).select(pl.col(m).mean()).item()
            avg_loss = df_result.filter(pl.col(m) < 0).select(pl.col(m).mean()).item()

            if avg_loss is None or avg_loss == 0:
                pl_ratio = 99.99 
            else:
                pl_ratio = avg_win / abs(avg_loss)

            print(f"{m:<10} | {win_rate:>6.2f}%  | {avg_ret:>8.2f}%   | {med_ret:>10.2f}%    | {max_ret:>8.2f}%   | {min_ret:>8.2f}%   | {pl_ratio:>6.2f}")
        print("-" * 115)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
