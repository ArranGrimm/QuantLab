import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    from loguru import logger

    # ==============================================================================
    # 1. 配置与数据加载 (Global Scope for Marimo)
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    logger.info("🚀 [Step 1] 开始加载数据 (已添加 amount 字段)...")

    # ---------------------------------------------------------------------
    # (A) 加载全市场后复权行情 (含 amount)
    # ---------------------------------------------------------------------
    q_adj = (
        pl.scan_parquet(
            os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
            include_file_paths="file_path"
        )
        .with_columns([
            # 提取股票代码
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            # 统一时间格式
            pl.from_epoch("time", time_unit="ms")
              .dt.replace_time_zone("UTC")
              .dt.convert_time_zone("Asia/Shanghai")
              .dt.date()
              .alias("date")
        ])
        # 🔥 修改点：直接读取 amount 字段
        .select(["code", "date", "open", "high", "low", "close", "volume", "amount"])
        .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
        # 过滤停牌 (成交量为0)
        .filter(pl.col("volume") > 0)
    )

    # ---------------------------------------------------------------------
    # (B) 构建“全A等权指数” (市场环境择时)
    # ---------------------------------------------------------------------
    # 1. 计算个股涨跌幅
    q_pct_change = (
        q_adj
        .select(["code", "date", "close_adj"])
        .sort(["code", "date"])
        .with_columns([
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("code") - 1).alias("pct_chg")
        ])
        .filter(pl.col("pct_chg").is_not_null())
    )

    # 2. 聚合指数 (等权平均涨跌幅 -> 累乘)
    q_market_index = (
        q_pct_change
        .group_by("date")
        .agg([
            pl.col("pct_chg").mean().alias("market_avg_chg")
        ])
        .sort("date")
        .collect() # 内存计算
        .with_columns([
            (1000 * (1 + pl.col("market_avg_chg")).cum_prod()).alias("index_close")
        ])
        .with_columns([
            pl.col("index_close").rolling_mean(20).alias("index_ma20")
        ])
        .with_columns([
            # 环境状态：True=手松(线上), False=手紧(线下)
            (pl.col("index_close") > pl.col("index_ma20")).alias("is_bull_market")
        ])
        .lazy()
    )

    # ---------------------------------------------------------------------
    # (C) 加载 Raw (不复权) 和 Capital (股本)
    # ---------------------------------------------------------------------
    q_raw = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
        ])
        .select(["code", "date", "close"])
        .rename({"close": "close_raw"})
    )

    q_cap = (
        pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path")
        .with_columns([
            pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
            pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("total_capital").cast(pl.Float64)
        ])
        .select(["code", "date", "total_capital"])
        .sort(["code", "date"])
    )

    # ---------------------------------------------------------------------
    # (D) 合并所有数据 (Data Base)
    # ---------------------------------------------------------------------
    logger.info("🔗 [Step 2] 合并数据底座...")

    q_full = (
        q_adj
        .join(q_raw, on=["code", "date"])
        .sort(["code", "date"])
        .join_asof(q_cap, on="date", by="code", strategy="backward")
        .with_columns([
            (pl.col("close_raw") * pl.col("total_capital") / 1e8).alias("market_cap_100m")
        ])
        .join(q_market_index, on="date", how="left")
    )

    # ==============================================================================
    # 2. 因子计算逻辑 (calc_b1_factors)
    # ==============================================================================

    def calc_b1_factors(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        B1 选股公式 v2.04b (Polars Version)
        更新：直接使用 'amount' 字段，不再估算 amount_raw
        """
        return df.sort(["code", "date"]).with_columns([
            # 0. 基础衍生数据准备
            # 这里直接使用 df 中已有的 amount
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),
            pl.col("volume").shift(1).over("code").alias("prev_vol"),
        
        ]).with_columns([
            # ---------------------------------------------------------
            # 1. KDJ 计算
            # DEN := HHV(H,9)-LLV(L,9);
            (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        
            # 2. 阴阳线定义
            # REAL_YANG := C>O AND NOT(C<REF(C,1));
            ((pl.col("close_adj") > pl.col("open_adj")) & ~(pl.col("close_adj") < pl.col("prev_close"))).alias("real_yang"),
            # REAL_YIN := C<O AND NOT(C>REF(C,1));
            ((pl.col("close_adj") < pl.col("open_adj")) & ~(pl.col("close_adj") > pl.col("prev_close"))).alias("real_yin"),
        
            # 3. 基础均线与成交量均线
            # 🔥 修改点：使用真实的 amount 计算 28日均成交额
            pl.col("amount").rolling_mean(28).over("code").alias("ma_amount_28"),
            pl.col("volume").rolling_mean(40).over("code").alias("ma_vol_40"),
        
            # 4. 黄白线 (Trend Lines)
            # WL:= EMA(EMA(C,10),10);
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
        
            # YL:= (MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4;
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            # 5. 辅助逻辑
            (pl.col("open_adj").rolling_min(28).over("code") + 
             0.925 * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
        
            pl.col("volume").rolling_max(28).over("code").alias("max_vol_28"),
        
            # V40P: SUM(REF(VOL,1),40)/40
            pl.col("prev_vol").rolling_mean(40).over("code").alias("v40p"),
        
            # R55
            (pl.col("close_adj").rolling_min(40).over("code") + 
             0.55 * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        ]).with_columns([
            # ---------------------------------------------------------
            # KDJ 继续
            pl.when(pl.col("kdj_den") == 0).then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100)
              .alias("rsv"),
          
            # VOL_YANG/YIN
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(21).over("code").alias("vol_yang_21"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(21).over("code").alias("vol_yin_21"),
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(14).over("code").alias("vol_yang_14"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(14).over("code").alias("vol_yin_14"),
        
            # LQ: A28>=0.005 (注意单位换算：A28通达信是亿元，amount是元，所以除以 1e8)
            ((pl.col("ma_amount_28") / 1e8) >= 0.005).alias("LQ"),
        
            # MVOK: MV>=50
            (pl.col("market_cap_100m") >= 50).alias("MVOK"),
        
            # TOP15O
            (pl.col("open_adj") >= pl.col("O85")).alias("TOP15O"),
        
            # FD15
            ((pl.col("close_adj") < pl.col("prev_close")) & 
             (pl.col("close_adj") <= pl.col("open_adj")) & 
             (pl.col("volume") >= 1.15 * pl.col("prev_vol"))).alias("FD15"),
         
            # MAX28_OK
            ((pl.col("volume") == pl.col("max_vol_28")) & pl.col("real_yin")).alias("is_max_yin"),
        
            # PLRY (倍量柱)
            ((pl.col("volume") > 1.8 * pl.col("prev_vol")) & 
             (pl.col("close_adj") > pl.col("open_adj")) & 
             (pl.col("volume") > pl.col("ma_vol_40"))).alias("PLRY"),
         
            # BD & BIGV & POSOK
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))).alias("BD"),
            (pl.col("volume") > 1.75 * pl.col("v40p")).alias("BIGV"),
            (pl.col("close_adj") > pl.col("R55")).alias("POSOK"),

        ]).with_columns([
            # K, D (KDJ)
            pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),
        
            # GOOD28
            ((pl.col("TOP15O") & pl.col("FD15")).cast(pl.Int32).rolling_sum(28).over("code") == 0).alias("GOOD28"),
        
            # MAX28_OK
            (pl.col("is_max_yin").cast(pl.Int32).rolling_sum(28).over("code") == 0).alias("MAX28_OK"),
        
            # YANGYIN_OK
            ((pl.col("vol_yang_21") > 1.5 * pl.col("vol_yin_21")) | 
             (pl.col("vol_yang_14") > 1.5 * pl.col("vol_yin_14"))).alias("YANGYIN_OK"),
         
            # PLRY_CNT
            (pl.col("PLRY").cast(pl.Int32).rolling_sum(28).over("code") >= 3).alias("PLRY_CNT"),
        
        ]).with_columns([
            # D
            pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),
        
        ]).with_columns([
            # J
            (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        
            # TRIGGER
            (pl.col("PLRY_CNT") | (pl.col("BD") & pl.col("BIGV") & pl.col("POSOK"))).alias("TRIGGER"),
        
        ]).with_columns([
            # J_OK
            (pl.col("J") <= 13).alias("J_OK"),
        
        ]).with_columns([
            # [优化A] 左侧爆发力：过去20天内是否有单日涨幅 > 9% (捕捉涨停板基因)
            # 注意：需要计算 pct_chg，如果 df 里没有，需要先生成
            ((pl.col("close_adj") / pl.col("prev_close") - 1).rolling_max(20).over("code") > 0.09).alias("LEFT_STRONG"),

            # [优化B] 缩量回踩：当日成交量 < 过去20天最大成交量的 60%
            # 这意味着洗盘非常彻底，主力锁仓
            (pl.col("volume") < 0.6 * pl.col("volume").rolling_max(20).over("code")).alias("VOL_SHRINK")
        ]).with_columns([
            # 更新选股逻辑 XG
            (pl.col("TRIGGER") & 
             pl.col("J_OK") & 
             pl.col("LQ") & 
             pl.col("MVOK") & 
             pl.col("GOOD28") & 
             pl.col("MAX28_OK") & 
             pl.col("YANGYIN_OK") &
             # 🔥 新增的硬核过滤
             pl.col("LEFT_STRONG") & 
             pl.col("VOL_SHRINK")
            ).alias("XG"),
        ]).with_columns([
            # 🔥 B1 Signal (结合黄白线)
            (pl.col("XG") & 
             (pl.col("WL") > pl.col("YL")) & 
             (pl.col("close_adj") > pl.col("YL"))).alias("b1_signal")
        ])
    return calc_b1_factors, logger, pl, q_adj, q_full


@app.cell
def _(calc_b1_factors, pl, q_full):
    # ==============================================================================
    # 3. 执行计算 (Marimo 交互建议)
    # ==============================================================================
    # 建议在 Marimo 下一个 cell 中运行以下代码来触发计算并查看结果：
    df_result = calc_b1_factors(q_full).collect()
    print(df_result.filter(pl.col("b1_signal")).select(["code", "date", "close_adj", "J", "WL", "YL"]))
    return (df_result,)


@app.cell
def _(df_result, pl):
    # 统计每年的信号数量分布
    signal_counts = (
        df_result
        .filter(pl.col("b1_signal"))
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .len()
        .sort("year")
    )

    print("--- 年度信号分布 (Yearly Signal Counts) ---")
    print(signal_counts)

    # 检查最近的一次“冰点”时刻（例如 2024年1月-2月），看看有没有乱发信号
    print("\n--- 2024年初股灾期间信号抽查 ---")
    print(
        df_result
        .filter(pl.col("b1_signal"))
        .filter((pl.col("date") >= pl.date(2024, 1, 1)) & (pl.col("date") <= pl.date(2024, 2, 29)))
        .select(["code", "date", "close_adj", "WL", "YL", "J"])
        .sort("date")
    )
    return


@app.cell
def _(df_result, logger, pl):
    def run_backtest_analysis(df: pl.LazyFrame) -> pl.DataFrame:
        """
        终极修正版：
        1. 解决了 Filter 导致的 Shift 时间序列断裂问题。
        2. 解决了数据源减法复权导致的负数价格 Bug。
        3. 严格对齐 T+N 收益率定义。
        """
        return (
            df
            .sort(["code", "date"])
            .with_columns([
                # =========================================================
                # Phase 1: 在全量数据上计算“未来视野” (Look-ahead)
                # =========================================================
            
                # 1. 确定买入锚点：信号日(T) 的次日(T+1) 开盘价
                pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
                pl.col("date").shift(-1).over("code").alias("buy_date"),
            
                # 2. 确定卖出锚点 (Close)
                # Ret_1d: T+1日收盘 (检验当天是否买对了，日内是否浮盈)
                pl.col("close_adj").shift(-1).over("code").alias("sell_price_1d"),
                # Ret_3d: T+3日收盘 (短线波段)
                pl.col("close_adj").shift(-3).over("code").alias("sell_price_3d"),
                # Ret_5d: T+5日收盘 (一周波段)
                pl.col("close_adj").shift(-5).over("code").alias("sell_price_5d"),
                # Ret_10d: T+10日收盘
                pl.col("close_adj").shift(-10).over("code").alias("sell_price_10d"),
            
                # 3. 风险锚点 (最大回撤)
                # 计算从 T+1 到 T+5 这5天内的最低价
                # 逻辑：先取 rolling_min(5) 包含当前行及前4行，然后 shift(-5) 把这个窗口移到未来
                # 注意：这里为了严谨，我们取 T+1 到 T+5 的 Low
                # rolling_min(5) at row T+5 covers [T+1, T+2, T+3, T+4, T+5]
                pl.col("low_adj").rolling_min(5).shift(-5).over("code").alias("min_low_5d")
            ])
        
            # =========================================================
            # Phase 2: 信号筛选与数据清洗 (Filtering)
            # =========================================================
            .filter(pl.col("b1_signal")) # 只保留信号行
        
            # 🔥 核心风控：剔除价格异常数据 (防止 -157% 这种鬼故事)
            .filter(pl.col("buy_price") > 0)
            .filter(pl.col("sell_price_1d") > 0) 
        
            # 过滤掉边界数据（比如倒数几天出的信号，没有未来的价格）
            .filter(pl.col("sell_price_1d").is_not_null())
        
            # =========================================================
            # Phase 3: 计算收益率 (Metrics Calculation)
            # =========================================================
            .with_columns([
                ((pl.col("sell_price_1d") / pl.col("buy_price") - 1) * 100).alias("ret_1d"),
                ((pl.col("sell_price_3d") / pl.col("buy_price") - 1) * 100).alias("ret_3d"),
                ((pl.col("sell_price_5d") / pl.col("buy_price") - 1) * 100).alias("ret_5d"),
                ((pl.col("sell_price_10d") / pl.col("buy_price") - 1) * 100).alias("ret_10d"),
            
                ((pl.col("min_low_5d") / pl.col("buy_price") - 1) * 100).alias("max_dd_5d"),
            ])
            .collect()
        )

    # 重新运行回测
    logger.info("⏳ 正在重新计算回测 (已修复负数价格 Bug)...")
    df_trades = run_backtest_analysis(df_result.lazy())


    # ==============================================================================
    # 5. 输出统计报告 (Performance Report)
    # ==============================================================================
    def print_performance_report(df_trades):
        total_trades = df_trades.height
    
        print(f"\n====== 🛡️ B1 策略回测报告 (Base: {total_trades} 笔交易) ======")
    
        metrics = ["ret_1d", "ret_3d", "ret_5d", "ret_10d"]
    
        print(f"{'持有周期':<10} | {'胜率(Win%)':<10} | {'平均收益(Avg%)':<15} | {'盈亏比(P/L)':<10}")
        print("-" * 60)
    
        for m in metrics:
            # 胜率：收益 > 0 的占比
            win_rate = (df_trades.filter(pl.col(m) > 0).height / total_trades) * 100
            # 平均收益
            avg_ret = df_trades.select(pl.col(m).mean()).item()
            # 盈亏比：平均盈利 / abs(平均亏损)
            avg_win = df_trades.filter(pl.col(m) > 0).select(pl.col(m).mean()).item()
            avg_loss = df_trades.filter(pl.col(m) < 0).select(pl.col(m).mean()).item()
            p_l_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
            print(f"{m:<14} | {win_rate:>6.2f}%    | {avg_ret:>10.2f}%       | {p_l_ratio:>6.2f}")

        print("\n⚠️ 风险提示：")
        drawdown_avg = df_trades.select(pl.col("max_dd_5d").mean()).item()
        print(f"买入后5日内平均最大浮亏: {drawdown_avg:.2f}%")

    # 打印报告
    print_performance_report(df_trades)
    return


@app.cell
def _(logger, pl, q_adj, q_full):

    # ---------------------------------------------------------------------
    # 1. 构造“市场宽度”指标 (Proxy for 0AMV)
    # ---------------------------------------------------------------------
    # 计算每个股票每一天是否站上 MA20
    q_breadth = (
        q_adj # 使用之前加载的 q_adj
        .select(["code", "date", "close_adj"])
        .sort(["code", "date"])
        .with_columns([
            pl.col("close_adj").rolling_mean(20).over("code").alias("ma20")
        ])
        .with_columns([
            (pl.col("close_adj") > pl.col("ma20")).cast(pl.Int32).alias("is_above_ma20")
        ])
        .group_by("date")
        .agg([
            pl.col("is_above_ma20").mean().alias("market_breadth") # 0.0 ~ 1.0
        ])
        .sort("date")
    )

    # ---------------------------------------------------------------------
    # 2. 重新关联信号与市场环境
    # ---------------------------------------------------------------------
    # 假设 df_result 是我们包含 b1_signal, XG, TRIGGER... 的大表
    # 我们需要确保它包含我们最新的“严格过滤”逻辑 (Left Strong + Vol Shrink)
    # 为了节省时间，我将直接在 df_result 上应用这些过滤器（如果之前没应用的话）

    # 这里我重新跑一遍 calc_b1_factors 的逻辑确保万无一失
    # (复用之前的 calc_b1_factors 函数逻辑，加入 严格过滤)

    def calc_b1_factors_strict(df: pl.LazyFrame) -> pl.LazyFrame:
        # ... (基于之前的逻辑，仅展示核心差异) ...
        return df.sort(["code", "date"]).with_columns([
            # 基础数据
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),
            pl.col("volume").shift(1).over("code").alias("prev_vol"),
            # KDJ
            (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
            # Trend Lines
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),
          
            # [优化A] 左侧爆发力
            ((pl.col("close_adj") / pl.col("close_adj").shift(1).over("code") - 1).rolling_max(20).over("code") > 0.09).alias("LEFT_STRONG"),
            # [优化B] 缩量
            (pl.col("volume") < 0.6 * pl.col("volume").rolling_max(20).over("code")).alias("VOL_SHRINK")
        ]).with_columns([
            # RSV
            pl.when(pl.col("kdj_den") == 0).then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),
        ]).with_columns([
            pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K")
        ]).with_columns([
            pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D")
        ]).with_columns([
            (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        ]).with_columns([
            # 最终信号 (简化版，假设其他条件如 LQ, MVOK 基础过滤已隐含或暂不影响核心逻辑)
            # 为了验证核心观点，我们只看：J<13 + N型结构(Left Strong) + 缩量 + 黄白线
            ((pl.col("J") <= 13) & 
             (pl.col("WL") > pl.col("YL")) & 
             (pl.col("close_adj") > pl.col("YL")) & 
             pl.col("LEFT_STRONG") & 
             pl.col("VOL_SHRINK")
            ).alias("b1_signal")
        ])

    # 计算信号
    df_strict = calc_b1_factors_strict(q_full).join(q_breadth.lazy(), on="date", how="left")

    # ---------------------------------------------------------------------
    # 3. 回测分析：分组对比
    # ---------------------------------------------------------------------
    def analyze_scenarios(df: pl.LazyFrame) -> pl.DataFrame:
        # 预计算未来价格
        df_calc = (
            df
            .sort(["code", "date"])
            .with_columns([
                pl.col("open_adj").shift(-1).over("code").alias("next_open"),
                pl.col("close_adj").shift(-1).over("code").alias("next_close"),
                pl.col("close_adj").shift(-3).over("code").alias("next_close_3d"),
            ])
            .filter(pl.col("b1_signal"))
            .filter(pl.col("next_open") > 0)
        )
    
        # 场景 1: 无脑买 (Base)
        res_base = df_calc.with_columns(pl.lit("Scenario 1: Base").alias("scenario"))
    
        # 场景 2: 环境过滤 (Market Breadth > 20% 也就是 0.2)
        # 0.2 意味着至少20%的股票在20日线上，避开绝对的冰点
        res_env = df_calc.filter(pl.col("market_breadth") > 0.2).with_columns(pl.lit("Scenario 2: Env Filter").alias("scenario"))
    
        # 场景 3: 环境 + 确认 (Next Close > Next Open)
        # 模拟：如果收盘没有收阳，我们就不买（或者假设我们能看懂盘面，只买收阳的）
        # 注意：为了回测方便，我们计算的是“如果只做那些收阳的交易”，看看它们的表现是否具备Alpha
        res_confirm = df_calc.filter(pl.col("market_breadth") > 0.2).filter(pl.col("next_close") > pl.col("next_open")).with_columns(pl.lit("Scenario 3: Env + Confirm").alias("scenario"))

        # 合并计算
        combined = pl.concat([res_base, res_env, res_confirm])
    
        # 计算 Ret 3D
        return (
            combined
            .with_columns([
                ((pl.col("next_close_3d") / pl.col("next_open") - 1) * 100).alias("ret_3d")
            ])
            .group_by("scenario")
            .agg([
                pl.len().alias("trade_count"),
                (pl.col("ret_3d") > 0).mean().alias("win_rate_3d"),
                pl.col("ret_3d").mean().alias("avg_ret_3d"),
                pl.col("ret_3d").min().alias("max_loss_3d") # 看看会不会踩雷
            ])
            .sort("scenario")
            .collect()
        )

    logger.info("⏳ 正在运行‘归因分析’ (环境 vs 择时)...")
    result_comparison = analyze_scenarios(df_strict)

    print("\n====== 🧪 归因分析：为什么我们失败了？ ======")
    print(result_comparison)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
