import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
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
        ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
    ]


    print("🚀 [Step 1] 加载原始行情数据...")

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
    print("🔗 [Step 2] 合并基础数据...")
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
    # ⚙️ 策略参数配置 (Configuration) - 调整这里即可，无需动核心逻辑
    # ==============================================================================
    CONFIG = {
        # --- 基础门槛 ---
        "J_THRESHOLD": 14,          # J值门槛 (J <= 13)
        "MV_THRESHOLD": 25,         # 市值门槛 (>= 25亿) 不然选不出来野马电池
        "LIQUIDITY_THRESHOLD": 0.005, # 流动性门槛 (28日均额 >= 0.5亿)

        # --- 能量结构 (红肥绿瘦) ---
        "YANGYIN_RATIO": 1.25,       # 阳量/阴量 倍数要求
        "YANGYIN_PERIOD_1": 21,     # 考察周期1 (21天)
        "YANGYIN_PERIOD_2": 14,     # 考察周期2 (14天)

        # --- 关键K (Trigger) ---
        "KEY_K_LOOKBACK": 28,       # 关键K有效窗口 (过去28天)
    
        # 路径1: 倍量群 (Cluster)
        "CLUSTER_VOL_RATIO": 1.8,   # 倍量倍数 (1.8倍)
        "CLUSTER_COUNT": 3,         # 出现次数 (>=3次)
        "CLUSTER_PERIOD": 28,       # 考察周期 (28天)
    
        # 路径2: 单日暴力突破 (Violent K)
        "VIOLENT_VOL_RATIO": 1.75,  # 爆量倍数 (1.75倍 V40P)
        "VIOLENT_POS_PCT": 0.55,    # 价格分位 (R55)
    
        # --- 风控 (Risk Control) ---
        "BAD_K_LOOKBACK": 28,       # 坏K线回溯周期 (28天)
        "BAD_K_OPEN_PCT": 0.925,    # 高开分位 (O85)
        "BAD_K_VOL_RATIO": 1.15,    # 放量阴线倍数

        # 为了挽救澄天伟业
        # 新增一个参数：衰竭缩量阈值 (Shrink Threshold)
        # 只要缩到高点的 30% 以下，就视为衰竭
        "SHRINK_RATIO": 0.30
    }

    # ==============================================================================
    # 🧠 核心计算引擎 (Polars Implementation)
    # ==============================================================================
    def calc_b1_factors(df: pl.LazyFrame) -> pl.LazyFrame:
        print("🛠️ [Strategy] 启动 B1 v2.04b (优化版)...")
    
        return df.sort(["code", "date"]).with_columns([
            # 0. 基础衍生数据 (REF)
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),
            pl.col("volume").shift(1).over("code").alias("prev_vol"),
            pl.col("open_adj").shift(1).over("code").alias("prev_open"),
        
            # 均线系统 (用于趋势判断)
            # WL: EMA(EMA(C,10),10)
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            # YL: (MA14+MA28+MA57+MA114)/4
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

        ]).with_columns([
            # 1. KDJ 指标 (保持原高效实现)
            # DEN := HHV(H,9)-LLV(L,9)
            (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        
            # 2. 实体定义 (Real Body)
            # REAL_YANG := C>O AND NOT(C<REF(C,1)) -> 实际上通达信意思是 阳线且未跌
            # 注意：通达信 NOT(A<B) 等价于 A>=B
            ((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
            ((pl.col("close_adj") < pl.col("open_adj")) & (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        
            # 3. 基础均量 (MA)
            # AVG40 := MA(VOL, 40)
            pl.col("volume").rolling_mean(40).over("code").alias("avg40"),
            # V40P := SUM(REF(VOL,1),40)/40  <-- 关键！不含当日的前40天均量
            pl.col("volume").shift(1).rolling_mean(40).over("code").alias("v40p"),
        
        ]).with_columns([
            # --- KDJ 计算 ---
            pl.when(pl.col("kdj_den") == 0).then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),
          
            # --- 能量结构 (YANGYIN) ---
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(CONFIG["YANGYIN_PERIOD_1"]).over("code").alias("vol_yang_p1"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(CONFIG["YANGYIN_PERIOD_1"]).over("code").alias("vol_yin_p1"),
            (pl.col("volume") * pl.col("real_yang")).rolling_sum(CONFIG["YANGYIN_PERIOD_2"]).over("code").alias("vol_yang_p2"),
            (pl.col("volume") * pl.col("real_yin")).rolling_sum(CONFIG["YANGYIN_PERIOD_2"]).over("code").alias("vol_yin_p2"),
        
            # --- 风控辅助 (O85, MAXVOL) ---
            # O85 := LLV(O,28) + 0.925*(HHV-LLV)
            (pl.col("open_adj").rolling_min(28).over("code") + 
             CONFIG["BAD_K_OPEN_PCT"] * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
         
            # MAXVOL28 := HHV(VOL, 28)
            pl.col("volume").rolling_max(28).over("code").alias("max_vol_28"),
        
            # --- 关键K辅助 (R55) ---
            # R55 := LLV(C,40) + 0.55*(HHV-LLV)
            (pl.col("close_adj").rolling_min(40).over("code") + 
             CONFIG["VIOLENT_POS_PCT"] * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        ]).with_columns([
            # K, D, J
            pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),
        
            # J_OK := J <= 13
            # (J值将在下一步计算，这里先准备 K)
        
            # YANGYIN_OK
            ((pl.col("vol_yang_p1") > CONFIG["YANGYIN_RATIO"] * pl.col("vol_yin_p1")) | 
             (pl.col("vol_yang_p2") > CONFIG["YANGYIN_RATIO"] * pl.col("vol_yin_p2"))).alias("YANGYIN_OK"),
         
            # LQ & MVOK (流动性与市值)
            ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= CONFIG["LIQUIDITY_THRESHOLD"]).alias("LQ"),
            ((pl.col("market_cap_100m")) >= CONFIG["MV_THRESHOLD"]).alias("MVOK"),

            # -------------------------------------------------------------------------
            # 🔥 针对“澄天伟业”的特效药：极致缩量豁免
            # -------------------------------------------------------------------------
            # 这种情况下，红绿比指标失真，不予考核。
            (pl.col("volume") < (CONFIG["SHRINK_RATIO"] * pl.col("max_vol_28"))).alias("IS_VOL_EXHAUSTED"),
        
            # ==============================================================================
            # 🔥 [关键修改] 风控逻辑重构
            # ==============================================================================
        
            # 1. 准备 MAX28_OK 的博弈数据 (Game Theory Data)
            # 只有是阴线时才取量，否则为0
            pl.when(pl.col("real_yin")).then(pl.col("volume")).otherwise(0).alias("vol_yin_masked"),
            # 只有是阳线时才取量，否则为0
            pl.when(pl.col("real_yang")).then(pl.col("volume")).otherwise(0).alias("vol_yang_masked"),

            # 2. 计算 GOOD28 的计数 (逻辑保持不变，依然严厉，但在最终判定时给特赦)
            (
                (pl.col("open_adj") >= pl.col("O85")) & 
                (pl.col("close_adj") < pl.col("prev_close")) & 
                (pl.col("close_adj") <= pl.col("open_adj")) & 
                (pl.col("volume") >= CONFIG["BAD_K_VOL_RATIO"] * pl.col("prev_vol"))
            ).cast(pl.Int32).rolling_sum(CONFIG["BAD_K_LOOKBACK"]).over("code").alias("bad_k_count"),
        
            # {==== ①：倍量柱 (PLRY) ====}
            # PLRY := VOL>1.8*REF(VOL,1) AND C>O AND VOL>AVG40
            (
                (pl.col("volume") > CONFIG["CLUSTER_VOL_RATIO"] * pl.col("prev_vol")) &
                (pl.col("close_adj") > pl.col("open_adj")) &
                (pl.col("volume") > pl.col("avg40"))
            ).alias("PLRY"),
        
            # {==== ②：关键K (KEY_K) ====}
            # BD := C>REF(C,1) AND C>=O
            # BIGV := VOL>1.75*V40P
            # POSOK := C>R55
            # KEY_K := BD AND BIGV AND POSOK
            (
                ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))) &
                (pl.col("volume") > CONFIG["VIOLENT_VOL_RATIO"] * pl.col("v40p")) &
                (pl.col("close_adj") > pl.col("R55"))
            ).alias("KEY_K"),

        ]).with_columns([
            # Finalize Indicators
            pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),
        
            # ==============================================================================
            # ⚖️ 最终风控判定 (Final Verdict)
            # ==============================================================================

            # 🔥 修正后的 MAX28_OK：动态博弈
            # 逻辑：只要过去28天内，【最大阳量】 >= 【最大阴量】，就说明天量阴已被化解或不是威胁
            (
                pl.col("vol_yang_masked").rolling_max(28).over("code") >= 
                pl.col("vol_yin_masked").rolling_max(28).over("code")
            ).alias("MAX28_OK"),

            # 🔥 修正后的 GOOD28：智能特赦 (Smart Amnesty)
            # 逻辑：要么完全没有坏K线；如果有，只要当前收盘价站稳黄线(YL)，视为洗盘结束，给予通行
            (
                (pl.col("bad_k_count") == 0) | 
                (pl.col("close_adj") > pl.col("YL") * 0.97)
            ).alias("GOOD28"),

            # PLRY_CNT := COUNT(PLRY, 28) >= 3
            (pl.col("PLRY").cast(pl.Int32).rolling_sum(CONFIG["CLUSTER_PERIOD"]).over("code") >= CONFIG["CLUSTER_COUNT"]).alias("PLRY_CNT"),
        
            # 🔥 EXIST(KEY_K, 28)
            (pl.col("KEY_K").cast(pl.Int32).rolling_max(CONFIG["KEY_K_LOOKBACK"]).over("code") == 1).alias("KEY_K_EXIST"),
        
        ]).with_columns([
            # J := 3*K - 2*D
            (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        
            # TRIGGER := PLRY_CNT OR EXIST(KEY_K, 28)
            (pl.col("PLRY_CNT") | pl.col("KEY_K_EXIST")).alias("TRIGGER"),
        
        ]).with_columns([
            # J_OK
            (pl.col("J") <= CONFIG["J_THRESHOLD"]).alias("J_OK"),
        ]).with_columns([
            # {==== 选股 (XG) ====}
            # XG := TRIGGER AND J_OK AND LQ AND MVOK AND GOOD28 AND MAX28_OK AND YANGYIN_OK
            (
                pl.col("TRIGGER") & 
                pl.col("J_OK") & 
                pl.col("LQ") & 
                pl.col("MVOK") & 
                pl.col("GOOD28") & 
                (pl.col("MAX28_OK") | pl.col("IS_VOL_EXHAUSTED")) &
                (pl.col("YANGYIN_OK") | pl.col("IS_VOL_EXHAUSTED"))
                # pl.col("MAX28_OK") & 
                # (pl.col("YANGYIN_OK")
            ).alias("XG")
        ]).with_columns([
            # B1 := A1*(WL>YL)*(C>YL)
            (
                pl.col("XG") & 
                (pl.col("WL") > pl.col("YL") * 0.99) & 
                (pl.col("close_adj") > pl.col("YL") * 0.985)
            ).alias("b1_signal")
        ])
    return MANUAL_LOOSE_PERIODS, calc_b1_factors, datetime, pl, q_full


@app.cell
def _(calc_b1_factors, q_full):
    # 3. 执行计算
    print("⏳ 计算原始 B1 信号...")
    df_signals = calc_b1_factors(q_full)
    return (df_signals,)


@app.cell
def _(MANUAL_LOOSE_PERIODS, datetime, pl):
    # ==============================================================================
    # 4. 回测引擎：实战派 (Aggressive Entry + Hard Stop-Loss)
    # ==============================================================================
    def run_strategy_realistic(df_signals: pl.LazyFrame) -> pl.DataFrame:
        print("🛠️ [Step 4] 启动实战回测：开盘突击 + 止损风控...")

        # 1. 择时日历构建 (保持原逻辑，这部分没问题)
        all_dates = df_signals.select("date").unique().collect()["date"].to_list()
        df_dates = pl.DataFrame({"date": all_dates}).with_columns(pl.lit(0).alias("is_loose"))
    
        loose_date_set = set()
        for s_str, e_str in MANUAL_LOOSE_PERIODS:
            try:
                s = datetime.strptime(s_str, "%Y-%m-%d").date()
                e = datetime.strptime(e_str, "%Y-%m-%d").date()
                loose_date_set.update([d for d in all_dates if s <= d <= e])
            except: pass

        df_regime = df_dates.with_columns(
            pl.col("date").is_in(list(loose_date_set)).cast(pl.Int32).alias("is_loose")
        )

        # 2. 核心交易逻辑
        # 设定参数
        STOP_LOSS_PCT = 0.07  # 7% 硬止损
    
        return (
            df_signals
            .join(df_regime.lazy(), on="date", how="left")
            .sort(["code", "date"])
            .with_columns([
                # --- 进攻视角：T+1 开盘就买 ---
                pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
            
                # --- 上帝视角：预取未来 N 天的数据 ---
                # 收盘价 (用于计算死拿的收益)
                pl.col("close_adj").shift(-3).over("code").alias("close_3d"),
                pl.col("close_adj").shift(-5).over("code").alias("close_5d"),
                pl.col("close_adj").shift(-10).over("code").alias("close_10d"),
            
                # 最低价 (用于判断是否触发止损)
                # 逻辑：获取未来窗口内的最低价。如果最低价击穿止损线，则收益锁定为 -7%
                # shift(-N) 将 T+N 移到 T，rolling_min(N) 向前回溯 N 天。
                # 组合起来：shift(-N).rolling_min(N) = T+1 到 T+N 期间的最低价
                pl.col("low_adj").rolling_min(3).shift(-3).over("code").alias("low_min_3d"),
                pl.col("low_adj").rolling_min(5).shift(-5).over("code").alias("low_min_5d"),
                pl.col("low_adj").rolling_min(10).shift(-10).over("code").alias("low_min_10d"),
            ])
            # 核心过滤
            .filter(pl.col("b1_signal"))
            .filter(pl.col("is_loose") == 1)
            .filter(pl.col("buy_price") > 0) # 确保有次日数据
            .with_columns([
                # 计算止损价
                (pl.col("buy_price") * (1 - STOP_LOSS_PCT)).alias("stop_price")
            ])
            .with_columns([
                # --- 收益计算核心逻辑 (Vectorized Stop-Loss) ---
                # 公式：如果期间最低价 < 止损价，则收益 = -7%；否则 = (期末收盘 - 买入)/买入
            
                # 3日收益 (带止损)
                pl.when(pl.col("low_min_3d") <= pl.col("stop_price"))
                  .then(pl.lit(-STOP_LOSS_PCT)) # 触发止损
                  .otherwise((pl.col("close_3d") / pl.col("buy_price")) - 1)
                  .alias("ret_3d"),

                # 5日收益 (带止损)
                pl.when(pl.col("low_min_5d") <= pl.col("stop_price"))
                  .then(pl.lit(-STOP_LOSS_PCT))
                  .otherwise((pl.col("close_5d") / pl.col("buy_price")) - 1)
                  .alias("ret_5d"),
              
                # 10日收益 (带止损) - 这里是盈亏比拉开的关键
                pl.when(pl.col("low_min_10d") <= pl.col("stop_price"))
                  .then(pl.lit(-STOP_LOSS_PCT))
                  .otherwise((pl.col("close_10d") / pl.col("buy_price")) - 1)
                  .alias("ret_10d"),
              
                # 对照组：无止损死拿 10 天 (Pure Hold)
                ((pl.col("close_10d") / pl.col("buy_price")) - 1).alias("ret_10d_raw")
            ])
            .with_columns([
                # 计算缩量程度 (越小越好)
                (pl.col("volume") / pl.col("avg40")).alias("vol_ratio"),
                # 标记是否处于冷却期 (过去 10 天内是否有过信号)
                # shift(1) 代表回看昨天及以前，避免屏蔽今天的信号
                (pl.col("b1_signal").cast(pl.Int32).shift(1).rolling_max(10).over("code").fill_null(0) == 0).alias("is_cool")
            ])
            .with_columns([
                pl.col("vol_ratio").rank("ordinal", descending=False).over("date").alias("daily_rank")
            ])
            .filter(
                (pl.col("b1_signal") == True) & 
                (pl.col("is_cool") == True) 
                # (pl.col("daily_rank") <= 5)
            )
            .collect()
        )


    # # ==============================================================================
    # # 5. 改进版报告输出 (注重胜率 vs 赔率)
    # # ==============================================================================
    # df_result = run_strategy_realistic(df_signals)

    # print(f"\n====== ⚔️ Ztalk 实战回测 (T+1开盘买 + 7%止损) ======")
    # total_trades = df_result.height
    # print(f"✅ 交易信号总数: {total_trades}")

    # if total_trades > 0:
    #     print("-" * 100)
    #     print(f"{'策略模式':<12} | {'胜率':<8} | {'均值':<8} | {'盈亏比(Odds)':<10} | {'期望值(Exp)':<10}")
    #     print("-" * 100)
    
    #     # 辅助打印函数
    #     def print_metric(name, col_name):
    #         df_valid = df_result.filter(pl.col(col_name).is_not_null())
    #         cnt = df_valid.height
    #         if cnt == 0: return
        
    #         win_cnt = df_valid.filter(pl.col(col_name) > 0).height
    #         win_rate = win_cnt / cnt
        
    #         avg_ret = df_valid.select(pl.col(col_name).mean()).item()
        
    #         avg_win = df_valid.filter(pl.col(col_name) > 0).select(pl.col(col_name).mean()).item()
    #         avg_loss = df_valid.filter(pl.col(col_name) <= 0).select(pl.col(col_name).mean()).item()
        
    #         if avg_loss == 0 or avg_loss is None: 
    #             odds = 99.9 
    #         else:
    #             odds = abs(avg_win / avg_loss)
            
    #         expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
    #         print(f"{name:<12} | {win_rate*100:>6.1f}% | {avg_ret*100:>6.2f}% | {odds:>10.2f}x  | {expectancy*100:>8.2f}%")

    #     print_metric("持仓 3天", "ret_3d")
    #     print_metric("持仓 5天", "ret_5d")
    #     print_metric("持仓10天", "ret_10d")
    #     print("-" * 100)
    #     print_metric("死拿10天(对照)", "ret_10d_raw")
    #     print("-" * 100)
    return


@app.cell
def _(MANUAL_LOOSE_PERIODS, datetime, df_signals, pl):
    # ==============================================================================
    # 4. 回测引擎：实战派 (动态技术止损版 - K线最低价风控)
    # ==============================================================================
    def run_strategy_realistic_dynamic_stop(df_signals: pl.LazyFrame, return_days: list) -> pl.DataFrame:
        print("🛠️ [Step 4] 启动实战回测：开盘突击 + 动态技术止损 (K线最低价下浮2%)...")

        # 1. 择时日历构建 (保持原逻辑)
        # 注意：这里为了防止 lazy schema 问题，建议先 collect 日期列表
        all_dates = df_signals.select("date").unique().collect()["date"].to_list()
        df_dates = pl.DataFrame({"date": all_dates}).with_columns(pl.lit(0).alias("is_loose"))
    
        loose_date_set = set()
        for s_str, e_str in MANUAL_LOOSE_PERIODS:
            try:
                s = datetime.strptime(s_str, "%Y-%m-%d").date()
                e = datetime.strptime(e_str, "%Y-%m-%d").date()
                loose_date_set.update([d for d in all_dates if s <= d <= e])
            except: pass

        df_regime = df_dates.with_columns(
            pl.col("date").is_in(list(loose_date_set)).cast(pl.Int32).alias("is_loose")
        )

        expr_list = [
            # --- 进攻视角：T+1 开盘就买 ---
            pl.col("open_adj").shift(-1).over("code").alias("buy_price"),
            pl.col("low_adj").shift(-1).over("code").alias("buy_price_low"),
            # --- 辅助指标：冷却与排序 ---
            # 标记是否处于冷却期
            (pl.col("b1_signal").cast(pl.Int32).shift(1).rolling_max(10).over("code").fill_null(0) == 0).alias("is_cool"),
            # 计算缩量比 (用于排序)
            (pl.col("volume") / pl.col("avg40")).alias("vol_ratio")
        ]

        return_expr_list = []
    
        for rd in return_days:
            # --- 上帝视角：预取未来 N 天的数据 ---
            expr_list.append(
                pl.col("close_adj").shift(-rd).over("code").alias(f"close_{rd}d"),
            )
            # 最低价 (用于判断是否触发止损)
            expr_list.append(
                pl.col("close_adj").rolling_min(rd).shift(-rd).over("code").alias(f"low_min_{rd}d"),
            )
            # x日收益
            return_expr_list.append(
                pl.when(pl.col(f"low_min_{rd}d") <= pl.col("stop_price_tech"))
                  .then(pl.col("risk_pct")) # 触发止损，亏损额度即为 risk_pct
                  .otherwise((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1)
                  .alias(f"ret_{rd}d")
            )
            # 对照组：无止损死拿
            return_expr_list.append(
                ((pl.col(f"close_{rd}d") / pl.col("buy_price")) - 1).alias(f"ret_{rd}d_raw")
            )
    
        # 2. 核心交易逻辑
        return (
            df_signals
            .join(df_regime.lazy(), on="date", how="left")
            .sort(["code", "date"])
            # ==============================================================================
            # 🔥 关键修改：在过滤之前，在全量数据上计算所有价格和指标
            # ==============================================================================
            .with_columns(expr_list)
            .with_columns(
                # --- 防守视角：动态技术止损 (Dynamic Technical Stop) ---
                # 逻辑：取信号当日(T)的最低价，向下浮动 2% 作为硬防守线
                # 这比固定 7% 更贴合个股走势
                (pl.col("buy_price_low") * 0.98).alias("stop_price_tech"),   
            )
            # ==============================================================================
            # 🔥 核心过滤 (Filtering) - 必须在 shift 计算之后
            # ==============================================================================
            .filter(pl.col("b1_signal"))        # 必须是信号
            .filter(pl.col("is_loose") == 1)    # 必须是活跃市值多头
            .filter(pl.col("is_cool") == True)  # 必须是新鲜信号(非重复)
            .filter(pl.col("buy_price") > 0)    # 确保明天有开盘价
            # ==============================================================================
            # 🔥 每日排序 (Ranking)
            # ==============================================================================
            .with_columns([
                pl.col("vol_ratio").rank("ordinal", descending=False).over("date").alias("daily_rank")
            ])
            # .filter(pl.col("daily_rank") <= 10) # 每天只买前5
            # ==============================================================================
            # 🔥 收益结算 (Settlement)
            # ==============================================================================
            .with_columns([
                # 计算这一单的实际风险比例 (Stop / Buy - 1)
                # 比如：买入日最低价很低，导致止损线在买入价下方 5%，则 risk_pct = -0.05
                ((pl.col("stop_price_tech") / pl.col("buy_price")) - 1).alias("risk_pct")
            ])
            .with_columns(return_expr_list)
            .collect()
        )

    # ==============================================================================
    # 5. 执行并打印报告 (使用新函数)
    # ==============================================================================
    # 注意：传入的是 df_signals (LazyFrame)，函数内部会自动处理全量计算和过滤
    return_days = [5, 10, 15, 20, 25, 30]

    df_result_dynamic = run_strategy_realistic_dynamic_stop(df_signals, return_days)

    print(f"\n====== ⚔️ Ztalk 实战回测 (动态技术止损版) ======")
    total_trades_dynamic = df_result_dynamic.height
    print(f"✅ 交易信号总数: {total_trades_dynamic}")

    if total_trades_dynamic > 0:
        print("-" * 100)
        print(f"{'策略模式':<12} | {'胜率':<8} | {'均值':<8} | {'盈亏比(Odds)':<10} | {'期望值(Exp)':<10}")
        print("-" * 100)
    
        # 辅助打印函数 (逻辑不变)
        def print_metric_dynamic(name, col_name, df_res):
            df_valid = df_res.filter(pl.col(col_name).is_not_null())
            cnt = df_valid.height
            if cnt == 0: return
        
            win_cnt = df_valid.filter(pl.col(col_name) > 0).height
            win_rate = win_cnt / cnt
        
            avg_ret = df_valid.select(pl.col(col_name).mean()).item()
        
            avg_win = df_valid.filter(pl.col(col_name) > 0).select(pl.col(col_name).mean()).item()
            avg_loss = df_valid.filter(pl.col(col_name) <= 0).select(pl.col(col_name).mean()).item()
        
            if avg_loss == 0 or avg_loss is None: 
                odds = 99.9 
            else:
                odds = abs(avg_win / avg_loss)
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
            print(f"{name:<12} | {win_rate*100:>6.1f}% | {avg_ret*100:>6.2f}% | {odds:>10.2f}x  | {expectancy*100:>8.2f}%")

        for rd in return_days:
            print_metric_dynamic(f"持仓{rd}天", f"ret_{rd}d", df_result_dynamic)
        print("-" * 100)
        for rd in return_days:
            print_metric_dynamic(f"死拿{rd}天(对照)", f"ret_{rd}d_raw", df_result_dynamic)
        print("-" * 100)
    return (df_result_dynamic,)


@app.cell
def _(datetime, df_result_dynamic, pl):
    hn_df = df_result_dynamic.filter(
        (pl.col("code") == "688799_SH") &
        (pl.col("date") >= datetime(2025, 5, 5))
    )
    return (hn_df,)


@app.cell
def _(hn_df):
    hn_df.select([
        'date',
        'ret_5d',
        'ret_10d',
        'ret_15d',
        'ret_20d',
        'ret_25d',
        'ret_30d',
    ])
    return


@app.cell
def _(df_result_dynamic, pl):
    # ==============================================================================
    # 6. 附录：年度交易频率压力测试 (Stress Test)
    # ==============================================================================
    def analyze_yearly_intensity(df_result: pl.DataFrame, target_year: int):
        print(f"\n====== 📊 {target_year} 年度交易强度分析 ======")
    
        # 1. 提取年份并过滤
        # 注意：需确认 date 是 Date 类型还是 String 类型，这里做了兼容处理
        try:
            # 尝试作为 Date 类型处理
            df_year = df_result.filter(pl.col("date").dt.year() == target_year)
        except:
            # 如果报错，说明是 String 类型，按字符串切片处理
            df_year = df_result.filter(pl.col("date").str.slice(0, 4) == str(target_year))
        
        total_signals = df_year.height
    
        if total_signals == 0:
            print(f"⚠️ {target_year} 年没有交易信号 (可能是数据未包含或择时全空)。")
            return

        # 2. 按日期聚合，统计每天的信号数量
        df_daily_counts = (
            df_year
            .group_by("date")
            .agg(pl.len().alias("trade_count"))
            .sort("trade_count", descending=True)
        )
    
        # 3. 计算统计指标
        active_days = df_daily_counts.height
        avg_trades = df_daily_counts.select(pl.col("trade_count").mean()).item()
        median_trades = df_daily_counts.select(pl.col("trade_count").median()).item()
        max_trades = df_daily_counts.select(pl.col("trade_count").max()).item()
    
        # 4. 打印报告
        print(f"📅 交易天数: {active_days} 天 (资金活跃度)")
        print(f"🔫 总开枪数: {total_signals} 次")
        print("-" * 40)
        print(f"📉 平均每天: {avg_trades:.1f} 只")
        print(f"⚖️ 中位每天: {median_trades:.1f} 只 (最常见的情况)")
        print(f"🔥 爆发极值: {max_trades} 只 (那天你忙得过来吗？)")
        print("-" * 40)
    
        # 5. 打印最忙碌的 Top 3 日子，看看发生了什么
        print("🥵 最忙碌的 3 天:")
        for row in df_daily_counts.head(3).iter_rows(named=True):
            print(f"   {row['date']}: {row['trade_count']} 只")

    # ==============================================================================
    # 运行分析
    # ==============================================================================
    # 统计 2024 或 2025 年的数据 (取决于你的数据源到哪一天)
    # 如果你只有 26969 条数据，大概率覆盖了多年
    analyze_yearly_intensity(df_result_dynamic, 2024) 
    analyze_yearly_intensity(df_result_dynamic, 2025)
    return


@app.cell
def _(df_signals, pl):
    PERFECT_CASES_CONFIG = [
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "600184_SH", "date": "2025-07-10", "name": "光电股份(回踩)"},
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升(反包)"}
    ]
    def verify_perfect_cases(df_signals: pl.LazyFrame, cases_config: list):
        print("🔍 [Audit] 启动十大完美案例专项验证...")
    
        # 1. 构造案例查询表
        # 注意：这里我们假设 df_signals 包含了全量计算数据
        # 我们需要重新计算未来 30 天的数据，因为之前的 df_result 可能被 filter 掉了
    
        # 构造 Polars DataFrame 用于 Join
        target_df = pl.DataFrame(cases_config).with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        )
    
        # 2. 扩展计算未来 30 天的收益数据
        # 我们直接在 df_signals 上通过 code 和 date 关联，不需要全量重算
        # 但为了获取未来数据，我们需要先在 df_signals 里 shift
    
        print("⏳ 正在回溯历史行情 (T+1 -> T+30)...")
    
        # 定义评估周期
        horizons = [5, 10, 15, 20, 25, 30]
    
        # 核心验证逻辑
        audit_df = (
            df_signals
            .sort(["code", "date"])
            .with_columns([
                # 买入价：T+1 开盘价 (实战标准)
                pl.col("open_adj").shift(-1).over("code").alias("audit_buy_price"),
            
                # 择时状态 (复用之前的逻辑)
                # 这里简单起见，我们直接检查 b1_signal 字段
            ])
        )
    
        # 动态生成不同周期的收益列
        exprs = []
        for h in horizons:
            # 持有涨幅: (Close_T+N - Buy) / Buy
            exprs.append(
                ((pl.col("close_adj").shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"ret_{h}d")
            )
            # 最高涨幅: (Max_High_T+1_to_T+N - Buy) / Buy
            # rolling_max(h) 往前看，shift(-h) 移到现在
            exprs.append(
                ((pl.col("high_adj").rolling_max(h).shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"max_{h}d")
            )

        # 执行计算并关联目标
        result = (
            audit_df
            .with_columns(exprs)
            .join(target_df.lazy(), on=["code", "date"], how="inner") # 只保留这10个
            .collect()
        )
    
        # 3. 输出报表
        print("\n====== ✨ 十大完美案例验证报告 ✨ ======")
        print(f"{'名称':<10} | {'代码':<8} | {'信号?':<5} | {'买入价':<6} | {'5日最高':<8} | {'10日最高':<8} | {'20日最高':<8} | {'30日最高':<8} | {'30日持有':<8}")
        print("-" * 120)
    
        for row in result.iter_rows(named=True):
            name = row['name']
            code = row['code']
            is_signal = "✅" if row['b1_signal'] else "❌"
            buy = row['audit_buy_price']
        
            # 格式化涨幅
            def fmt(val): return f"{val*100:>6.2f}%" if val is not None else "   N/A"
        
            # 打印核心行
            print(f"{name:<10} | {code:<8} | {is_signal:<5} | {buy:<6.2f} | {fmt(row['max_5d']):<8} | {fmt(row['max_10d']):<8} | {fmt(row['max_20d']):<8} | {fmt(row['max_30d']):<8} | {fmt(row['ret_30d']):<8}")
        
            # 如果没选出来，打印原因
            if not row['b1_signal']:
                # 简单诊断一下原因
                reasons = []
                if not row['J_OK']: reasons.append(f"J值({row['J']:.1f})>13")
                if not row['MAX28_OK']: reasons.append("有天量阴")
                if not row['YANGYIN_OK']: reasons.append("红绿比不足")
                if not row['TRIGGER']: reasons.append("无关键K")
                print(f"   ⚠️ 落选原因: {', '.join(reasons)}")
            
        print("-" * 120)


    # ==============================================================================
    # 执行验证
    # ==============================================================================
    # 假设 df_signals 依然在内存中 (即 run_strategy_b_with_manual_regime 的输入)
    if 'df_signals' in locals():
        verify_perfect_cases(df_signals, PERFECT_CASES_CONFIG)
    else:
        print("⚠️ df_signals 不在内存中，请先运行 Step 3 的 calc_b1_factors")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
