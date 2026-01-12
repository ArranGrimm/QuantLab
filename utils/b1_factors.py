import polars as pl

def calc_b1_factors_tg(df: pl.LazyFrame, CONFIG: dict = {}) -> pl.LazyFrame:
    """
    【Ztalk 量化核心】B1 选股代码天宫 (完整复刻修正版)
    修正点：
    1. 解决 Polars 上下文依赖问题 (s19依赖s18等)。
    2. 重写 S30 黄金坑逻辑，精准还原 HHVBARS 回溯。
    3. 补全 S37 诱多形态的趋势线条件。
    4. 假设输入 df 已剔除 ST 股。
    """
    print("🛠️ [Strategy] 启动 B1 天宫全因子评分系统 (Final Fixed)...")

    # ==============================================================================
    # 0. 基础算子 (Toolbox)
    # ==============================================================================
    def tdx_sma(series, n, m=1):
        """通达信 SMA: Y = (X*M + Y'*(N-M))/N"""
        return series.ewm_mean(alpha=m/n, adjust=False, min_periods=0)

    def tdx_exist(condition_col, n):
        """通达信 EXIST: N周期内是否存在"""
        return condition_col.cast(pl.Int32).rolling_max(n).fill_null(0) > 0
    
    def tdx_count(condition_col, n):
        """通达信 COUNT: N周期内满足次数"""
        return condition_col.cast(pl.Int32).rolling_sum(n).fill_null(0)

    def value_at_last_cond(value_col, cond_col):
        """通达信 REF(Val, BARSLAST(Cond)): 取最近一次条件成立时的值"""
        return pl.when(cond_col).then(value_col).otherwise(None).forward_fill()

    # ==============================================================================
    # 1. 数据预处理 (Preprocessing)
    # ==============================================================================
    # 假设列名: code, date, open_adj, high_adj, low_adj, close_adj, volume
    
    base_df = df.sort(["code", "date"]).with_columns([
        pl.col("close_adj").shift(1).over("code").alias("ref_c_1"),
        pl.col("open_adj").shift(1).over("code").alias("ref_o_1"),
        pl.col("volume").shift(1).over("code").alias("ref_v_1"),
        pl.col("volume").shift(2).over("code").alias("ref_v_2"),
        pl.col("volume").shift(3).over("code").alias("ref_v_3"),
        pl.col("volume").shift(4).over("code").alias("ref_v_4"),
        
        # 涨停价计算 (非ST逻辑)
        # 300/688/301 -> 20%, 其他 -> 10%
        pl.when(pl.col("code").str.contains(r"^(300|688|301)"))
          .then(pl.col("close_adj").shift(1).over("code") * 1.20)
          .otherwise(pl.col("close_adj").shift(1).over("code") * 1.10).alias("zt_price_raw")
    ]).with_columns([
        # TR (真实波幅)
        pl.max_horizontal([
            (pl.col("high_adj") - pl.col("low_adj")),
            (pl.col("high_adj") - pl.col("ref_c_1")).abs(),
            (pl.col("low_adj") - pl.col("ref_c_1")).abs()
        ]).alias("TR"),
        
        # 涨停价取整 (通达信逻辑: 乘100四舍五入再除100)
        (pl.col("zt_price_raw") * 100 + 0.5).floor() / 100
    ]).with_columns([
        # 一字涨停判定
        ((pl.col("open_adj") == pl.col("close_adj")) & 
         (pl.col("close_adj") == pl.col("high_adj")) & 
         (pl.col("close_adj") >= pl.col("zt_price_raw") * 0.995)).alias("is_zt_flat"),
         
        # 波动率 (MA(TR, 30))
        pl.col("TR").rolling_mean(30).over("code").alias("volatility_tr"),
    ])

    # ==============================================================================
    # 2. 核心指标计算 (Indicators)
    # ==============================================================================
    indic_df = base_df.with_columns([
        # --- Ztalk 知行双线 ---
        # 短期趋势: EMA(EMA(C,10),10)
        pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
          .ewm_mean(span=10, adjust=False).over("code").alias("zx_short"),
        # 多空线: MA14+28+57+114 / 4
        ((pl.col("close_adj").rolling_mean(14).over("code") + 
          pl.col("close_adj").rolling_mean(28).over("code") + 
          pl.col("close_adj").rolling_mean(57).over("code") + 
          pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("zx_long"),

        # --- MACD ---
        (pl.col("close_adj").ewm_mean(span=12, adjust=False).over("code") - 
         pl.col("close_adj").ewm_mean(span=26, adjust=False).over("code")).alias("dif"),
         
        # --- KDJ ---
        (pl.col("high_adj").rolling_max(9).over("code").alias("hhv9")),
        (pl.col("low_adj").rolling_min(9).over("code").alias("llv9")),
        
        # --- BBI ---
        ((pl.col("close_adj").rolling_mean(3).over("code") + 
          pl.col("close_adj").rolling_mean(6).over("code") + 
          pl.col("close_adj").rolling_mean(12).over("code") + 
          pl.col("close_adj").rolling_mean(24).over("code")) / 4).alias("bbi"),
          
        # 基础涨跌幅
        ((pl.col("close_adj") - pl.col("ref_c_1")) / pl.col("ref_c_1") * 100).alias("pct_chg"),
        ((pl.col("high_adj") - pl.col("low_adj")) / pl.col("ref_c_1") * 100).alias("amplitude"),

    ]).with_columns([
        # RSV
        pl.when(pl.col("hhv9") == pl.col("llv9")).then(50)
          .otherwise((pl.col("close_adj") - pl.col("llv9")) / (pl.col("hhv9") - pl.col("llv9")) * 100).alias("rsv"),
          
        # RSI 辅助变量
        (pl.col("close_adj") - pl.col("ref_c_1")).alias("delta"),
        
        # 知行线历史波动 (用于 s26)
        pl.col("zx_long").shift(15).over("code").alias("zx_long_15"),
        pl.col("zx_long").shift(30).over("code").alias("zx_long_30"),
        pl.col("zx_long").shift(45).over("code").alias("zx_long_45"),
        pl.col("zx_long").shift(60).over("code").alias("zx_long_60"),

    ]).with_columns([
        tdx_sma(pl.col("rsv"), 3, 1).over("code").alias("K"),
        
        # RSI 计算 (N1=3, N2=14, N3=28, N4=57)
        (tdx_sma(pl.max_horizontal(pl.col("delta"), 0), 14, 1).over("code") / 
         tdx_sma(pl.col("delta").abs(), 14, 1).over("code") * 100).alias("rsi2"),
        (tdx_sma(pl.max_horizontal(pl.col("delta"), 0), 28, 1).over("code") / 
         tdx_sma(pl.col("delta").abs(), 28, 1).over("code") * 100).alias("rsi3"),
        (tdx_sma(pl.max_horizontal(pl.col("delta"), 0), 57, 1).over("code") / 
         tdx_sma(pl.col("delta").abs(), 57, 1).over("code") * 100).alias("rsi4"),
        (tdx_sma(pl.max_horizontal(pl.col("delta"), 0), 3, 1).over("code") / 
         tdx_sma(pl.col("delta").abs(), 3, 1).over("code") * 100).alias("rsi1_sim"),
         
        # 知行线波动平均
        ((pl.col("zx_long_15") + pl.col("zx_long_30") + pl.col("zx_long_45") + pl.col("zx_long_60")) / 4).alias("zx_wave_avg"),
         
    ]).with_columns([
        tdx_sma(pl.col("K"), 3, 1).over("code").alias("D"),
        (3 * pl.col("K") - 2 * tdx_sma(pl.col("K"), 3, 1).over("code")).alias("J"),
        
        # 知行线平均_今
        pl.when(pl.col("zx_wave_avg") != 0)
          .then((pl.col("zx_long") - pl.col("zx_wave_avg")) / pl.col("zx_wave_avg"))
          .otherwise(0).alias("zx_avg_now"),
        
        # 大长阳/大长阴 (Strict TR Logic)
        ((pl.col("close_adj") > pl.col("open_adj")) & 
         (pl.col("pct_chg") > pl.col("volatility_tr") * 1.5) & 
         (pl.col("pct_chg") > 2)).alias("big_yang"),
         
        ((pl.col("close_adj") < pl.col("open_adj")) & 
         (pl.col("pct_chg").abs() > pl.col("volatility_tr") * 1.1) & 
         (pl.col("pct_chg").abs() > 2)).alias("big_yin"),
         
        # 关键量能状态
        (pl.col("volume") == pl.col("volume").rolling_max(60).over("code")).alias("is_v60"),
        (pl.col("volume") == pl.col("volume").rolling_max(30).over("code")).alias("is_v30"),
        (pl.col("volume") == pl.col("volume").rolling_max(20).over("code")).alias("is_v20"),
        (pl.col("volume") == pl.col("volume").rolling_max(10).over("code")).alias("is_v10"),
    ])

    # ==============================================================================
    # 3. 形态特征提取 (Pattern Recognition)
    # ==============================================================================
    pat_df = indic_df.with_columns([
        # 传递 DIF 高点
        (pl.col("high_adj") == pl.col("high_adj").rolling_max(30).over("code")).alias("is_h30"),
        
        # 传递 量能极大值时的 C 和 O
        value_at_last_cond(pl.col("close_adj"), pl.col("is_v60")).alias("c_at_v60"),
        value_at_last_cond(pl.col("open_adj"),  pl.col("is_v60")).alias("o_at_v60"),
        value_at_last_cond(pl.col("close_adj"), pl.col("is_v30")).alias("c_at_v30"),
        value_at_last_cond(pl.col("open_adj"),  pl.col("is_v30")).alias("o_at_v30"),
        value_at_last_cond(pl.col("close_adj"), pl.col("is_v20")).alias("c_at_v20"),
        value_at_last_cond(pl.col("open_adj"),  pl.col("is_v20")).alias("o_at_v20"),
        value_at_last_cond(pl.col("close_adj"), pl.col("is_v10")).alias("c_at_v10"),
        value_at_last_cond(pl.col("open_adj"),  pl.col("is_v10")).alias("o_at_v10"),

    ]).with_columns([
        value_at_last_cond(pl.col("dif"), pl.col("is_h30")).alias("high_point_dif"),
        value_at_last_cond(pl.col("dif").rolling_max(15).over("code").shift(1), pl.col("is_h30")).alias("prev_15_high_dif"),
        value_at_last_cond(pl.col("dif").rolling_max(20).over("code").shift(1), pl.col("is_h30")).alias("prev_20_high_dif"),
        
        # 智能参考量
        pl.when(pl.col("ref_v_1") <= pl.col("volume") / 8).then(pl.col("volume").shift(2).over("code"))
          .otherwise(pl.col("ref_v_1")).alias("ref_vol_smart"),
    ]).with_columns([
        # 关键K
        ((pl.col("close_adj") > pl.col("ref_c_1")) & 
         (pl.col("volume") > pl.col("ref_vol_smart") * 1.8) & 
         pl.col("big_yang") & 
         (pl.col("volume") > pl.col("volume").rolling_mean(40).over("code"))).alias("key_k"),
         
        # 次高点基础
        (
            (pl.col("high_adj").rolling_max(4).over("code") == pl.col("high_adj").rolling_max(60).over("code")) & 
            (pl.col("high_adj") != pl.col("high_adj").rolling_max(60).over("code")) &
            (pl.col("volume") > pl.col("volume").rolling_mean(5).over("code")) &
            (pl.col("close_adj") < pl.col("open_adj")) &
            (((pl.col("close_adj") / pl.col("close_adj").shift(10).over("code") - 1) > 0.1) | 
             ((pl.col("close_adj") / pl.col("close_adj").shift(50).over("code") - 1) > 0.5))
        ).alias("sub_high_base")
    ])

    # ==============================================================================
    # 4. 全因子计算 - 第一阶段：独立因子
    # ==============================================================================
    step1_df = pat_df.with_columns([
        # 准备地量数据 (一字板放巨量排除)
        pl.when(pl.col("is_zt_flat")).then(1e8).otherwise(pl.col("volume")).alias("vol_no_limit"),
        # 净买入量计算
        pl.when(pl.col("close_adj") > pl.col("open_adj")).then(pl.col("volume")).otherwise(-pl.col("volume")).alias("signed_vol"),
        pl.when(pl.col("close_adj") > pl.col("open_adj")).then(pl.col("volume")).otherwise(0).alias("pos_vol"),
        pl.when(pl.col("close_adj") < pl.col("open_adj")).then(pl.col("volume")).otherwise(0).alias("neg_vol"),
    ]).with_columns([
        # s1: MACD多头
        pl.when(pl.col("dif") >= 0).then(0.6).otherwise(0).alias("s1"),
        # s2: 站上MA60
        pl.when(pl.col("close_adj") > pl.col("close_adj").rolling_mean(60).over("code")).then(0.3).otherwise(0).alias("s2"),
        # s3: 涨幅适中
        pl.when(pl.col("pct_chg").is_between(-2, 1.8)).then(1.5).otherwise(-3).alias("s3"),
        # s4: 振幅 < 7
        pl.when(pl.col("amplitude") < 7).then(0.5).otherwise(-1).alias("s4"),
        # s5: 振幅 < 4
        pl.when(pl.col("amplitude") < 4).then(0.8).otherwise(0).alias("s5"),
        # s6: RSI低位
        (pl.when(pl.col("rsi1_sim") < 20).then(0.8).otherwise(0) + 
         pl.when(pl.col("rsi1_sim") < 23).then(0.7).otherwise(0)).alias("s6"),
        # s7: 趋势线夹层
        (pl.when((pl.col("zx_short") > pl.col("close_adj")) & (pl.col("close_adj") > pl.col("zx_long"))).then(1.3).otherwise(0) +
         pl.when(pl.col("close_adj") < pl.col("zx_long")).then(-3).otherwise(0) +
         pl.when(pl.col("close_adj") * 1.003 < pl.col("zx_long")).then(-3).otherwise(0)).alias("s7"),
         
        # s8-s9: 极致缩量
        (pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(30).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(26).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(24).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(22).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(20).over("code")).then(0.3).otherwise(0)).alias("s8"),
         
        (pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(18).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(16).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(14).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(12).over("code")).then(0.3).otherwise(0) +
         pl.when(pl.col("volume") == pl.col("vol_no_limit").rolling_min(10).over("code")).then(0.3).otherwise(0)).alias("s9"),

        # s10: 普通缩量
        pl.when(
            (pl.col("volume") == pl.col("volume").rolling_min(20).over("code")) | 
            (pl.col("volume") == pl.col("volume").rolling_min(19).over("code")) | 
            (pl.col("volume") == pl.col("volume").rolling_min(18).over("code"))
        ).then(0.5).otherwise(0).alias("s10"),

        # s11: 高量柱属性
        (pl.when((~pl.col("is_v60")) & (pl.col("c_at_v60") > pl.col("o_at_v60"))).then(1.0).otherwise(0) + 
         pl.when((~pl.col("is_v30")) & (pl.col("c_at_v30") > pl.col("o_at_v30"))).then(0.5).otherwise(0) +
         pl.when((~pl.col("is_v20")) & (pl.col("c_at_v20") > pl.col("o_at_v20"))).then(0.4).otherwise(0) +
         pl.when((~pl.col("is_v10")) & (pl.col("c_at_v10") > pl.col("o_at_v10"))).then(0.3).otherwise(0)).alias("s11"),
         
        # s12/s13: 高量柱阴线惩罚
        (pl.when((~pl.col("is_v60")) & (pl.col("c_at_v60") < pl.col("o_at_v60"))).then(-0.5).otherwise(0) + 
         pl.when((~pl.col("is_v30")) & (pl.col("c_at_v30") < pl.col("o_at_v30"))).then(-0.5).otherwise(0)).alias("s12"),
         
        (pl.when((~pl.col("is_v20")) & (pl.col("c_at_v20") < pl.col("o_at_v20"))).then(-0.6).otherwise(0) + 
         pl.when((~pl.col("is_v10")) & (pl.col("c_at_v10") < pl.col("o_at_v10"))).then(-0.8).otherwise(0)).alias("s13"),

        # s14: 资金流
        (pl.when(pl.col("signed_vol").rolling_sum(50).over("code") > 0).then(0.4).otherwise(0) + 
         pl.when(pl.col("signed_vol").rolling_sum(40).over("code") > 0).then(0.4).otherwise(0) + 
         pl.when(pl.col("signed_vol").rolling_sum(30).over("code") > 0).then(0.4).otherwise(0) + 
         pl.when(pl.col("signed_vol").rolling_sum(20).over("code") > 0).then(0.4).otherwise(0)).alias("s14"),

        # s15: 阳阴比 (完整7项)
        (pl.when(pl.col("pos_vol").rolling_sum(30).over("code") > 1.25 * pl.col("neg_vol").rolling_sum(30).over("code")).then(0.4).otherwise(0) +
         pl.when(pl.col("pos_vol").rolling_sum(30).over("code") > 1.50 * pl.col("neg_vol").rolling_sum(30).over("code")).then(0.5).otherwise(0) +
         pl.when(pl.col("pos_vol").rolling_sum(30).over("code") > 2.00 * pl.col("neg_vol").rolling_sum(30).over("code")).then(0.6).otherwise(0) +
         pl.when(pl.col("neg_vol").rolling_sum(50).over("code") > pl.col("pos_vol").rolling_sum(50).over("code")).then(-0.4).otherwise(0) +
         pl.when(pl.col("neg_vol").rolling_sum(40).over("code") > pl.col("pos_vol").rolling_sum(40).over("code")).then(-0.4).otherwise(0) +
         pl.when(pl.col("neg_vol").rolling_sum(30).over("code") > pl.col("pos_vol").rolling_sum(30).over("code")).then(-0.4).otherwise(0) +
         pl.when(pl.col("neg_vol").rolling_sum(20).over("code") > pl.col("pos_vol").rolling_sum(20).over("code")).then(-0.4).otherwise(0)).alias("s15"),

        # s16: BBI 上升
        pl.when(pl.col("bbi") > pl.col("bbi").shift(20).over("code")).then(0.5).otherwise(0).alias("s16"),

        # s17: 乖离惩罚 (整体乘以-1)
        (
            (((pl.col("low_adj") - pl.col("zx_long")).abs() * 2.5 > (pl.col("close_adj") - pl.col("high_adj").rolling_max(10).over("code")).abs()).cast(pl.Int32) + 
             ((pl.col("low_adj") - pl.col("zx_long")).abs() * 3.0 > (pl.col("close_adj") - pl.col("high_adj").rolling_max(10).over("code")).abs()).cast(pl.Int32))
            * (-1)).alias("s17"),
         
        # s22: 顶背离
        (pl.when(pl.col("high_point_dif") < pl.col("prev_20_high_dif")).then(-0.5).otherwise(0) + 
         pl.when(pl.col("high_point_dif") < pl.col("prev_15_high_dif")).then(-0.5).otherwise(0)).alias("s22"),
         
        # s23: 堆量检查 (完整4档 + 阳线条件)
        (pl.when(tdx_exist((pl.col("volume") > pl.col("volume").rolling_mean(30).over("code") * 4.0) & (pl.col("close_adj") > pl.col("open_adj")), 20)).then(0.5).otherwise(0) +
         pl.when(tdx_exist((pl.col("volume") > pl.col("volume").rolling_mean(30).over("code") * 4.5) & (pl.col("close_adj") > pl.col("open_adj")), 20)).then(0.4).otherwise(0) +
         pl.when(tdx_exist((pl.col("volume") > pl.col("volume").rolling_mean(30).over("code") * 5.0) & (pl.col("close_adj") > pl.col("open_adj")), 20)).then(0.3).otherwise(0) +
         pl.when(tdx_exist((pl.col("volume") > pl.col("volume").rolling_mean(30).over("code") * 5.5) & (pl.col("close_adj") > pl.col("open_adj")), 20)).then(0.2).otherwise(0)).alias("s23"),
         
        # s24: 区间振幅过大风险 (完整5档)
        (pl.when(tdx_exist(((pl.col("high_adj").rolling_max(20).over("code") / pl.col("low_adj").rolling_min(20).over("code") - 1) * 100 > 60), 20)).then(-0.8).otherwise(0) +
         pl.when(tdx_exist(((pl.col("high_adj").rolling_max(20).over("code") / pl.col("low_adj").rolling_min(20).over("code") - 1) * 100 > 70), 20)).then(-0.7).otherwise(0) +
         pl.when(tdx_exist(((pl.col("high_adj").rolling_max(20).over("code") / pl.col("low_adj").rolling_min(20).over("code") - 1) * 100 > 80), 20)).then(-0.6).otherwise(0) +
         pl.when(tdx_exist(((pl.col("high_adj").rolling_max(20).over("code") / pl.col("low_adj").rolling_min(20).over("code") - 1) * 100 > 90), 20)).then(-0.5).otherwise(0) +
         pl.when(tdx_exist(((pl.col("high_adj").rolling_max(20).over("code") / pl.col("low_adj").rolling_min(20).over("code") - 1) * 100 > 100), 20)).then(-0.4).otherwise(0)).alias("s24"),
         
        # s27: 量能萎缩
        pl.when(pl.col("pos_vol").rolling_sum(20).over("code") > pl.col("pos_vol").rolling_sum(21).over("code").shift(35)).then(0.5).otherwise(-1).alias("s27"),
        
        # s28: 双线死叉
        pl.when((pl.col("zx_short") < pl.col("zx_short").shift(1).over("code")) & 
                (pl.col("zx_long") < pl.col("zx_long").shift(1).over("code")))
          .then(-2).otherwise(0).alias("s28"),
          
        # s29: 一字板开板
        pl.when(tdx_count((pl.col("is_zt_flat")) & (pl.col("close_adj") <= pl.col("ref_c_1")), 60) > 2).then(-1.5).otherwise(0).alias("s29"),
        
        # s31: 连续缩量阴跌
        pl.when(
            (pl.col("volume") < pl.col("ref_v_1")) & (pl.col("ref_v_1") < pl.col("ref_v_2")) &
            (pl.col("close_adj") < pl.col("open_adj")) & (pl.col("ref_c_1") < pl.col("ref_o_1"))
        ).then(-1).otherwise(0).alias("s31"),
        
        # s32: 关键K
        pl.when(tdx_count(pl.col("key_k"), 20) >= 1).then(1).otherwise(0).alias("s32"),
        
        # s33: 大阳缩量
        pl.when(tdx_count(pl.col("big_yang") & (pl.col("volume") < pl.col("ref_v_1") * 0.8), 20) >= 2).then(-1).otherwise(0).alias("s33"),
        
        # s34: 巨阴出货
        pl.when(tdx_count(pl.col("big_yin") & (pl.col("volume") > pl.col("ref_v_1") * 1.2), 10) >= 1).then(-1).otherwise(0).alias("s34"),
        
        # s35: RSI 多头
        pl.when((pl.col("rsi2") > pl.col("rsi3")) & (pl.col("rsi3") > pl.col("rsi4"))).then(1).otherwise(0).alias("s35"),
        
        # s36: 涨停过多
        pl.when(tdx_count(pl.col("is_zt_flat"), 20) >= 2).then(-1.5).otherwise(0).alias("s36"),
        
        # s37: 诱多形态 (修复：补全趋势线条件)
        pl.when((pl.col("open_adj") > pl.col("zx_short")) & 
                (pl.col("zx_short") > pl.col("zx_long")) & 
                (pl.col("close_adj") < pl.col("zx_long")) & 
                (pl.col("volume") > pl.col("ref_v_1")))
          .then(-3).otherwise(0).alias("s37"),
        
        # s38: 次高点
        pl.when(tdx_count(pl.col("sub_high_base"), 10) > 0).then(-1).otherwise(0).alias("s38"),
        
        # s39: 活跃度丧失
        pl.when((tdx_count((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("volume") > 1.4 * pl.col("volume").rolling_mean(90).over("code")), 30) == 0) &
                (tdx_count(pl.col("key_k"), 30) == 0)).then(-2).otherwise(0).alias("s39")
    ])

    # ==============================================================================
    # 5. 全因子计算 - 第二阶段：复杂回溯 (S30 黄金坑)
    # ==============================================================================
    # 逻辑: 过去15天内有一天J>95, 且那一天到现在的跌幅<3%
    s30_expr_list = []
    for i in range(1, 16):
        # 条件A: i天前是过去15天(ref 1)内J的最大值所在日
        is_max = (pl.col("J").shift(i) == pl.col("J").shift(1).rolling_max(15).over("code"))
        # 条件B: 那天的 J > 95
        j_ok = (pl.col("J").shift(i) > 95)
        # 条件C: 价格约束 ABS((C - Ref(C, N))/Ref(C, N)) <= 0.03
        yc = pl.col("close_adj").shift(i)
        price_stable = (((pl.col("close_adj") - yc) / yc).abs() <= 0.03)
        # 条件D: 区间震荡小 (HHV(H, N+1) - LLV(L, N+1)) / YC < 0.12
        range_stable = ((pl.col("high_adj").rolling_max(i+1).over("code") - pl.col("low_adj").rolling_min(i+1).over("code")) / yc < 0.12)
        
        s30_expr_list.append(is_max & j_ok & price_stable & range_stable)

    step2_df = step1_df.with_columns([
        pl.when(pl.any_horizontal(s30_expr_list)).then(2.8).otherwise(0).alias("s30")
    ])

    # ==============================================================================
    # 6. 全因子计算 - 第三阶段：依赖因子 (必须按顺序计算)
    # ==============================================================================
    step3_df = step2_df.with_columns([
        # s26 依赖 s23 (完整3档)
        (pl.when((pl.col("s23")==0) & (pl.col("zx_avg_now") < 0.05)).then(-1).otherwise(0) +
         pl.when((pl.col("s23")==0) & (pl.col("zx_avg_now") < 0.075)).then(-0.8).otherwise(0) +
         pl.when((pl.col("s23")==0) & (pl.col("zx_avg_now") < 0.10)).then(-0.5).otherwise(0)).alias("s26"),
        # s25 依赖 s23 (完整跳空阳线: C>=O AND L>REF(H,1) AND 跳空幅度判断)
        # 跳空阳线条件: (L-REF(C,1))/REF(C,1) > 0.032 OR > 波动率*0.01
        (pl.when(tdx_count(
            (pl.col("close_adj") >= pl.col("open_adj")) & 
            (pl.col("low_adj") > pl.col("high_adj").shift(1).over("code")) &
            (((pl.col("low_adj") - pl.col("ref_c_1")) / pl.col("ref_c_1") > 0.032) | 
             ((pl.col("low_adj") - pl.col("ref_c_1")) / pl.col("ref_c_1") > pl.col("volatility_tr") * 0.01)), 20) >= 1).then(-1.2).otherwise(0) + 
         pl.when((pl.col("s23") > 0) & (tdx_count(
            (pl.col("close_adj") >= pl.col("open_adj")) & 
            (pl.col("low_adj") > pl.col("high_adj").shift(1).over("code")) &
            (((pl.col("low_adj") - pl.col("ref_c_1")) / pl.col("ref_c_1") > 0.032) | 
             ((pl.col("low_adj") - pl.col("ref_c_1")) / pl.col("ref_c_1") > pl.col("volatility_tr") * 0.01)), 20) >= 1)).then(-1).otherwise(0)).alias("s25"),
         
        # s18 (贴线潜伏 - 基础) - 独立但为后续铺垫
        pl.when(
            ((pl.col("close_adj") - pl.col("zx_short")) / pl.col("zx_short")).is_between(-0.015, 0.023) & 
            (pl.col("pct_chg").is_between(-2, 1.8)) & 
            (pl.col("amplitude") < 4)
        ).then(1.5).otherwise(-0.5).alias("s18"),
    ]).with_columns([
        # s19 依赖 s18
        pl.when(
            (pl.col("s18") <= 0) & 
            ((pl.col("close_adj") - pl.col("zx_short")) / pl.col("zx_short")).is_between(-0.015, 0.03) & 
            (pl.col("pct_chg").is_between(-2, 1.8)) & 
            (pl.col("amplitude") < 4)
        ).then(2).otherwise(0).alias("s19"),
    ]).with_columns([
        # s20 依赖 s18, s19
        pl.when((pl.col("s18") <= 0) & (pl.col("s19") == 0) & 
                ((pl.col("close_adj") - pl.col("zx_long"))/pl.col("zx_long") <= 0.025) &
                (pl.col("pct_chg").is_between(-2, 1.8)) &
                (pl.col("amplitude") < 4)
        ).then(0.6).otherwise(0).alias("s20"),
    ]).with_columns([
        # s21 依赖 s18, s19, s20 (夹心层惩罚)
        pl.when((pl.col("s18")<=0) & (pl.col("s19")==0) & (pl.col("s20")==0) & 
                (pl.col("close_adj") < pl.col("zx_short")) & (pl.col("close_adj") > pl.col("zx_long")))
          .then(-1.5).otherwise(0).alias("s21"),
    ])

    # ==============================================================================
    # 7. 汇总与输出
    # ==============================================================================
    b1_factors = [f"s{i}" for i in range(1, 40)]
    
    final_df = step3_df.with_columns([
        pl.sum_horizontal(b1_factors).alias("B1_Total_Score")
    ]).with_columns([
        # 最终选股逻辑:
        # 1. KDJ_J 低位 (< 13)
        # 2. MACD 多头 (DIF >= 0)
        # 3. 趋势多头 (知行短期 > 知行多空)
        # 4. 总分 > 0
        pl.when((pl.col("J") < 13) & 
                (pl.col("dif") >= 0) & 
                (pl.col("zx_short") > pl.col("zx_long")) &
                (pl.col("B1_Total_Score") > 0))
          .then(pl.col("B1_Total_Score"))
          .otherwise(0).alias("B1_Final_Score")
    ]).with_columns(
        (pl.col("B1_Final_Score") >= CONFIG.get("X", 10)).alias("b1_signal")
    ).select([
        "code", 
        "date", 
        "B1_Total_Score", # 保留总分方便看
        "B1_Final_Score", # 保留最终选股分
        "b1_signal",      # 保留选股信号
        "open_adj",       # 保留开盘价方便验证
        "high_adj",       # 保留最高价方便验证
        "low_adj",        # 保留最低价方便验证
        "close_adj",      # 保留收盘价方便验证
        "zx_short",       # 保留趋势线方便验证
        "zx_long",
        # --- 排序因子 ---
        "volume",         # 成交量 (用于缩量排序)
        "amplitude",      # 振幅 (用于低波动排序)
        "J",              # KDJ J值 (用于超卖排序)
    ])

    return final_df