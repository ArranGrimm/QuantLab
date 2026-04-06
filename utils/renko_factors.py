"""
砖型图反转选股因子计算模块

短线策略：捕捉超短周期 KDJ 变体指标的 V 型反转拐点

数学本质：
- RSV4 = (C - LLV(L,4)) / (HHV(H,4) - LLV(L,4)) * 100
- 砖型图 = MAX(D6 + S4 - 14, 0)
  其中 S4 = SMA(RSV4, 4, 1), K6 = SMA(RSV4, 6, 1), D6 = SMA(K6, 6, 1)
- 触发信号：砖型图从下降转为上升的第一天（拐点）
"""
import polars as pl

# ==============================================================================
# 默认配置
# ==============================================================================
DEFAULT_CONFIG = {
    # === 砖型图指标参数 ===
    "RENKO_HHV_PERIOD": 4,        # HHV/LLV 周期 (原版 4，短于标准 KDJ 的 9)
    "RENKO_SMA_FAST": 4,          # VAR2A 平滑周期 (快线, α=1/4)
    "RENKO_SMA_SLOW": 6,          # VAR4A/VAR5A 平滑周期 (慢线, α=1/6)
    "RENKO_OFFSET": 90,           # VAR1A 偏移量
    "RENKO_THRESHOLD": 4,         # 砖型图截断阈值

    # === 开仓质量条件 ===
    "REVERSAL_RATIO": 2 / 3,      # 红砖长度 >= 昨日绿砖长度的 2/3
    "GREEN_STREAK_MIN": 5,        # 转红前至少连续 N 根绿砖
    "RENKO_MIN": 2.0,             # 砖型图最小高度 (过滤零附近噪音)
    "SHAPE_THRESHOLD": 0.035,     # 形态收敛: K线实体占比 < 3.5% (砖大K小)
    "BIAS_WL_MAX": 3.0,           # 贴近白线: |close-WL|/WL*100 < 3% (白线附近起爆)

    # === B1 联动 ===
    "B1_LOOKBACK": 3,             # 近 N 天内需出现过 B1 信号

    # === 基础门槛 ===
    "MV_THRESHOLD": 6.5,          # 市值下限 (亿)
    "LIQUIDITY_THRESHOLD": 0.005, # 流动性下限 (亿)

    # === 量能过滤 ===
    "YANGYIN_RATIO": 1.25,        # 红绿比阈值 (短线适当放宽)
    "YANGYIN_PERIOD": 14,         # 红绿比周期
}


def calc_renko_factors_base(df: pl.LazyFrame, config: dict = None) -> pl.LazyFrame:
    """
    砖型图反转因子计算 (Base 裸信号版)

    仅计算砖型图指标和拐点信号，不做任何过滤。
    用于分析裸信号的统计特征。

    Args:
        df: 输入 LazyFrame，需包含列:
            code, date, open_adj, high_adj, low_adj, close_adj, volume, amount, market_cap_100m
        config: 策略参数配置

    Returns:
        LazyFrame，包含 renko_signal 信号列
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    p = cfg["RENKO_HHV_PERIOD"]
    offset = cfg["RENKO_OFFSET"]
    threshold = cfg["RENKO_THRESHOLD"]

    print("🛠️ [Strategy] 启动砖型图反转 (Base 裸信号版)...")

    # TDX SMA(X, N, M) = EMA with α=M/N → Polars ewm_mean(com=N/M-1)
    com_fast = cfg["RENKO_SMA_FAST"] - 1   # SMA(X,4,1) → com=3 → α=1/4
    com_slow = cfg["RENKO_SMA_SLOW"] - 1   # SMA(X,6,1) → com=5 → α=1/6

    return (
        df.lazy().sort(["code", "date"])
        .with_columns([
            pl.col("high_adj").rolling_max(p).over("code").alias("hhv"),
            pl.col("low_adj").rolling_min(p).over("code").alias("llv"),
        ])
        .with_columns(
            (pl.col("hhv") - pl.col("llv")).alias("renko_den"),
        )
        .with_columns([
            # VAR1A = (HHV-C)/(HHV-LLV)*100 - offset = (100 - RSV) - offset
            pl.when(pl.col("renko_den") == 0)
              .then(100.0 - offset)
              .otherwise((pl.col("hhv") - pl.col("close_adj")) / pl.col("renko_den") * 100 - offset)
              .alias("var1a"),
            # VAR3A = RSV (4日)
            pl.when(pl.col("renko_den") == 0)
              .then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("llv")) / pl.col("renko_den") * 100)
              .alias("var3a"),
        ])
        .with_columns([
            # VAR2A = SMA(VAR1A, 4, 1) + 100
            (pl.col("var1a").ewm_mean(com=com_fast, adjust=False).over("code") + 100).alias("var2a"),
            # VAR4A = SMA(VAR3A, 6, 1)
            pl.col("var3a").ewm_mean(com=com_slow, adjust=False).over("code").alias("var4a"),
        ])
        .with_columns(
            # VAR5A = SMA(VAR4A, 6, 1) + 100
            (pl.col("var4a").ewm_mean(com=com_slow, adjust=False).over("code") + 100).alias("var5a"),
        )
        .with_columns(
            # 砖型图 = MAX(VAR5A - VAR2A - threshold, 0)
            pl.max_horizontal(
                pl.col("var5a") - pl.col("var2a") - threshold,
                pl.lit(0.0),
            ).alias("renko"),
        )
        .with_columns([
            pl.col("renko").shift(1).over("code").alias("prev_renko"),
            pl.col("renko").shift(2).over("code").alias("prev2_renko"),
        ])
        .with_columns([
            (pl.col("renko") > pl.col("prev_renko")).alias("renko_rising"),
            (pl.col("prev_renko") > pl.col("renko")).alias("renko_falling"),
        ])
        .with_columns(
            pl.col("renko_rising").shift(1).over("code").fill_null(False).alias("prev_rising"),
        )
        .with_columns([
            # 拐点: 昨天未上升 + 今天上升
            (~pl.col("prev_rising") & pl.col("renko_rising")).alias("RENKO_TURN"),
            # 反转力度: 今日红砖 >= 昨日绿砖的 REVERSAL_RATIO
            (
                (pl.col("renko") - pl.col("prev_renko")) >=
                cfg["REVERSAL_RATIO"] * (pl.col("prev2_renko") - pl.col("prev_renko"))
            ).alias("REVERSAL_OK"),
            # 连续绿砖: 转红前至少 GREEN_STREAK_MIN 根连续绿砖
            (
                pl.col("renko_falling").cast(pl.Int32)
                  .rolling_sum(cfg["GREEN_STREAK_MIN"])
                  .shift(1)
                  .over("code")
                == cfg["GREEN_STREAK_MIN"]
            ).alias("GREEN_STREAK_OK"),
        ])
        .with_columns(
            (
                pl.col("RENKO_TURN") &
                pl.col("REVERSAL_OK") &
                pl.col("GREEN_STREAK_OK")
            ).alias("renko_signal"),
        )
    )


def calc_renko_factors(df: pl.LazyFrame, config: dict = None) -> pl.LazyFrame:
    """
    砖型图反转因子计算 (趋势过滤版)

    在 Base 基础上叠加:
    - WL > YL (双均线多头排列)
    - C > YL (股价在多空线上方，避免逆势抄底)
    - 市值/流动性门槛
    - 红绿比量能确认

    Args:
        df: 输入 LazyFrame，需包含列:
            code, date, open_adj, high_adj, low_adj, close_adj, volume, amount, market_cap_100m
        config: 策略参数配置

    Returns:
        LazyFrame，包含 renko_signal 信号列及中间因子
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    p = cfg["RENKO_HHV_PERIOD"]
    offset = cfg["RENKO_OFFSET"]
    threshold = cfg["RENKO_THRESHOLD"]
    com_fast = cfg["RENKO_SMA_FAST"] - 1
    com_slow = cfg["RENKO_SMA_SLOW"] - 1

    print("🛠️ [Strategy] 启动砖型图反转 (趋势过滤版)...")

    return (
        df.lazy().sort(["code", "date"])
        .with_columns([
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),

            # Ztalk 双均线系统
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
              .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            ((pl.col("close_adj").rolling_mean(14).over("code") +
              pl.col("close_adj").rolling_mean(28).over("code") +
              pl.col("close_adj").rolling_mean(57).over("code") +
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            # 砖型图极值
            pl.col("high_adj").rolling_max(p).over("code").alias("hhv"),
            pl.col("low_adj").rolling_min(p).over("code").alias("llv"),
        ])
        .with_columns([
            (pl.col("hhv") - pl.col("llv")).alias("renko_den"),
            # 真阳/真阴
            ((pl.col("close_adj") > pl.col("open_adj")) &
             (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
            ((pl.col("close_adj") < pl.col("open_adj")) &
             (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        ])
        .with_columns([
            pl.when(pl.col("renko_den") == 0)
              .then(100.0 - offset)
              .otherwise((pl.col("hhv") - pl.col("close_adj")) / pl.col("renko_den") * 100 - offset)
              .alias("var1a"),
            pl.when(pl.col("renko_den") == 0)
              .then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("llv")) / pl.col("renko_den") * 100)
              .alias("var3a"),
            # 红绿比
            (pl.col("volume") * pl.col("real_yang"))
              .rolling_sum(cfg["YANGYIN_PERIOD"]).over("code").alias("vol_yang"),
            (pl.col("volume") * pl.col("real_yin"))
              .rolling_sum(cfg["YANGYIN_PERIOD"]).over("code").alias("vol_yin"),
        ])
        .with_columns([
            (pl.col("var1a").ewm_mean(com=com_fast, adjust=False).over("code") + 100).alias("var2a"),
            pl.col("var3a").ewm_mean(com=com_slow, adjust=False).over("code").alias("var4a"),
            # 基础门槛
            ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
            (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),
            (pl.col("vol_yang") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin")).alias("YANGYIN_OK"),
        ])
        .with_columns(
            (pl.col("var4a").ewm_mean(com=com_slow, adjust=False).over("code") + 100).alias("var5a"),
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("var5a") - pl.col("var2a") - threshold,
                pl.lit(0.0),
            ).alias("renko"),
        )
        .with_columns([
            pl.col("renko").shift(1).over("code").alias("prev_renko"),
            pl.col("renko").shift(2).over("code").alias("prev2_renko"),
        ])
        .with_columns([
            (pl.col("renko") > pl.col("prev_renko")).alias("renko_rising"),
            (pl.col("prev_renko") > pl.col("renko")).alias("renko_falling"),
        ])
        .with_columns(
            pl.col("renko_rising").shift(1).over("code").fill_null(False).alias("prev_rising"),
        )
        .with_columns([
            (~pl.col("prev_rising") & pl.col("renko_rising")).alias("RENKO_TURN"),
            # 反转力度: 今日红砖 >= 昨日绿砖的 REVERSAL_RATIO
            (
                (pl.col("renko") - pl.col("prev_renko")) >=
                cfg["REVERSAL_RATIO"] * (pl.col("prev2_renko") - pl.col("prev_renko"))
            ).alias("REVERSAL_OK"),
            # 连续绿砖: 转红前至少 GREEN_STREAK_MIN 根连续绿砖
            (
                pl.col("renko_falling").cast(pl.Int32)
                  .rolling_sum(cfg["GREEN_STREAK_MIN"])
                  .shift(1)
                  .over("code")
                == cfg["GREEN_STREAK_MIN"]
            ).alias("GREEN_STREAK_OK"),
        ])
        .with_columns(
            # 最终信号: 拐点 + 反转力度 + 连续绿砖 + 趋势过滤 + 基础门槛
            (
                pl.col("RENKO_TURN") &
                pl.col("REVERSAL_OK") &
                pl.col("GREEN_STREAK_OK") &
                (pl.col("WL") > pl.col("YL")) &
                (pl.col("close_adj") > pl.col("YL")) &
                pl.col("LQ") &
                pl.col("MVOK") &
                pl.col("YANGYIN_OK")
            ).alias("renko_signal"),
        )
    )


def calc_renko_factors_wmacd(df: pl.LazyFrame, config: dict = None, require_b1: bool = False) -> pl.LazyFrame:
    """
    砖型图反转因子计算 (周线MACD共振版)

    在趋势过滤版基础上叠加:
    - Running Weekly MACD: 周线红柱 (rw_hist>0) + DIF水上 (rw_dif>0)
    - Running Monthly MACD: 月线红柱 (rm_hist>0)
    - 形态收敛 + 贴近白线 + 砖型图高度门槛
    - [可选] B1 联动: 近 N 天内需出现过 B1 信号

    Args:
        df: 输入 LazyFrame。若 require_b1=True，需已包含 b1_signal 列
            (先调用 calc_b1_factors_wmacd 再传入)
        config: 策略参数配置
        require_b1: 是否要求近期有 B1 信号作为前置条件
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    p = cfg["RENKO_HHV_PERIOD"]
    offset = cfg["RENKO_OFFSET"]
    threshold = cfg["RENKO_THRESHOLD"]
    com_fast = cfg["RENKO_SMA_FAST"] - 1
    com_slow = cfg["RENKO_SMA_SLOW"] - 1

    mode = "周线MACD共振 + B1联动" if require_b1 else "周线MACD共振"
    print(f"🛠️ [Strategy] 启动砖型图反转 ({mode})...")

    # ===== Phase 1: Running Weekly/Monthly MACD =====
    df_sorted = df.lazy().sort(["code", "date"])

    df_with_time = df_sorted.with_columns([
        pl.col("date").dt.truncate("1w").alias("week_start"),
        pl.col("date").dt.truncate("1mo").alias("month_start"),
    ])

    df_weekly = (
        df_with_time.sort(["code", "date"])
        .group_by(["code", "week_start"])
        .agg(pl.col("close_adj").last().alias("weekly_close"))
        .sort(["code", "week_start"])
        .with_columns([
            pl.col("weekly_close").ewm_mean(span=12, adjust=False).over("code").alias("w_ema12"),
            pl.col("weekly_close").ewm_mean(span=26, adjust=False).over("code").alias("w_ema26"),
        ])
        .with_columns((pl.col("w_ema12") - pl.col("w_ema26")).alias("w_dif"))
        .with_columns(pl.col("w_dif").ewm_mean(span=9, adjust=False).over("code").alias("w_dea"))
    )

    df_weekly_prev = df_weekly.select([
        "code", "week_start",
        pl.col("w_ema12").shift(1).over("code").alias("prev_w_ema12"),
        pl.col("w_ema26").shift(1).over("code").alias("prev_w_ema26"),
        pl.col("w_dea").shift(1).over("code").alias("prev_w_dea"),
    ])

    df_monthly = (
        df_with_time.sort(["code", "date"])
        .group_by(["code", "month_start"])
        .agg(pl.col("close_adj").last().alias("monthly_close"))
        .sort(["code", "month_start"])
        .with_columns([
            pl.col("monthly_close").ewm_mean(span=12, adjust=False).over("code").alias("m_ema12"),
            pl.col("monthly_close").ewm_mean(span=26, adjust=False).over("code").alias("m_ema26"),
        ])
        .with_columns((pl.col("m_ema12") - pl.col("m_ema26")).alias("m_dif"))
        .with_columns(pl.col("m_dif").ewm_mean(span=9, adjust=False).over("code").alias("m_dea"))
    )

    df_monthly_prev = df_monthly.select([
        "code", "month_start",
        pl.col("m_ema12").shift(1).over("code").alias("prev_m_ema12"),
        pl.col("m_ema26").shift(1).over("code").alias("prev_m_ema26"),
        pl.col("m_dea").shift(1).over("code").alias("prev_m_dea"),
    ])

    a12, a26, a9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0

    df_daily = (
        df_with_time
        .join(df_weekly_prev, on=["code", "week_start"], how="left")
        .join(df_monthly_prev, on=["code", "month_start"], how="left")
        .with_columns([
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_w_ema12")).alias("rw_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_w_ema26")).alias("rw_ema26"),
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_m_ema12")).alias("rm_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_m_ema26")).alias("rm_ema26"),
        ])
        .with_columns([
            (pl.col("rw_ema12") - pl.col("rw_ema26")).alias("rw_dif"),
            (pl.col("rm_ema12") - pl.col("rm_ema26")).alias("rm_dif"),
        ])
        .with_columns([
            (a9 * pl.col("rw_dif") + (1 - a9) * pl.col("prev_w_dea")).alias("rw_dea"),
            (a9 * pl.col("rm_dif") + (1 - a9) * pl.col("prev_m_dea")).alias("rm_dea"),
        ])
        .with_columns([
            (2 * (pl.col("rw_dif") - pl.col("rw_dea"))).alias("rw_hist"),
            (2 * (pl.col("rm_dif") - pl.col("rm_dea"))).alias("rm_hist"),
        ])
        .with_columns([
            ((pl.col("rw_hist") > 0) & (pl.col("rw_dif") > 0)).alias("WEEKLY_MACD_OK"),
            (pl.col("rm_hist") > 0).alias("MONTHLY_MACD_OK"),
        ])
    )

    # ===== Phase 2: 砖型图因子计算 =====
    df_result = (
        df_daily.sort(["code", "date"])
        .with_columns([
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),

            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
              .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            ((pl.col("close_adj").rolling_mean(14).over("code") +
              pl.col("close_adj").rolling_mean(28).over("code") +
              pl.col("close_adj").rolling_mean(57).over("code") +
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            pl.col("high_adj").rolling_max(p).over("code").alias("hhv"),
            pl.col("low_adj").rolling_min(p).over("code").alias("llv"),
        ])
        .with_columns([
            (pl.col("hhv") - pl.col("llv")).alias("renko_den"),
            ((pl.col("close_adj") > pl.col("open_adj")) &
             (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
            ((pl.col("close_adj") < pl.col("open_adj")) &
             (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        ])
        .with_columns([
            pl.when(pl.col("renko_den") == 0)
              .then(100.0 - offset)
              .otherwise((pl.col("hhv") - pl.col("close_adj")) / pl.col("renko_den") * 100 - offset)
              .alias("var1a"),
            pl.when(pl.col("renko_den") == 0)
              .then(50.0)
              .otherwise((pl.col("close_adj") - pl.col("llv")) / pl.col("renko_den") * 100)
              .alias("var3a"),
            (pl.col("volume") * pl.col("real_yang"))
              .rolling_sum(cfg["YANGYIN_PERIOD"]).over("code").alias("vol_yang"),
            (pl.col("volume") * pl.col("real_yin"))
              .rolling_sum(cfg["YANGYIN_PERIOD"]).over("code").alias("vol_yin"),
        ])
        .with_columns([
            (pl.col("var1a").ewm_mean(com=com_fast, adjust=False).over("code") + 100).alias("var2a"),
            pl.col("var3a").ewm_mean(com=com_slow, adjust=False).over("code").alias("var4a"),
            ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
            (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),
            (pl.col("vol_yang") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin")).alias("YANGYIN_OK"),
        ])
        .with_columns(
            (pl.col("var4a").ewm_mean(com=com_slow, adjust=False).over("code") + 100).alias("var5a"),
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("var5a") - pl.col("var2a") - threshold,
                pl.lit(0.0),
            ).alias("renko"),
        )
        .with_columns([
            pl.col("renko").shift(1).over("code").alias("prev_renko"),
            pl.col("renko").shift(2).over("code").alias("prev2_renko"),
        ])
        .with_columns([
            (pl.col("renko") > pl.col("prev_renko")).alias("renko_rising"),
            (pl.col("prev_renko") > pl.col("renko")).alias("renko_falling"),
        ])
        .with_columns(
            pl.col("renko_rising").shift(1).over("code").fill_null(False).alias("prev_rising"),
        )
        .with_columns([
            (~pl.col("prev_rising") & pl.col("renko_rising")).alias("RENKO_TURN"),
            (
                (pl.col("renko") - pl.col("prev_renko")) >=
                cfg["REVERSAL_RATIO"] * (pl.col("prev2_renko") - pl.col("prev_renko"))
            ).alias("REVERSAL_OK"),
            (
                pl.col("renko_falling").cast(pl.Int32)
                  .rolling_sum(cfg["GREEN_STREAK_MIN"])
                  .shift(1)
                  .over("code")
                == cfg["GREEN_STREAK_MIN"]
            ).alias("GREEN_STREAK_OK"),
            # 砖型图高度门槛: 过滤零附近噪音 ("绿的也大，红的也大")
            (pl.col("renko") >= cfg["RENKO_MIN"]).alias("RENKO_SIZE_OK"),
            # 形态收敛: K线实体小 ("砖可以很大，K线不能很长")
            (((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("open_adj")) < cfg["SHAPE_THRESHOLD"]).alias("SHAPE_OK"),
            # 贴近白线: 白线附近起爆 ("离白线的距离决定寿命")
            (((pl.col("close_adj") - pl.col("WL")).abs() / pl.col("WL") * 100) < cfg["BIAS_WL_MAX"]).alias("NEAR_WL"),
        ])
    )

    # ===== Phase 3: B1 联动 (可选) =====
    if require_b1:
        df_result = df_result.with_columns(
            # 近 B1_LOOKBACK 天内 (含今天) 是否出现过 B1 信号
            (
                pl.col("b1_signal").cast(pl.Int32)
                  .rolling_max(cfg["B1_LOOKBACK"])
                  .over("code")
                == 1
            ).fill_null(False).alias("B1_NEARBY"),
        )

    # ===== Phase 4: 最终信号 =====
    signal_expr = (
        pl.col("RENKO_TURN") &
        pl.col("REVERSAL_OK") &
        pl.col("GREEN_STREAK_OK") &
        pl.col("RENKO_SIZE_OK") &
        pl.col("SHAPE_OK") &
        pl.col("NEAR_WL") &
        (pl.col("WL") > pl.col("YL")) &
        (pl.col("close_adj") > pl.col("YL")) &
        pl.col("LQ") &
        pl.col("MVOK") &
        pl.col("YANGYIN_OK") &
        pl.col("WEEKLY_MACD_OK") &
        pl.col("MONTHLY_MACD_OK")
    )

    if require_b1:
        signal_expr = signal_expr & pl.col("B1_NEARBY")

    return df_result.with_columns(signal_expr.alias("renko_signal"))
