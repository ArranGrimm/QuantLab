"""
B1 选股因子计算模块 (V3.0 精简优化版)
基于 V2.04b + 10大完美案例优化

核心逻辑：
- 触发器 (倍量柱/关键K) + J低位 + 流动性 + 无坏K线 + 红绿比 + 双均线多头
- 视觉量化：形态收敛 + 极致缩量 + 均线基因
"""
import polars as pl
from .sector_factors import get_sector_status
# ==============================================================================
# 默认配置 (基于 10 大完美案例优化)
# ==============================================================================
DEFAULT_CONFIG = {
    # === [核心] 视觉量化严选 ===
    "SHAPE_THRESHOLD": 0.035,       # 形态收敛：实体幅度 < 3.5%
    "VOL_SHRINK_THRESHOLD": 0.30,   # 量能窒息：< 30% * 28天最大阳量

    # === [核心] Ztalk 双均线基因 ===
    "BIAS_WL_RANGE": (-5.5, 5.5),   # 贴线程度
    "BIAS_YL_RANGE": (-1.0, 12.0),  # 回踩深度
    "BIAS_WL_YL_RANGE": (2.0, 32.0),# 趋势强度

    # === 基础门槛 ===
    "J_THRESHOLD": 13.8,            # J值上限
    "MV_THRESHOLD": 6.5,            # 市值下限 (亿)
    "LIQUIDITY_THRESHOLD": 0.005,   # 流动性下限 (亿)

    # === 能量结构 ===
    "YANGYIN_RATIO": 1.33,          # 红绿比阈值
    "YANGYIN_PERIOD_1": 21,         # 红绿比周期1
    "YANGYIN_PERIOD_2": 14,         # 红绿比周期2

    # === 关键K & 触发器 ===
    "KEY_K_LOOKBACK": 28,           # 关键K回溯周期
    "CLUSTER_VOL_RATIO": 1.8,       # 倍量柱量比
    "CLUSTER_COUNT": 3,             # 倍量柱数量
    "CLUSTER_PERIOD": 28,           # 倍量柱统计周期
    "VIOLENT_VOL_RATIO": 1.75,      # 暴力K量比
    "VIOLENT_POS_PCT": 0.55,        # 暴力K位置

    # === 风控 (坏K线) ===
    "BAD_K_LOOKBACK": 28,           # 坏K回溯周期
    "BAD_K_OPEN_PCT": 0.925,        # 坏K开盘位置
    "BAD_K_VOL_RATIO": 1.15,        # 坏K量比
    "BAD_K_AMNESTY_RATIO": 0.66,    # 坏K赦免比例
}


def calc_b1_factors_opt(df: pl.LazyFrame, config: dict = None, sector_calc: bool = False) -> pl.LazyFrame:
    """
    B1 选股因子计算 (V3.0 精简优化版)
    
    Args:
        df: 输入 LazyFrame，需包含列: 
            code, date, open_adj, high_adj, low_adj, close_adj, volume, amount, market_cap_100m
        config: 策略参数配置，默认使用 DEFAULT_CONFIG
        sector_calc: 是否计算板块因子，默认不计算
    Returns:
        LazyFrame，包含 b1_signal 等选股信号列
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    print("🛠️ [Strategy] 启动 B1 V3.0 (精简优化版)...")

    if sector_calc:
        df_sector_status = get_sector_status(df)
        print("🔍 [Sector] 板块状态计算完成")
        df = df.join(df_sector_status, on=["date", "industry"], how="left")
        print("🔍 [Sector] 板块状态合并完成")
        # 填充空值 (没有板块的票，默认 SECTOR_OK = False)
        df = df.with_columns(pl.col("SECTOR_OK").fill_null(False))

    df_b1_signals = df.lazy().sort(["code", "date"]).with_columns([
        # 0. 基础位移
        pl.col("close_adj").shift(1).over("code").alias("prev_close"),
        pl.col("volume").shift(1).over("code").alias("prev_vol"),
        pl.col("open_adj").shift(1).over("code").alias("prev_open"),

        # 1. Ztalk 双均线系统 (WL & YL)
        pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
          .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
        ((pl.col("close_adj").rolling_mean(14).over("code") + 
          pl.col("close_adj").rolling_mean(28).over("code") + 
          pl.col("close_adj").rolling_mean(57).over("code") + 
          pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

    ]).with_columns([
        # 2. 基础指标
        (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        ((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
        ((pl.col("close_adj") < pl.col("open_adj")) & (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        pl.col("volume").rolling_mean(40).over("code").alias("avg40"),
        pl.col("volume").shift(1).rolling_mean(40).over("code").alias("v40p"),

        # Ztalk 均线乖离率
        ((pl.col("close_adj") - pl.col("WL")) / pl.col("WL") * 100).alias("Bias_C_WL"), 
        ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_C_YL"), 
        ((pl.col("WL") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_WL_YL"),

    ]).with_columns([
        # RSV
        pl.when(pl.col("kdj_den") == 0).then(50.0)
          .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),

        # 红绿量能累加
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yang_p1"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yin_p1"),
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yang_p2"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yin_p2"),

        # O85 & R55
        (pl.col("open_adj").rolling_min(28).over("code") + 
         cfg["BAD_K_OPEN_PCT"] * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
        (pl.col("close_adj").rolling_min(40).over("code") + 
         cfg["VIOLENT_POS_PCT"] * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        # Max Yang Vol
        pl.when(pl.col("real_yang")).then(pl.col("volume")).otherwise(0)
          .rolling_max(28).over("code").alias("max_yang_vol_28"),

    ]).with_columns([
        # K
        pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),

        # 红绿比
        ((pl.col("vol_yang_p1") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p1")) | 
         (pl.col("vol_yang_p2") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p2"))).alias("YANGYIN_OK"),

        # 流动性
        ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
        (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),

        # BAD_K 计算
        (
            (pl.col("open_adj") >= pl.col("O85")) & 
            (pl.col("close_adj") < pl.col("prev_close")) & 
            (pl.col("close_adj") <= pl.col("open_adj")) & 
            (pl.col("volume") >= cfg["BAD_K_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("volume") >= cfg["BAD_K_AMNESTY_RATIO"] * pl.col("max_yang_vol_28")) 
        ).cast(pl.Int32).rolling_sum(cfg["BAD_K_LOOKBACK"]).over("code").alias("bad_k_count"),

        # 触发器
        (
            (pl.col("volume") > cfg["CLUSTER_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("close_adj") > pl.col("open_adj")) &
            (pl.col("volume") > pl.col("avg40"))
        ).alias("PLRY"),

        (
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))) &
            (pl.col("volume") > cfg["VIOLENT_VOL_RATIO"] * pl.col("v40p")) &
            (pl.col("close_adj") > pl.col("R55"))
        ).alias("KEY_K"),

    ]).with_columns([
        # D
        pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),

        # MAX28_OK: 好量压制坏量
        (
            pl.when(~pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code") 
            >= 
            pl.when(pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code")
        ).alias("MAX28_OK"),

        (pl.col("bad_k_count") == 0).alias("GOOD28"),
        (pl.col("PLRY").cast(pl.Int32).rolling_sum(cfg["CLUSTER_PERIOD"]).over("code") >= cfg["CLUSTER_COUNT"]).alias("PLRY_CNT"),
        (pl.col("KEY_K").cast(pl.Int32).rolling_max(cfg["KEY_K_LOOKBACK"]).over("code") == 1).alias("KEY_K_EXIST"),

    ]).with_columns([
        (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        (pl.col("PLRY_CNT") | pl.col("KEY_K_EXIST")).alias("TRIGGER"),

        # 视觉量化判定
        (((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("open_adj")) < cfg["SHAPE_THRESHOLD"]).alias("SHAPE_OK"),
        (pl.col("volume") < (cfg["VOL_SHRINK_THRESHOLD"] * pl.col("max_yang_vol_28"))).alias("VOL_SHRINK_OK"),

        # 均线基因指纹
        (
            pl.col("Bias_C_WL").is_between(cfg["BIAS_WL_RANGE"][0], cfg["BIAS_WL_RANGE"][1]) & 
            pl.col("Bias_C_YL").is_between(cfg["BIAS_YL_RANGE"][0], cfg["BIAS_YL_RANGE"][1]) & 
            pl.col("Bias_WL_YL").is_between(cfg["BIAS_WL_YL_RANGE"][0], cfg["BIAS_WL_YL_RANGE"][1])
        ).alias("ZTALK_GENE_OK"),

    ]).with_columns([
        (pl.col("J") <= cfg["J_THRESHOLD"]).alias("J_OK")
    ])
    
    if sector_calc:
        return df_b1_signals.with_columns([
            # 最终选股信号叠加板块效应
            (
                pl.col("TRIGGER") & 
                pl.col("J_OK") & 
                pl.col("LQ") & 
                pl.col("MVOK") & 
                pl.col("GOOD28") & 
                pl.col("MAX28_OK") & 
                pl.col("YANGYIN_OK") &
                pl.col("SHAPE_OK") & 
                pl.col("VOL_SHRINK_OK") &
                pl.col("ZTALK_GENE_OK") &
                pl.col("SECTOR_OK")
            ).alias("b1_signal")
        ])
    
    return df_b1_signals.with_columns([
        # 最终选股信号
        (
            pl.col("TRIGGER") & 
            pl.col("J_OK") & 
            pl.col("LQ") & 
            pl.col("MVOK") & 
            pl.col("GOOD28") & 
            pl.col("MAX28_OK") & 
            pl.col("YANGYIN_OK") &
            pl.col("SHAPE_OK") & 
            pl.col("VOL_SHRINK_OK") &
            pl.col("ZTALK_GENE_OK")
        ).alias("b1_signal")
    ])


def calc_b1_factors_dynamic_j(df: pl.LazyFrame, config: dict = None) -> pl.LazyFrame:
    """
    B1 选股因子计算 (V3.0.1a 动态J值版)
    基于 calc_b1_factors_opt，将固定 J 阈值替换为多周期拐点均值动态阈值，去掉板块逻辑。

    动态J核心：
    - J拐点: J局部极小 & J<55 & J<D & J<K & K<D
    - MIN_J = avg(短28 / 中57 / 长114+10) 三周期拐点均值
    - J_OK := J <= MIN_J
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    print("🛠️ [Strategy] 启动 B1 V3.0.1a (动态J值版)...")

    df_b1_signals = df.lazy().sort(["code", "date"]).with_columns([
        pl.col("close_adj").shift(1).over("code").alias("prev_close"),
        pl.col("volume").shift(1).over("code").alias("prev_vol"),
        pl.col("open_adj").shift(1).over("code").alias("prev_open"),

        pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
          .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
        ((pl.col("close_adj").rolling_mean(14).over("code") + 
          pl.col("close_adj").rolling_mean(28).over("code") + 
          pl.col("close_adj").rolling_mean(57).over("code") + 
          pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

    ]).with_columns([
        (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        ((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
        ((pl.col("close_adj") < pl.col("open_adj")) & (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        pl.col("volume").rolling_mean(40).over("code").alias("avg40"),
        pl.col("volume").shift(1).rolling_mean(40).over("code").alias("v40p"),

        ((pl.col("close_adj") - pl.col("WL")) / pl.col("WL") * 100).alias("Bias_C_WL"), 
        ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_C_YL"), 
        ((pl.col("WL") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_WL_YL"),

    ]).with_columns([
        pl.when(pl.col("kdj_den") == 0).then(50.0)
          .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),

        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yang_p1"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yin_p1"),
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yang_p2"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yin_p2"),

        (pl.col("open_adj").rolling_min(28).over("code") + 
         cfg["BAD_K_OPEN_PCT"] * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
        (pl.col("close_adj").rolling_min(40).over("code") + 
         cfg["VIOLENT_POS_PCT"] * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        pl.when(pl.col("real_yang")).then(pl.col("volume")).otherwise(0)
          .rolling_max(28).over("code").alias("max_yang_vol_28"),

    ]).with_columns([
        pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),

        ((pl.col("vol_yang_p1") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p1")) | 
         (pl.col("vol_yang_p2") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p2"))).alias("YANGYIN_OK"),

        ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
        (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),

        (
            (pl.col("open_adj") >= pl.col("O85")) & 
            (pl.col("close_adj") < pl.col("prev_close")) & 
            (pl.col("close_adj") <= pl.col("open_adj")) & 
            (pl.col("volume") >= cfg["BAD_K_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("volume") >= cfg["BAD_K_AMNESTY_RATIO"] * pl.col("max_yang_vol_28")) 
        ).cast(pl.Int32).rolling_sum(cfg["BAD_K_LOOKBACK"]).over("code").alias("bad_k_count"),

        (
            (pl.col("volume") > cfg["CLUSTER_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("close_adj") > pl.col("open_adj")) &
            (pl.col("volume") > pl.col("avg40"))
        ).alias("PLRY"),

        (
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))) &
            (pl.col("volume") > cfg["VIOLENT_VOL_RATIO"] * pl.col("v40p")) &
            (pl.col("close_adj") > pl.col("R55"))
        ).alias("KEY_K"),

    ]).with_columns([
        pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),

        (
            pl.when(~pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code") 
            >= 
            pl.when(pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code")
        ).alias("MAX28_OK"),

        (pl.col("bad_k_count") == 0).alias("GOOD28"),
        (pl.col("PLRY").cast(pl.Int32).rolling_sum(cfg["CLUSTER_PERIOD"]).over("code") >= cfg["CLUSTER_COUNT"]).alias("PLRY_CNT"),
        (pl.col("KEY_K").cast(pl.Int32).rolling_max(cfg["KEY_K_LOOKBACK"]).over("code") == 1).alias("KEY_K_EXIST"),

    ]).with_columns([
        (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        (pl.col("PLRY_CNT") | pl.col("KEY_K_EXIST")).alias("TRIGGER"),

        (((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("open_adj")) < cfg["SHAPE_THRESHOLD"]).alias("SHAPE_OK"),
        (pl.col("volume") < (cfg["VOL_SHRINK_THRESHOLD"] * pl.col("max_yang_vol_28"))).alias("VOL_SHRINK_OK"),

        (
            pl.col("Bias_C_WL").is_between(cfg["BIAS_WL_RANGE"][0], cfg["BIAS_WL_RANGE"][1]) & 
            pl.col("Bias_C_YL").is_between(cfg["BIAS_YL_RANGE"][0], cfg["BIAS_YL_RANGE"][1]) & 
            pl.col("Bias_WL_YL").is_between(cfg["BIAS_WL_YL_RANGE"][0], cfg["BIAS_WL_YL_RANGE"][1])
        ).alias("ZTALK_GENE_OK"),

    ]).with_columns([
        # 动态J: 拐点检测 (J局部极小 & J<55 & J<D<K)
        (
            (pl.col("J") < pl.col("J").shift(1).over("code")) & 
            (pl.col("J") < pl.col("J").shift(-1).over("code")) & 
            (pl.col("J") < 55) & 
            (pl.col("J") < pl.col("D")) & 
            (pl.col("J") < pl.col("K")) & 
            (pl.col("K") < pl.col("D"))
        ).alias("J_TURN"),

    ]).with_columns([
        pl.when(pl.col("J_TURN")).then(pl.col("J")).otherwise(0.0).alias("J_AT_TURN"),
        pl.col("J_TURN").cast(pl.Int32).alias("J_TURN_I"),

    ]).with_columns([
        # 三周期拐点均值
        (pl.col("J_AT_TURN").rolling_sum(28).over("code") / 
         pl.col("J_TURN_I").rolling_sum(28).over("code").cast(pl.Float64).clip(lower_bound=1.0)).alias("MIN_J_S"),
        (pl.col("J_AT_TURN").rolling_sum(57).over("code") / 
         pl.col("J_TURN_I").rolling_sum(57).over("code").cast(pl.Float64).clip(lower_bound=1.0)).alias("MIN_J_M"),
        (pl.col("J_AT_TURN").rolling_sum(114).over("code") / 
         pl.col("J_TURN_I").rolling_sum(114).over("code").cast(pl.Float64).clip(lower_bound=1.0) + 10).alias("MIN_J_L"),

    ]).with_columns([
        ((pl.col("MIN_J_S") + pl.col("MIN_J_M") + pl.col("MIN_J_L")) / 3.0).alias("MIN_J"),

    ]).with_columns([
        (pl.col("J") <= pl.col("MIN_J")).alias("J_OK"),

    ])

    return df_b1_signals.with_columns([
        (
            pl.col("TRIGGER") & 
            pl.col("J_OK") & 
            pl.col("LQ") & 
            pl.col("MVOK") & 
            pl.col("GOOD28") & 
            pl.col("MAX28_OK") & 
            pl.col("YANGYIN_OK") &
            pl.col("SHAPE_OK") & 
            pl.col("VOL_SHRINK_OK") &
            pl.col("ZTALK_GENE_OK")
        ).alias("b1_signal")
    ])


def calc_b1_factors_wmacd(df: pl.LazyFrame, config: dict = None) -> pl.LazyFrame:
    """
    B1 选股因子计算 (V3.0 + 周线MACD过滤版)
    基于 calc_b1_factors_opt，增加周线 MACD 强势区间过滤，去掉板块逻辑。

    周线MACD逻辑 (Ztalk "假传万卷书、真传一句话"):
    - 利用日线收盘价作为当前未完成周的临时周收盘价 (running weekly MACD)
    - 对已完成周使用实际周收盘价，仅对当周做一步 EMA 更新，计算量极低
    - WEEKLY_MACD_OK: 周线 MACD 红柱 (HIST>0) 且 DIF 水上 (DIF>0)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    print("🛠️ [Strategy] 启动 B1 V3.0 + 周线MACD...")

    # ===== Phase 1: Running Weekly MACD =====
    df_sorted = df.lazy().sort(["code", "date"])

    df_with_week = df_sorted.with_columns(
        pl.col("date").dt.truncate("1w").alias("week_start")
    )

    df_weekly = (
        df_with_week
        .sort(["code", "date"])
        .group_by(["code", "week_start"])
        .agg(pl.col("close_adj").last().alias("weekly_close"))
        .sort(["code", "week_start"])
    )

    df_weekly = (
        df_weekly
        .with_columns([
            pl.col("weekly_close").ewm_mean(span=12, adjust=False).over("code").alias("w_ema12"),
            pl.col("weekly_close").ewm_mean(span=26, adjust=False).over("code").alias("w_ema26"),
        ])
        .with_columns(
            (pl.col("w_ema12") - pl.col("w_ema26")).alias("w_dif")
        )
        .with_columns(
            pl.col("w_dif").ewm_mean(span=9, adjust=False).over("code").alias("w_dea")
        )
    )

    df_weekly_prev = df_weekly.select([
        "code", "week_start",
        pl.col("w_ema12").shift(1).over("code").alias("prev_w_ema12"),
        pl.col("w_ema26").shift(1).over("code").alias("prev_w_ema26"),
        pl.col("w_dea").shift(1).over("code").alias("prev_w_dea"),
    ])

    a12, a26, a9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0

    df_daily = (
        df_with_week
        .join(df_weekly_prev, on=["code", "week_start"], how="left")
        .with_columns([
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_w_ema12")).alias("rw_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_w_ema26")).alias("rw_ema26"),
        ])
        .with_columns(
            (pl.col("rw_ema12") - pl.col("rw_ema26")).alias("rw_dif")
        )
        .with_columns(
            (a9 * pl.col("rw_dif") + (1 - a9) * pl.col("prev_w_dea")).alias("rw_dea")
        )
        .with_columns(
            (2 * (pl.col("rw_dif") - pl.col("rw_dea"))).alias("rw_hist")
        )
        .with_columns(
            ((pl.col("rw_hist") > 0) & (pl.col("rw_dif") > 0)).alias("WEEKLY_MACD_OK")
        )
    )

    # ===== Phase 2: B1 因子计算 (标准 V3.0 流程) =====
    df_b1_signals = df_daily.sort(["code", "date"]).with_columns([
        pl.col("close_adj").shift(1).over("code").alias("prev_close"),
        pl.col("volume").shift(1).over("code").alias("prev_vol"),
        pl.col("open_adj").shift(1).over("code").alias("prev_open"),

        pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
          .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
        ((pl.col("close_adj").rolling_mean(14).over("code") + 
          pl.col("close_adj").rolling_mean(28).over("code") + 
          pl.col("close_adj").rolling_mean(57).over("code") + 
          pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

    ]).with_columns([
        (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        ((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
        ((pl.col("close_adj") < pl.col("open_adj")) & (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        pl.col("volume").rolling_mean(40).over("code").alias("avg40"),
        pl.col("volume").shift(1).rolling_mean(40).over("code").alias("v40p"),

        ((pl.col("close_adj") - pl.col("WL")) / pl.col("WL") * 100).alias("Bias_C_WL"), 
        ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_C_YL"), 
        ((pl.col("WL") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_WL_YL"),

    ]).with_columns([
        pl.when(pl.col("kdj_den") == 0).then(50.0)
          .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),

        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yang_p1"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yin_p1"),
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yang_p2"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yin_p2"),

        (pl.col("open_adj").rolling_min(28).over("code") + 
         cfg["BAD_K_OPEN_PCT"] * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
        (pl.col("close_adj").rolling_min(40).over("code") + 
         cfg["VIOLENT_POS_PCT"] * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        pl.when(pl.col("real_yang")).then(pl.col("volume")).otherwise(0)
          .rolling_max(28).over("code").alias("max_yang_vol_28"),

    ]).with_columns([
        pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),

        ((pl.col("vol_yang_p1") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p1")) | 
         (pl.col("vol_yang_p2") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p2"))).alias("YANGYIN_OK"),

        ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
        (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),

        (
            (pl.col("open_adj") >= pl.col("O85")) & 
            (pl.col("close_adj") < pl.col("prev_close")) & 
            (pl.col("close_adj") <= pl.col("open_adj")) & 
            (pl.col("volume") >= cfg["BAD_K_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("volume") >= cfg["BAD_K_AMNESTY_RATIO"] * pl.col("max_yang_vol_28")) 
        ).cast(pl.Int32).rolling_sum(cfg["BAD_K_LOOKBACK"]).over("code").alias("bad_k_count"),

        (
            (pl.col("volume") > cfg["CLUSTER_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("close_adj") > pl.col("open_adj")) &
            (pl.col("volume") > pl.col("avg40"))
        ).alias("PLRY"),

        (
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))) &
            (pl.col("volume") > cfg["VIOLENT_VOL_RATIO"] * pl.col("v40p")) &
            (pl.col("close_adj") > pl.col("R55"))
        ).alias("KEY_K"),

    ]).with_columns([
        pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),

        (
            pl.when(~pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code") 
            >= 
            pl.when(pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code")
        ).alias("MAX28_OK"),

        (pl.col("bad_k_count") == 0).alias("GOOD28"),
        (pl.col("PLRY").cast(pl.Int32).rolling_sum(cfg["CLUSTER_PERIOD"]).over("code") >= cfg["CLUSTER_COUNT"]).alias("PLRY_CNT"),
        (pl.col("KEY_K").cast(pl.Int32).rolling_max(cfg["KEY_K_LOOKBACK"]).over("code") == 1).alias("KEY_K_EXIST"),

    ]).with_columns([
        (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        (pl.col("PLRY_CNT") | pl.col("KEY_K_EXIST")).alias("TRIGGER"),

        (((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("open_adj")) < cfg["SHAPE_THRESHOLD"]).alias("SHAPE_OK"),
        (pl.col("volume") < (cfg["VOL_SHRINK_THRESHOLD"] * pl.col("max_yang_vol_28"))).alias("VOL_SHRINK_OK"),

        (
            pl.col("Bias_C_WL").is_between(cfg["BIAS_WL_RANGE"][0], cfg["BIAS_WL_RANGE"][1]) & 
            pl.col("Bias_C_YL").is_between(cfg["BIAS_YL_RANGE"][0], cfg["BIAS_YL_RANGE"][1]) & 
            pl.col("Bias_WL_YL").is_between(cfg["BIAS_WL_YL_RANGE"][0], cfg["BIAS_WL_YL_RANGE"][1])
        ).alias("ZTALK_GENE_OK"),

    ]).with_columns([
        (pl.col("J") <= cfg["J_THRESHOLD"]).alias("J_OK")
    ])

    # ===== Phase 3: 最终信号 (B1 + 周线MACD过滤) =====
    return df_b1_signals.with_columns([
        (
            pl.col("TRIGGER") & 
            pl.col("J_OK") & 
            pl.col("LQ") & 
            pl.col("MVOK") & 
            pl.col("GOOD28") & 
            pl.col("MAX28_OK") & 
            pl.col("YANGYIN_OK") &
            pl.col("SHAPE_OK") & 
            pl.col("VOL_SHRINK_OK") &
            pl.col("ZTALK_GENE_OK") &
            pl.col("WEEKLY_MACD_OK")
        ).alias("b1_signal")
    ])


def calc_b1_factors_base(df: pl.LazyFrame, config: dict = None) -> pl.LazyFrame:
    """
    B1 选股因子计算 (V2.0.4b 还原版)
    
    Args:
        df: 输入 LazyFrame，需包含列: 
            code, date, open_adj, high_adj, low_adj, close_adj, volume, amount, market_cap_100m
        config: 策略参数配置，默认使用 DEFAULT_CONFIG
    
    Returns:
        LazyFrame，包含 b1_signal 等选股信号列
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    print("🛠️ [Strategy] 启动 B1 V2.0.4b 还原版...")

    return df.sort(["code", "date"]).with_columns([
        # 0. 基础位移
        pl.col("close_adj").shift(1).over("code").alias("prev_close"),
        pl.col("volume").shift(1).over("code").alias("prev_vol"),
        pl.col("open_adj").shift(1).over("code").alias("prev_open"),

        # 1. Ztalk 双均线系统 (WL & YL)
        pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
          .ewm_mean(span=10, adjust=False).over("code").alias("WL"),
        ((pl.col("close_adj").rolling_mean(14).over("code") + 
          pl.col("close_adj").rolling_mean(28).over("code") + 
          pl.col("close_adj").rolling_mean(57).over("code") + 
          pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

    ]).with_columns([
        # 2. 基础指标
        (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
        ((pl.col("close_adj") > pl.col("open_adj")) & (pl.col("close_adj") >= pl.col("prev_close"))).alias("real_yang"),
        ((pl.col("close_adj") < pl.col("open_adj")) & (pl.col("close_adj") <= pl.col("prev_close"))).alias("real_yin"),
        pl.col("volume").rolling_mean(40).over("code").alias("avg40"),
        pl.col("volume").shift(1).rolling_mean(40).over("code").alias("v40p"),

    ]).with_columns([
        # RSV
        pl.when(pl.col("kdj_den") == 0).then(50.0)
          .otherwise((pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")) / pl.col("kdj_den") * 100).alias("rsv"),

        # 红绿量能累加
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yang_p1"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_1"]).over("code").alias("vol_yin_p1"),
        (pl.col("volume") * pl.col("real_yang")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yang_p2"),
        (pl.col("volume") * pl.col("real_yin")).rolling_sum(cfg["YANGYIN_PERIOD_2"]).over("code").alias("vol_yin_p2"),

        # O85 & R55
        (pl.col("open_adj").rolling_min(28).over("code") + 
         cfg["BAD_K_OPEN_PCT"] * (pl.col("open_adj").rolling_max(28).over("code") - pl.col("open_adj").rolling_min(28).over("code"))).alias("O85"),
        (pl.col("close_adj").rolling_min(40).over("code") + 
         cfg["VIOLENT_POS_PCT"] * (pl.col("close_adj").rolling_max(40).over("code") - pl.col("close_adj").rolling_min(40).over("code"))).alias("R55"),

        # Max Yang Vol
        pl.when(pl.col("real_yang")).then(pl.col("volume")).otherwise(0)
          .rolling_max(28).over("code").alias("max_yang_vol_28"),

    ]).with_columns([
        # K
        pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),

        # 红绿比
        ((pl.col("vol_yang_p1") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p1")) | 
         (pl.col("vol_yang_p2") > cfg["YANGYIN_RATIO"] * pl.col("vol_yin_p2"))).alias("YANGYIN_OK"),

        # 流动性
        ((pl.col("amount").rolling_mean(28).over("code") / 1e8) >= cfg["LIQUIDITY_THRESHOLD"]).alias("LQ"),
        (pl.col("market_cap_100m") >= cfg["MV_THRESHOLD"]).alias("MVOK"),

        # BAD_K 计算
        (
            (pl.col("open_adj") >= pl.col("O85")) & 
            (pl.col("close_adj") < pl.col("prev_close")) & 
            (pl.col("close_adj") <= pl.col("open_adj")) & 
            (pl.col("volume") >= cfg["BAD_K_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("volume") >= cfg["BAD_K_AMNESTY_RATIO"] * pl.col("max_yang_vol_28")) 
        ).cast(pl.Int32).rolling_sum(cfg["BAD_K_LOOKBACK"]).over("code").alias("bad_k_count"),

        # 触发器
        (
            (pl.col("volume") > cfg["CLUSTER_VOL_RATIO"] * pl.col("prev_vol")) &
            (pl.col("close_adj") > pl.col("open_adj")) &
            (pl.col("volume") > pl.col("avg40"))
        ).alias("PLRY"),

        (
            ((pl.col("close_adj") > pl.col("prev_close")) & (pl.col("close_adj") >= pl.col("open_adj"))) &
            (pl.col("volume") > cfg["VIOLENT_VOL_RATIO"] * pl.col("v40p")) &
            (pl.col("close_adj") > pl.col("R55"))
        ).alias("KEY_K"),

    ]).with_columns([
        # D
        pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),

        # MAX28_OK: 好量压制坏量
        (
            pl.when(~pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code") 
            >= 
            pl.when(pl.col("real_yin")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code")
        ).alias("MAX28_OK"),

        (pl.col("bad_k_count") == 0).alias("GOOD28"),
        (pl.col("PLRY").cast(pl.Int32).rolling_sum(cfg["CLUSTER_PERIOD"]).over("code") >= cfg["CLUSTER_COUNT"]).alias("PLRY_CNT"),
        (pl.col("KEY_K").cast(pl.Int32).rolling_max(cfg["KEY_K_LOOKBACK"]).over("code") == 1).alias("KEY_K_EXIST"),

    ]).with_columns([
        (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
        (pl.col("PLRY_CNT") | pl.col("KEY_K_EXIST")).alias("TRIGGER"),

    ]).with_columns([
        (pl.col("J") <= cfg["J_THRESHOLD"]).alias("J_OK"),

    ]).with_columns([
        # 最终选股信号
        (
            (pl.col("WL") > pl.col("YL")) & 
            (pl.col("close_adj") > pl.col("YL")) &
            pl.col("TRIGGER") & 
            pl.col("J_OK") & 
            pl.col("LQ") & 
            pl.col("MVOK") & 
            pl.col("GOOD28") & 
            pl.col("MAX28_OK") & 
            pl.col("YANGYIN_OK") 
        ).alias("b1_signal")
    ])
