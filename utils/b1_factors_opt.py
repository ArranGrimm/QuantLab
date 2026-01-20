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
