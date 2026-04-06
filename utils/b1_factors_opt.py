"""
B1 选股因子计算模块 (V3.0 精简优化版)
基于 V2.04b + 10大完美案例优化

核心逻辑：
- 触发器 (倍量柱/关键K) + J低位 + 流动性 + 无坏K线 + 红绿比 + 双均线多头
- 视觉量化：形态收敛 + 极致缩量 + 均线基因
"""
import polars as pl
from .sector_factors import get_sector_status

_A_SHARE_LOT_SIZE = 100.0

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

    # === 建仓波过热过滤 ===
    "WAVE_OVERHEAT_FILTER": False,  # 开关 (默认关闭, 需回测调参)
    "WAVE_MAX_TURNOVER": 30,        # 中长阳累计换手率阈值 (%)
    "WAVE_MAX_GAIN": 0.30,          # 累计涨幅阈值 (30%)
    "WAVE_YANG_THRESHOLD": 0.03,    # 中长阳判定: 实体涨幅 >= 3%
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

# ===== Phase 1: 准备周线和月线级别的上一周期截面 =====
    df_sorted = df.lazy().sort(["code", "date"])

    # 同时打上周和月的标签
    df_with_time = df_sorted.with_columns([
        pl.col("date").dt.truncate("1w").alias("week_start"),
        pl.col("date").dt.truncate("1mo").alias("month_start")
    ])

    # --- 周线级别预计算 ---
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
        # 新增：记录上周完整的红柱长度，用于过滤高位飘逸(动能衰退)
        (2 * (pl.col("w_dif").shift(1).over("code") - pl.col("w_dea").shift(1).over("code"))).alias("prev_w_hist")
    ])

    # --- 周线级别大周期择时 (MA多头排列 + WL>YL) ---
    df_weekly_trend = (
        df_weekly
        .with_columns([
            pl.col("weekly_close").rolling_mean(60).over("code").alias("w_ma60"),
            pl.col("weekly_close").rolling_mean(120).over("code").alias("w_ma120"),
            pl.col("weekly_close").rolling_mean(240).over("code").alias("w_ma240"),
            pl.col("weekly_close").ewm_mean(span=10, adjust=False).over("code")
              .ewm_mean(span=10, adjust=False).over("code").alias("w_wl"),
            ((pl.col("weekly_close").rolling_mean(14).over("code") +
              pl.col("weekly_close").rolling_mean(28).over("code") +
              pl.col("weekly_close").rolling_mean(57).over("code") +
              pl.col("weekly_close").rolling_mean(114).over("code")) / 4).alias("w_yl"),
        ])
        .with_columns([
            # TX1: 周线 MA60 上升 AND MA120 上升
            (
                (pl.col("w_ma60") > pl.col("w_ma60").shift(1).over("code")) &
                (pl.col("w_ma120") > pl.col("w_ma120").shift(1).over("code"))
            ).alias("w_tx1"),
            # TX2: 周线 MA60 > MA120 > MA240 (均线多头排列)
            (
                (pl.col("w_ma60") > pl.col("w_ma120")) &
                (pl.col("w_ma120") > pl.col("w_ma240"))
            ).alias("w_tx2"),
            # 周线 WL > YL (白线大于黄线)
            (pl.col("w_wl") > pl.col("w_yl")).alias("w_wl_gt_yl"),
        ])
        .select(["code", "week_start", "w_tx1", "w_tx2", "w_wl_gt_yl"])
    )

    # --- 月线级别预计算 ---
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

# ===== Phase 1.5: 在日线级别估算 Running MACD 并生成过滤因子 =====
    a12, a26, a9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0

    df_daily = (
        df_with_time
        .join(df_weekly_prev, on=["code", "week_start"], how="left")
        .join(df_monthly_prev, on=["code", "month_start"], how="left")
        .join(df_weekly_trend, on=["code", "week_start"], how="left")
        .with_columns([
            # 推算周线
            (a12 * pl.col("close_adj") + (1 - a12) * pl.col("prev_w_ema12")).alias("rw_ema12"),
            (a26 * pl.col("close_adj") + (1 - a26) * pl.col("prev_w_ema26")).alias("rw_ema26"),
            # 推算月线
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
            # 归一化周线 MACD 强度 (消除股价量纲，可用于排序)
            (pl.col("rw_dif") / pl.col("close_adj") * 100).alias("rw_dif_pct"),
        ])
        .with_columns([
            # 月线红柱：DIF 上穿 DEA 即可，捕捉零下金叉的早期主升浪 (数据验证: 11/11 完美案例 rm_hist > 0)
            (pl.col("rm_hist") > 0).alias("MONTHLY_MACD_OK"),
            # 周线大红区间过滤（包含防高位飘逸逻辑）：
            # 1. rw_hist > 0: 必须是金叉红柱状态
            # 2. rw_dif > 0: 必须在水上
            (
                (pl.col("rw_hist") > 0) & 
                (pl.col("rw_dif") > 0)
            ).alias("WEEKLY_MACD_OK"),
            # 周线大周期择时：MA多头排列(TX1+TX2)
            (
                pl.col("w_tx1").fill_null(False) &
                pl.col("w_tx2").fill_null(False)
            ).alias("WEEKLY_TREND_OK"),
            # 周线 WL > YL
            pl.col("w_wl_gt_yl").fill_null(False).alias("WEEKLY_WL_YL_OK"),
        ])
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

    # ===== Phase 2.5: 建仓波过热过滤 =====
    # 建仓波 = WL 金叉 YL 起, 累计中长阳换手率 + 累计涨幅超阈值 → 过热
    if cfg.get("WAVE_OVERHEAT_FILTER", False):
        yang_thr = cfg.get("WAVE_YANG_THRESHOLD", 0.03)
        print(f"  [Wave] 建仓波过热过滤: 中长阳>={yang_thr:.0%}, "
              f"换手>{cfg.get('WAVE_MAX_TURNOVER', 30)}%, "
              f"涨幅>{cfg.get('WAVE_MAX_GAIN', 0.30):.0%}")

        df_b1_signals = (
            df_b1_signals
            .with_columns([
                # 日换手率 (%): volume / 流通股本
                ((pl.col("volume") * _A_SHARE_LOT_SIZE) / pl.col("circulating_capital").fill_null(1) * 100)
                    .fill_nan(0.0).alias("turnover_rate"),
                # 中长阳线: 实体涨幅 >= 阈值
                ((pl.col("close_adj") / pl.col("open_adj") - 1) >= yang_thr).alias("_is_mid_yang"),
                # WL 金叉 YL (从下往上穿)
                (
                    (pl.col("WL") > pl.col("YL")) &
                    (pl.col("WL").shift(1).over("code") <= pl.col("YL").shift(1).over("code"))
                ).fill_null(False).alias("_wl_cross_yl"),
            ])
            .with_columns(
                pl.col("_wl_cross_yl").cast(pl.Int32).cum_sum().over("code").alias("wave_id")
            )
            .with_columns([
                # 波段起点收盘价
                pl.col("close_adj").first().over(["code", "wave_id"]).alias("_wave_start_close"),
                # 仅累加中长阳线当天的换手率
                pl.when(pl.col("_is_mid_yang"))
                    .then(pl.col("turnover_rate"))
                    .otherwise(0.0)
                    .cum_sum()
                    .over(["code", "wave_id"])
                    .alias("wave_yang_turnover"),
            ])
            .with_columns(
                ((pl.col("close_adj") - pl.col("_wave_start_close")) / pl.col("_wave_start_close"))
                    .alias("wave_gain")
            )
            .with_columns(
                (
                    (pl.col("wave_yang_turnover") > cfg.get("WAVE_MAX_TURNOVER", 30)) &
                    (pl.col("wave_gain") > cfg.get("WAVE_MAX_GAIN", 0.30)) &
                    (pl.col("wave_id") > 0)
                ).alias("WAVE_OVERHEAT")
            )
        )

    # ===== Phase 3: 最终信号 (B1 + 周线MACD过滤 + 可选大周期择时) =====
    signal_expr = (
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
        pl.col("WEEKLY_MACD_OK") &
        pl.col("MONTHLY_MACD_OK")
    )
    if cfg.get("WEEKLY_TREND_FILTER", False):
        signal_expr = signal_expr & pl.col("WEEKLY_TREND_OK")
    if cfg.get("WEEKLY_WL_YL_FILTER", False):
        signal_expr = signal_expr & pl.col("WEEKLY_WL_YL_OK")
    if cfg.get("WAVE_OVERHEAT_FILTER", False):
        signal_expr = signal_expr & ~pl.col("WAVE_OVERHEAT")

    return df_b1_signals.with_columns([
        signal_expr.alias("b1_signal")
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
