"""
截面轮动因子工程模块

严格复刻 "T+1 尾盘轮动" 策略:
  - 信号日 T: 基于 T-1 及更早数据计算因子 (shift(1), 零前视)
  - 执行: T 日 14:30-15:00 买入 → T+1 日尾盘卖出
  - Label: fwd_ret_1d = close_T+1 / close_T - 1

基于 128 个交易日的日K线量价数据。

因子大类 (7 类 42 个):
  1. 动量/反转  — 多周期收益率、skip-1-month 动量
  2. 波动率    — 已实现波动率、波动率变化率、最大回撤、波动压缩
  3. 成交量    — 换手率、量价相关、异常放量、换手加速度
  4. 技术指标  — RSI, MACD, 布林带, ATR, 均线偏离
  5. 微观结构  — 振幅、影线、缺口、日内位置
  6. A股T+1短线 — 隔夜/日内收益分解、冲高探底、价格位置、Amihud
  7. 处置效应   — 换手率衰减成本线偏离 (EWM近似, 20/60d)
"""
import polars as pl


FACTOR_COLS = [
    # ── 1. 动量/反转 ──
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "mom_skip1m", "ret_max_5d", "ret_min_5d",
    # ── 2. 波动率 ──
    "vol_20d", "vol_60d", "vol_ratio", "max_drawdown_20d",
    "vol_compress",
    # ── 3. 成交量 ──
    "turnover_rate", "turnover_ma_ratio", "vol_price_corr_20d",
    "vol_std_20d", "abnormal_vol", "turnover_accel",
    # ── 4. 技术指标 ──
    "rsi_14", "macd_hist", "bb_position", "atr_14_pct",
    "ma_bias_20", "ma_bias_60", "close_to_high_20d",
    # ── 5. 微观结构 ──
    "amplitude", "upper_shadow", "lower_shadow", "body_ratio",
    "intraday_pos",
    # ── 6. A股T+1短线因子 ──
    "overnight_ret",       # 隔夜收益 (prev_close → open)
    "intraday_ret",        # 日内收益 (open → close)
    "overnight_ret_ma5",   # 5日隔夜收益均值
    "intraday_ret_ma5",    # 5日日内收益均值
    "high_open_pct",       # 日内冲高幅度 (high - open) / open
    "open_low_pct",        # 日内探底幅度 (open - low) / open
    "price_pos_20d",       # 20日收盘价位置 (0=最低 1=最高)
    "amihud_illiq_20d",    # Amihud非流动性 20日均值
    # ── 7. 处置效应/行为金融 ──
    "disp_bias_20",        # 20日EWM估算成本偏离 (disposition effect)
    "disp_bias_60",        # 60日EWM估算成本偏离 (disposition effect)
]


def calc_rotation_factors(df: pl.LazyFrame, lookback: int = 128) -> pl.LazyFrame:
    """
    计算截面轮动所需的全部量价因子。

    重要: 每个 .with_columns() 步骤先物化中间列, 避免嵌套 .over() 导致 Polars panic。

    Args:
        df: load_daily_data_full 返回的 LazyFrame
        lookback: 最大回溯窗口 (默认 128 天, 仅用于文档说明)

    Returns:
        LazyFrame, 在原始列基础上新增 ~42 个因子列
    """
    print("[Rotation] 计算截面轮动因子...")

    result = (
        df.sort(["code", "date"])

        # ── Step 1: 物化 lag-1 列 (所有后续计算的基础) ──────────────
        .with_columns([
            pl.col("close_adj").shift(1).over("code").alias("_c1"),       # T-1 close
            pl.col("open_adj").shift(1).over("code").alias("_o1"),        # T-1 open
            pl.col("high_adj").shift(1).over("code").alias("_h1"),        # T-1 high
            pl.col("low_adj").shift(1).over("code").alias("_l1"),         # T-1 low
            pl.col("volume").shift(1).over("code").alias("_v1"),          # T-1 volume
            pl.col("amount").shift(1).over("code").alias("_amt1"),        # T-1 amount
            pl.col("close_adj").shift(2).over("code").alias("_c2"),       # T-2 close
        ])

        # ── Step 2: 日收益率 (lag-1) ─────────────────────────────────
        .with_columns([
            (pl.col("_c1") / pl.col("_c2") - 1).alias("_ret_lag1"),      # T-1 日收益率
            # RSI 辅助: gain / loss
            pl.max_horizontal(pl.col("_c1") - pl.col("_c2"), pl.lit(0.0)).alias("_gain_lag1"),
            pl.max_horizontal(pl.col("_c2") - pl.col("_c1"), pl.lit(0.0)).alias("_loss_lag1"),
            # True Range (T-1)
            pl.max_horizontal(
                pl.col("_h1") - pl.col("_l1"),
                (pl.col("_h1") - pl.col("_c2")).abs(),
                (pl.col("_l1") - pl.col("_c2")).abs(),
            ).alias("_tr_lag1"),
            # 换手率 (T-1)
            (pl.col("_v1") / pl.col("circulating_capital").fill_null(1) * 100)
                .fill_nan(0.0).alias("turnover_rate"),
        ])

        # ── Step 3: 动量/反转因子 ────────────────────────────────────
        # 都是单层 .over("code") 的 shift, 安全
        .with_columns([
            (pl.col("_c1") / pl.col("_c2") - 1).alias("ret_1d"),
            (pl.col("_c1") / pl.col("close_adj").shift(6).over("code") - 1).alias("ret_5d"),
            (pl.col("_c1") / pl.col("close_adj").shift(11).over("code") - 1).alias("ret_10d"),
            (pl.col("_c1") / pl.col("close_adj").shift(21).over("code") - 1).alias("ret_20d"),
            (pl.col("_c1") / pl.col("close_adj").shift(61).over("code") - 1).alias("ret_60d"),
            (pl.col("_c1") / pl.col("close_adj").shift(121).over("code") - 1).alias("ret_120d"),
            (
                (pl.col("_c1") / pl.col("close_adj").shift(64).over("code") - 1)
                - (pl.col("_c1") / pl.col("close_adj").shift(22).over("code") - 1)
            ).alias("mom_skip1m"),
        ])

        # ── Step 4: 基于 _ret_lag1 的 rolling 指标 ──────────────────
        # _ret_lag1 已是物化列, 单层 .over("code") 即可
        .with_columns([
            pl.col("_ret_lag1").rolling_max(5).over("code").alias("ret_max_5d"),
            pl.col("_ret_lag1").rolling_min(5).over("code").alias("ret_min_5d"),
            pl.col("_ret_lag1").rolling_std(20).over("code").alias("vol_20d"),
            pl.col("_ret_lag1").rolling_std(60).over("code").alias("vol_60d"),
            pl.col("_ret_lag1").rolling_std(5).over("code").alias("_vol_5d"),
        ])

        # ── Step 5: 波动率衍生 + 最大回撤 ───────────────────────────
        .with_columns([
            (pl.col("vol_20d") / pl.max_horizontal(pl.col("vol_60d"), pl.lit(1e-8)))
                .alias("vol_ratio"),
            (1 - pl.col("_c1") / pl.col("_c1").rolling_max(20).over("code"))
                .alias("max_drawdown_20d"),
            (pl.col("_vol_5d") / pl.max_horizontal(pl.col("vol_20d"), pl.lit(1e-8)))
                .alias("vol_compress"),
        ])

        # ── Step 6: 成交量因子 ──────────────────────────────────────
        # _v1 已是物化列
        .with_columns([
            pl.col("_v1").rolling_mean(20).over("code").alias("_v1_ma20"),
            pl.col("_v1").rolling_std(20).over("code").alias("_v1_std20"),
            pl.col("turnover_rate").rolling_mean(20).over("code").alias("_tr_ma20"),
            pl.col("turnover_rate").rolling_mean(5).over("code").alias("_tr_ma5"),
        ])
        .with_columns([
            (pl.col("_v1") / pl.max_horizontal(pl.col("_v1_ma20"), pl.lit(1.0)))
                .alias("abnormal_vol"),
            (pl.col("_v1_std20") / pl.max_horizontal(pl.col("_v1_ma20"), pl.lit(1.0)))
                .alias("vol_std_20d"),
            (pl.col("turnover_rate") / pl.max_horizontal(pl.col("_tr_ma20"), pl.lit(1e-8)))
                .alias("turnover_ma_ratio"),
            (pl.col("_tr_ma5") / pl.max_horizontal(pl.col("_tr_ma20"), pl.lit(1e-8)))
                .alias("turnover_accel"),
        ])

        # ── Step 7: 量价相关系数 ────────────────────────────────────
        # _v1 和 _c1 都是物化列, rolling 计算安全
        .with_columns([
            (pl.col("_v1") * pl.col("_c1")).rolling_mean(20).over("code").alias("_vc_mean"),
            pl.col("_v1").rolling_mean(20).over("code").alias("_v_mean"),
            pl.col("_c1").rolling_mean(20).over("code").alias("_c_mean"),
            pl.col("_v1").rolling_std(20).over("code").alias("_v_std"),
            pl.col("_c1").rolling_std(20).over("code").alias("_c_std"),
        ])
        .with_columns([
            (
                (pl.col("_vc_mean") - pl.col("_v_mean") * pl.col("_c_mean"))
                / pl.max_horizontal(pl.col("_v_std") * pl.col("_c_std"), pl.lit(1e-10))
            ).alias("vol_price_corr_20d"),
        ])

        # ── Step 8: 技术指标 — RSI, MACD, 布林带, ATR, 均线 ────────
        .with_columns([
            # RSI 14
            pl.col("_gain_lag1").ewm_mean(span=14, adjust=False).over("code").alias("_rsi_gain"),
            pl.col("_loss_lag1").ewm_mean(span=14, adjust=False).over("code").alias("_rsi_loss"),
            # MACD DIF (12, 26) — 基于 _c1 物化列
            pl.col("_c1").ewm_mean(span=12, adjust=False).over("code").alias("_ema12"),
            pl.col("_c1").ewm_mean(span=26, adjust=False).over("code").alias("_ema26"),
            # 布林带
            pl.col("_c1").rolling_mean(20).over("code").alias("_bb_mid"),
            pl.col("_c1").rolling_std(20).over("code").alias("_bb_std"),
            # ATR 14 — 基于 _tr_lag1 物化列
            pl.col("_tr_lag1").rolling_mean(14).over("code").alias("_atr14"),
            # 均线
            pl.col("_c1").rolling_mean(20).over("code").alias("_ma20"),
            pl.col("_c1").rolling_mean(60).over("code").alias("_ma60"),
            # 20 日最高价
            pl.col("_h1").rolling_max(20).over("code").alias("_high_20d"),
        ])
        .with_columns([
            (100 - 100 / (1 + pl.col("_rsi_gain")
             / pl.max_horizontal(pl.col("_rsi_loss"), pl.lit(1e-10))))
                .alias("rsi_14"),
            (pl.col("_ema12") - pl.col("_ema26")).alias("macd_hist"),
            (
                (pl.col("_c1") - (pl.col("_bb_mid") - 2 * pl.col("_bb_std")))
                / pl.max_horizontal(4 * pl.col("_bb_std"), pl.lit(1e-8))
            ).alias("bb_position"),
            (pl.col("_atr14") / pl.max_horizontal(pl.col("_c1"), pl.lit(0.01)))
                .alias("atr_14_pct"),
            ((pl.col("_c1") - pl.col("_ma20")) / pl.max_horizontal(pl.col("_ma20"), pl.lit(0.01)) * 100)
                .alias("ma_bias_20"),
            ((pl.col("_c1") - pl.col("_ma60")) / pl.max_horizontal(pl.col("_ma60"), pl.lit(0.01)) * 100)
                .alias("ma_bias_60"),
            (1 - pl.col("_c1") / pl.max_horizontal(pl.col("_high_20d"), pl.lit(0.01)))
                .alias("close_to_high_20d"),
        ])

        # ── Step 9: 微观结构因子 ────────────────────────────────────
        # 所有 _h1, _l1, _o1, _c1, _c2 都是物化列
        .with_columns([
            ((pl.col("_h1") - pl.col("_l1"))
             / pl.max_horizontal(pl.col("_c2"), pl.lit(0.01)))
                .alias("amplitude"),
            (
                (pl.col("_h1") - pl.max_horizontal("_c1", "_o1"))
                / pl.max_horizontal(pl.col("_h1") - pl.col("_l1"), pl.lit(1e-8))
            ).alias("upper_shadow"),
            (
                (pl.min_horizontal("_c1", "_o1") - pl.col("_l1"))
                / pl.max_horizontal(pl.col("_h1") - pl.col("_l1"), pl.lit(1e-8))
            ).alias("lower_shadow"),
            (
                (pl.col("_c1") - pl.col("_o1")).abs()
                / pl.max_horizontal(pl.col("_h1") - pl.col("_l1"), pl.lit(1e-8))
            ).alias("body_ratio"),
            (
                (pl.col("_c1") - pl.col("_l1"))
                / pl.max_horizontal(pl.col("_h1") - pl.col("_l1"), pl.lit(1e-8))
            ).alias("intraday_pos"),
        ])

        # ── Step 10: A股T+1短线 — 隔夜/日内分解 + 微观 ─────────────
        .with_columns([
            (pl.col("_o1") / pl.max_horizontal(pl.col("_c2"), pl.lit(0.01)) - 1)
                .alias("overnight_ret"),
            (pl.col("_c1") / pl.max_horizontal(pl.col("_o1"), pl.lit(0.01)) - 1)
                .alias("intraday_ret"),
            ((pl.col("_h1") - pl.col("_o1"))
             / pl.max_horizontal(pl.col("_o1"), pl.lit(0.01)))
                .alias("high_open_pct"),
            ((pl.col("_o1") - pl.col("_l1"))
             / pl.max_horizontal(pl.col("_o1"), pl.lit(0.01)))
                .alias("open_low_pct"),
            (pl.col("_ret_lag1").abs()
             / pl.max_horizontal(pl.col("_amt1"), pl.lit(1.0)))
                .alias("_amihud_raw"),
        ])

        # ── Step 11: rolling on Step 10 物化列 ──────────────────────
        .with_columns([
            pl.col("overnight_ret").rolling_mean(5).over("code")
                .alias("overnight_ret_ma5"),
            pl.col("intraday_ret").rolling_mean(5).over("code")
                .alias("intraday_ret_ma5"),
            pl.col("_amihud_raw").rolling_mean(20).over("code")
                .alias("amihud_illiq_20d"),
            pl.col("_c1").rolling_min(20).over("code").alias("_c1_min_20"),
            pl.col("_c1").rolling_max(20).over("code").alias("_c1_max_20"),
        ])

        # ── Step 12: 价格位置衍生 ──────────────────────────────────
        .with_columns([
            ((pl.col("_c1") - pl.col("_c1_min_20"))
             / pl.max_horizontal(
                 pl.col("_c1_max_20") - pl.col("_c1_min_20"), pl.lit(0.01)))
                .alias("price_pos_20d"),
        ])

        # ── Step 13: 处置效应 — EWM 估算持仓成本线 (EHC) ─────────
        # EHC = EWM(TypicalPrice × Volume) / EWM(Volume)
        # span=20 → α≈0.095, span=60 → α≈0.033
        .with_columns([
            ((pl.col("_c1") + pl.col("_h1") + pl.col("_l1")) / 3 * pl.col("_v1"))
                .alias("_tp_v1"),
        ])
        .with_columns([
            pl.col("_tp_v1").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_pv_20"),
            pl.col("_v1").ewm_mean(span=20, adjust=False).over("code").alias("_ewm_v_20"),
            pl.col("_tp_v1").ewm_mean(span=60, adjust=False).over("code").alias("_ewm_pv_60"),
            pl.col("_v1").ewm_mean(span=60, adjust=False).over("code").alias("_ewm_v_60"),
        ])
        .with_columns([
            (pl.col("_ewm_pv_20") / pl.max_horizontal(pl.col("_ewm_v_20"), pl.lit(1e-10)))
                .alias("_ehc_20"),
            (pl.col("_ewm_pv_60") / pl.max_horizontal(pl.col("_ewm_v_60"), pl.lit(1e-10)))
                .alias("_ehc_60"),
        ])

        # ── Step 14: 处置效应偏离度 ──────────────────────────────
        .with_columns([
            (pl.col("_c1") / pl.max_horizontal(pl.col("_ehc_20"), pl.lit(0.01)) - 1)
                .alias("disp_bias_20"),
            (pl.col("_c1") / pl.max_horizontal(pl.col("_ehc_60"), pl.lit(0.01)) - 1)
                .alias("disp_bias_60"),
        ])
    )

    print(f"[Rotation] 因子计算完成, 共 {len(FACTOR_COLS)} 个因子")
    return result


def cross_section_normalize(df: pl.LazyFrame, factor_cols: list[str] = None) -> pl.LazyFrame:
    """
    截面标准化: 每天内对所有因子做 Z-Score + Winsorize。

    1. Z-Score: (x - mean) / std  (截面内)
    2. Winsorize: clip 到 [-5, 5] (去极端值)
    """
    cols = factor_cols or FACTOR_COLS
    print(f"[Rotation] 截面标准化 {len(cols)} 个因子...")

    exprs = []
    for c in cols:
        mean_expr = pl.col(c).mean().over("date")
        std_expr = pl.col(c).std().over("date")
        z = (pl.col(c) - mean_expr) / pl.max_horizontal(std_expr, pl.lit(1e-8))
        exprs.append(z.clip(-5.0, 5.0).alias(c))

    return df.with_columns(exprs)
