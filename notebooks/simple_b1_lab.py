import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import pandas as pd
    import os
    from datetime import datetime
    from utils import load_60m_data_adj

    # ==============================================================================
    # 1. 配置与数据加载 (Clean Version)
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/baostock_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    # ==============================================================================
    PERFECT_CASES_CONFIG = [
        # 教科书案例, 10大完美案例
        {"code": "sh.688799", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "sz.300689", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "sh.600601", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "sh.688321", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "sz.002940", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "sz.301076", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "sh.600184", "date": "2025-07-10", "name": "光电股份(回踩)"},
        {"code": "sz.002074", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "sh.605378", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "sh.600366", "date": "2025-08-06", "name": "宁波韵升(反包)"},
        # 以下是自己发现的案例
        {"code": "sz.000547", "date": "2025-11-13", "name": "航天发展(标准)"}
    ]

    perfect_case_list = [item.get("code") for item in PERFECT_CASES_CONFIG] # 获取完美案例 codes 列表

    print("🚀 [Step 1] 加载原始行情数据...")
    q_full = conn.sql(f"""
                select code, date, time, 
                open as open_adj,
                high as high_adj,
                low as low_adj,
                close as close_adj,
                volume,
                amount
                from v_kline_60m_qfq
                where code in {perfect_case_list}
                order by code, time
                """).pl()
    return PERFECT_CASES_CONFIG, datetime, pl, q_full


@app.cell
def _(pl, q_full):
    print("📊 数据基本信息:")
    print(f"总记录数: {q_full.height}")
    print(f"股票数: {q_full['code'].n_unique()}")
    print(f"时间范围: {q_full['time'].min()} 到 {q_full['time'].max()}")
    # 每只股票的数据量
    stock_counts = q_full.group_by("code").agg(pl.len().alias("count"))
    print("\n📈 各股票数据量:")
    print(stock_counts.sort("count", descending=True))
    return


@app.cell
def _(PERFECT_CASES_CONFIG, pl):
    # 转换配置为 DataFrame 方便后续处理
    cases_df = pl.DataFrame(PERFECT_CASES_CONFIG)
    cases_df = cases_df.with_columns(
        pl.col("date").str.to_date().alias("date")
    )
    return (cases_df,)


@app.cell
def _(pl):
    def calculate_technical_indicators(df):
        """
        在 Polars 中计算 MACD, Bollinger, RSI
        """
        # 1. 预计算 EWM (Polars 自带 ewm_mean)
        # MACD 参数: 12, 26, 9
        ema12 = df["close_adj"].ewm_mean(span=12, adjust=False)
        ema26 = df["close_adj"].ewm_mean(span=26, adjust=False)
        dif = ema12 - ema26
        dea = dif.ewm_mean(span=9, adjust=False)
        macd_hist = (dif - dea) * 2
    
        # 2. Bollinger Bands (20, 2)
        # rolling_std 需要 polars 最新版或通过 expression
        ma20 = df["close_adj"].rolling_mean(window_size=20)
        std20 = df["close_adj"].rolling_std(window_size=20)
        up_band = ma20 + 2 * std20
        low_band = ma20 - 2 * std20
        bandwidth = (up_band - low_band) / ma20
    
        # 3. RSI (14) - 简化版逻辑
        delta = df["close_adj"].diff()
        up = delta.clip(lower_bound=0)
        down = delta.clip(upper_bound=0).abs()
        # 这里的 RMA (Wilder's Smoothing) 可以用 EWM span=2N-1 近似
        roll_up = up.ewm_mean(span=27, adjust=False) 
        roll_down = down.ewm_mean(span=27, adjust=False)
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
        return df.with_columns([
            dif.alias("MACD_DIF"),
            dea.alias("MACD_DEA"),
            macd_hist.alias("MACD_Hist"),
            low_band.alias("BOLL_Lower"),
            bandwidth.alias("BOLL_Width"),
            rsi.alias("RSI")
        ])

    def scoring_resonance(df_60m):
        """
        多周期共振评分系统 (Resonance Scoring)
        """
        q = calculate_technical_indicators(df_60m.sort(["code", "time"]))
    
        # --- 定义信号逻辑 ---
    
        # 1. 量能信号 (30分): 极致缩量
        # 逻辑: 量比 < 0.5 得 30分, < 0.7 得 15分
        vol_ratio = pl.col("volume") / pl.col("volume").rolling_mean(20)
        score_vol = pl.when(vol_ratio < 0.5).then(30) \
                      .when(vol_ratio < 0.7).then(15) \
                      .otherwise(0)
    
        # 2. 动量信号 (30分): MACD 绿柱缩短 或 底背离
        # 逻辑: 绿柱在缩短 (Hist[t] > Hist[t-1] 且 Hist < 0)
        # 或者 DIF 位于低位但开始拐头向上
        score_macd = pl.when((pl.col("MACD_Hist") < 0) & (pl.col("MACD_Hist") > pl.col("MACD_Hist").shift(1))).then(30) \
                       .otherwise(0)
                   
        # 3. 结构信号 (20分): 布林带下轨支撑
        # 逻辑: 最低价触碰下轨，或者 收盘价在下轨附近
        pct_b = (pl.col("close_adj") - pl.col("BOLL_Lower")) / (pl.col("close_adj") * 0.01) # 距离下轨的百分比
        score_boll = pl.when(pct_b.abs() < 1.0).then(20).otherwise(0) # 距离下轨 1% 以内
    
        # 4. 弹性信号 (20分): RSI 超卖反弹
        # 逻辑: RSI < 40
        score_rsi = pl.when(pl.col("RSI") < 40).then(20).otherwise(0)
    
        # --- 汇总得分 ---
        q_scored = q.with_columns(
            (score_vol + score_macd + score_boll + score_rsi).alias("Resonance_Score")
        )
    
        return q_scored

    # 你的下一步：
    # 将这个 scoring_resonance 函数应用到你所有的 perfect_cases 上，
    # 看看买点出现时的平均得分是不是都在 70-80 分以上？
    return (scoring_resonance,)


@app.cell
def _(q_full, scoring_resonance):
    q_scored = scoring_resonance(q_full)
    return (q_scored,)


@app.cell
def _(cases_df):
    cases_df
    return


@app.cell
def _(datetime, pl, q_scored):
    q_scored.filter(
        (pl.col("code") == 'sz.000547') &
        (pl.col("date") >= datetime(2025,11,11))
    ).select(['code', 'time', 'Resonance_Score', 'MACD_DIF', 'MACD_DEA', 'MACD_Hist', 'BOLL_Lower', 'BOLL_Width', 'RSI'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
