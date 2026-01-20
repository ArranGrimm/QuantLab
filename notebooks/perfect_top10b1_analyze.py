import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    import plotly.graph_objects as go
    from datetime import datetime

    # ==============================================================================
    # 1. 配置
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"
    # 我们的 10 大通缉令
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

    print("🚀 [Step 1] 加载原始行情数据...")

    # (A) 加载复前权行情
    files = [
        os.path.join(DATA_ROOT, "stock_day_adj", f"{case.get("code")}.parquet") for case in PERFECT_CASES_CONFIG
    ]

    q_adj = (
        pl.scan_parquet(
            files,
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
            pl.col("circulating_capital").cast(pl.Float64)
        ])
        .select(["code", "date", "circulating_capital"]).sort(["code", "date"])
    )

    # (C) 合并数据 (移除了所有市场指数相关代码)
    print("🔗 [Step 2] 合并基础数据...")
    q_full = (
        q_adj
        .join(q_raw, on=["code", "date"])
        .sort(["code", "date"])
        .join_asof(q_cap, on="date", by="code", strategy="backward")
        .with_columns([
            (pl.col("close_raw") * pl.col("circulating_capital") / 1e8).alias("market_cap_100m")
        ])
    )

    # ==============================================================================
    # 2. 数据探查
    # ==============================================================================
    # ==============================================================================
    # 1. 核心指标计算引擎 V2.0 (纯 Polars 实现)
    #    包含：Ztalk 双均线 (WL/YL) + 深度形态探查
    # ==============================================================================
    def calc_gene_vectors(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns([
            # --- A. 基础辅助 ---
            pl.col("close_adj").shift(1).over("code").alias("prev_close"),
            pl.col("open_adj").shift(1).over("code").alias("prev_open"),
            pl.col("volume").shift(1).over("code").alias("prev_vol"),

            # --- B. Ztalk 独家双均线系统 (WL & YL) ---
            # WL (白线): 双重 EMA10
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code")
              .ewm_mean(span=10, adjust=False).over("code").alias("WL"),

            # YL (黄线): 四均线加权 (14, 28, 57, 114)
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            # --- C. KDJ (N=9, M1=3, M2=3) ---
            (pl.col("high_adj").rolling_max(9).over("code") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_den"),
            (pl.col("close_adj") - pl.col("low_adj").rolling_min(9).over("code")).alias("kdj_num"),

        ]).with_columns([
            # RSV 计算
            pl.when(pl.col("kdj_den") == 0).then(50.0)
              .otherwise(pl.col("kdj_num") / pl.col("kdj_den") * 100).alias("rsv"),

            # --- D. 均线关系量化 (The Relationship) ---

            # 1. 股价 vs 白线 (%) -> "贴线程度" (太乖离是卖点，贴着是买点)
            ((pl.col("close_adj") - pl.col("WL")) / pl.col("WL") * 100).alias("Bias_C_WL"),

            # 2. 股价 vs 黄线 (%) -> "回踩深度" (回踩确认支撑的关键)
            ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_C_YL"),

            # 3. 白线 vs 黄线 (%) -> "趋势强度" (开口大小)
            ((pl.col("WL") - pl.col("YL")) / pl.col("YL") * 100).alias("Bias_WL_YL"),

        ]).with_columns([
            # K, D 计算
            pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K"),
            # 阳线判断 (用于计算缩量)
            ((pl.col("close_adj") > pl.col("open_adj"))).alias("is_yang"),

        ]).with_columns([
            pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D"),

            # Max Yang Vol (28天最大阳量)
            pl.when(pl.col("is_yang")).then(pl.col("volume")).otherwise(0)
              .rolling_max(28).over("code").alias("max_yang_vol_28"),

            # MA40 Vol (用于对比均量)
            pl.col("volume").rolling_mean(40).over("code").alias("vol_ma40"),

        ]).with_columns([
            (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),

            # --- E. 最终特征向量 (Feature Vectors) ---

            # 1. 缩量极致度 (Current / MaxYang)
            (pl.col("volume") / pl.col("max_yang_vol_28")).alias("Vol_Shrink_Ratio"),

            # 2. 实体大小 (Abs(C-O)/O)
            ( (pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("open_adj") * 100 ).alias("Body_Pct"),

            # 3. 基础市值
            (pl.col("market_cap_100m")).alias("MV"),

            # 4. Ztalk 趋势确认 (白>黄?)
            (pl.col("WL") > pl.col("YL")).alias("is_Bull_Trend")
        ])

    print("🧪 [Step 3] 计算全量指标 (基因测序)...")
    # 假设 q_full 是之前加载好的 LazyFrame
    df_indicators = calc_gene_vectors(q_full)

    # ==============================================================================
    # 2. 提取 10 大案例的"起爆前夜"指纹
    # ==============================================================================
    print("🔬 [Step 4] 提取完美案例指纹...")

    # 转换配置为 Polars DataFrame
    targets_df = pl.DataFrame(PERFECT_CASES_CONFIG).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    # 提取时空切片
    golden_samples = (
        df_indicators
        .join(targets_df.lazy(), on=["code", "date"], how="inner")
        .collect()
    )

    # ==============================================================================
    # 3. 纯 Polars 炫酷打印 (No Pandas Required)
    # ==============================================================================

    # 配置需要展示的列 (加入双均线探查)
    cols_to_show = [
        "name", "code", "date", 
        "J", "Vol_Shrink_Ratio", 
        "Bias_C_WL", "Bias_C_YL", "Bias_WL_YL", 
        "is_Bull_Trend"
    ]

    print("\n====== 🏆 10大完美案例·Ztalk均线基因图谱 (Pure Polars) ======")

    # 使用 pl.Config 上下文管理器来控制打印样式
    # tbl_cols=-1: 显示所有列
    # tbl_width_chars=200: 增加宽度防止换行
    # float_precision=2: 浮点数保留2位小数，看起来更清爽
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(golden_samples.select(cols_to_show))


    print("\n====== 📏 硬核包络线标准 (边界提取) ======")

    # 计算边界统计值
    stats_df = golden_samples.select([
        pl.col("J").min().alias("J_Min"),
        pl.col("J").max().alias("J_Max"),

        pl.col("Vol_Shrink_Ratio").max().alias("Shrink_Max (缩量上限)"),

        pl.col("Bias_C_WL").min().alias("Bias_C_WL_Min (贴线底限)"),
        pl.col("Bias_C_WL").max().alias("Bias_C_WL_Max (防追高)"),

        pl.col("Bias_C_YL").min().alias("Bias_C_YL_Min (回踩底限)"),
        pl.col("Bias_C_YL").max().alias("Bias_C_YL_Max (回踩上限)"),

        pl.col("Bias_WL_YL").min().alias("Bias_WL_YL_Min (粘合下限)"),
        pl.col("Bias_WL_YL").max().alias("Bias_WL_YL_Max (发散上限)"),

        pl.col("MV").min().alias("MV_Min (市值下限)"),
    ])

    # 技巧：使用 unpivot (原 melt) 替代转置 (.T)
    # 这样会生成一个 "指标 - 数值" 的纵向列表，比 Pandas 的转置更符合配置文件的格式
    stats_vertical = stats_df.unpivot(variable_name="Param (参数)", value_name="Value (边界值)")

    with pl.Config(tbl_rows=-1, float_precision=3):
        print(stats_vertical)

    # ==============================================================================
    # 4. 自动生成代码片段 (基于纯 Polars 提取的值)
    # ==============================================================================
    # 提取标量值 (Scalars) 用于 f-string
    row = stats_df.row(0)
    # 通过列名映射获取值 (更安全)
    cols = stats_df.columns
    vals = dict(zip(cols, row))

    print(f"\n====== 🚀 自动生成的 Ztalk 严选参数 ======")
    print(f"# 1. 均线位置")
    print(f"BIAS_WL_RANGE = ({vals['Bias_C_WL_Min (贴线底限)']:.2f}, {vals['Bias_C_WL_Max (防追高)']:.2f})")
    print(f"BIAS_YL_RANGE = ({vals['Bias_C_YL_Min (回踩底限)']:.2f}, {vals['Bias_C_YL_Max (回踩上限)']:.2f})")
    print(f"\n# 2. 趋势强度")
    print(f"TREND_GAP_RANGE = ({vals['Bias_WL_YL_Min (粘合下限)']:.2f}, {vals['Bias_WL_YL_Max (发散上限)']:.2f})")
    print(f"\n# 3. 情绪与量能")
    print(f"J_RANGE = ({vals['J_Min']:.2f}, {vals['J_Max']:.2f})")
    print(f"SHRINK_MAX = {vals['Shrink_Max (缩量上限)']:.2f}")
    return


@app.cell
def _():
    return


@app.cell
def _(get_source_data):
    # 假设你的prodt_cd列表有好多个: ['123', '456', '789']
    # 或者能从哪里复制进来，300多个产品代码问题不大
    # 变成这样也可以: "123,456,789"
    # 那调用的时候这样写就行

    prodt_cd = "123,456,789"
    input_params = {
        "prodt_cd": prodt_cd
    }
    df_data = get_source_data(input_params)


    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
