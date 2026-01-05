import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import torch
    import plotly.graph_objects as go
    import os

    # qmt数据目录
    DATA_ROOT = r"../QuantData/Ashare"

    try:
        from chronos import Chronos2Pipeline
        print("✅ 成功加载 Chronos2Pipeline")
    except ImportError:
        print("❌ 错误: 请更新库 pip install chronos-forecasting --upgrade")

    # ==========================================
    # 1. 加载 Chronos-2 (120M)
    # ==========================================
    # RTX 4050 6GB 跑这个模型没问题 (FP16模式下模型仅占 ~240MB，推理上下文占 1-2GB)
    MODEL_NAME = "amazon/chronos-2"

    print(f"🤖 正在加载 B1 专用模型: {MODEL_NAME} ...")

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="cuda",  # 使用 GPU
        torch_dtype=torch.bfloat16, # Ampere 架构(30/40系)推荐用 bfloat16，更稳
    )

    # 数据加载函数
    def load_data_subset(target_codes: list):
        """
        只加载指定 code_list 的数据，速度极快。
        返回 Pandas DataFrame，直接适配后续模型代码。
        """
        print(f"📦 正在通过 Polars 极速加载 {len(target_codes)} 只指定股票的数据...")

        # ==========================================
        # 1. 核心过滤逻辑 (Lazy Mode)
        # ==========================================

        # (A) 加载复前权行情 (带过滤)
        q_adj = (
            pl.scan_parquet(
                os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
                include_file_paths="file_path"
            )
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
            # 🔥 修改点 1: 在读取初期直接过滤 code
            .filter(pl.col("code").is_in(target_codes)) 
            .select(["code", "date", "open", "high", "low", "close", "volume", "amount"])
            .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
            .filter(pl.col("volume") > 0)
        )

        # (B) 加载 Raw 行情 (带过滤)
        q_raw = (
            pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path")
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
            # 🔥 修改点 2: 过滤
            .filter(pl.col("code").is_in(target_codes))
            .select(["code", "date", "close"]).rename({"close": "close_raw"})
        )

        # (C) 加载股本 (带过滤)
        q_cap = (
            pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path")
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
                pl.col("total_capital").cast(pl.Float64)
            ])
            # 🔥 修改点 3: 过滤
            .filter(pl.col("code").is_in(target_codes))
            .select(["code", "date", "total_capital"]).sort(["code", "date"])
        )

        # ==========================================
        # 2. 合并与计算
        # ==========================================
        # 执行 Join 计算
        df_lazy = (
            q_adj.join(q_raw, on=["code", "date"])
            .sort(["code", "date"])
            .join_asof(q_cap, on="date", by="code", strategy="backward")
            .with_columns([
                (pl.col("close_raw") * pl.col("total_capital") / 1e8).alias("market_cap_100m")
            ])
        )

        # 🔥 修改点 4: 直接 collect() 转为 Pandas
        # 因为数据量很小（只有几只票），转 Pandas 不会爆内存，且方便后续绘图
        return df_lazy.collect().to_pandas()


    # ==========================================
    # 2. 核心测试函数 (含“时间修复”补丁)
    # ==========================================
    def test_chronos2_fix_freq(df_all, code, date, name, seq_len=60):
        print(f"\n🧪 [Chronos-2] 正在分析: {name} ({code}) @ {date}")

        # --- A. 数据提取 ---
        df_all['date_dt'] = pd.to_datetime(df_all['date'])

        mask_hist = (df_all['code'] == code) & (df_all['date_dt'] <= pd.to_datetime(date))
        context_df = df_all[mask_hist].tail(seq_len).copy()

        if len(context_df) < 30:
            print("   ❌ 历史数据严重不足")
            return

        # =========================================================
        # 🩹 【补丁】构建连续的“伪造时间轴”
        # =========================================================
        # 逻辑：不管真实日期有无断层，我们生成一个完美的连续日期序列
        # 比如从 2020-01-01 开始，每天一天，绝不断更。
        # 这样 Chronos 就不会报 "Frequency" 错误了。

        fake_start_date = pd.to_datetime("2020-01-01")
        # 生成一个长度等于 seq_len 的连续时间序列
        continuous_dates = pd.date_range(start=fake_start_date, periods=len(context_df), freq="D")

        # 替换掉真实的日期 (仅用于欺骗模型)
        context_df['fake_date'] = continuous_dates

        # 构造输入 (Target=Price, Feature=Volume)
        context_input = context_df[['code', 'fake_date', 'close_adj', 'volume']].copy()

        # =========================================================

        # --- B. 调用预测 ---
        try:
            forecast_df = pipeline.predict_df(
                context_input,
                prediction_length=10,       
                quantile_levels=[0.1, 0.5, 0.9], 
                id_column="code",
                timestamp_column="fake_date", # 🔥 用伪造的时间列！
                target="close_adj"          
            )
        except Exception as e:
            print(f"❌ 预测发生异常: {e}")
            return

        # --- C. 结果解析 ---
        pred_prices = forecast_df['0.5'].values
        pred_low = forecast_df['0.1'].values
        pred_high = forecast_df['0.9'].values

        # 注意：预测出来的 fake_date 对我们没意义，我们需要把它映射回“未来10天”的概念
        # 但画图时为了方便，我们直接用 index 0, 1, 2... 来画

        # 计算预期涨幅
        current_price = context_input['close_adj'].iloc[-1]
        predicted_peak = np.max(pred_prices)
        pred_return = (predicted_peak - current_price) / current_price

        print(f"   💰 当前价: {current_price:.2f}")
        print(f"   🔮 AI预测最高: {predicted_peak:.2f} (预期涨幅: {pred_return:.2%})")

        # --- D. 真实数据验证 ---
        mask_future = (df_all['code'] == code) & (df_all['date_dt'] > pd.to_datetime(date))
        future_real = df_all[mask_future].head(10).copy()

        real_return = 0.0
        if not future_real.empty:
            real_peak = future_real['close_adj'].max()
            real_return = (real_peak - current_price) / current_price
            print(f"   👀 真实最高: {real_peak:.2f} (真实涨幅: {real_return:.2%})")

        # ==========================================
        # 3. Plotly 可视化 (使用相对坐标)
        # ==========================================
        fig = go.Figure()

        # 为了画图对齐，我们使用 相对索引 (0 ~ N)
        hist_len = len(context_df)
        x_hist = list(range(hist_len))
        x_pred = list(range(hist_len, hist_len + 10))

        # 1. 历史价格
        fig.add_trace(go.Scatter(
            x=x_hist, y=context_df['close_adj'],
            mode='lines', name='历史价格', line=dict(color='black')
        ))

        # 2. 预测区间
        fig.add_trace(go.Scatter(
            x=x_pred + x_pred[::-1],
            y=list(pred_high) + list(pred_low)[::-1],
            fill='toself', fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0), name='AI 置信区间'
        ))

        # 3. 预测中位数
        fig.add_trace(go.Scatter(
            x=x_pred, y=pred_prices,
            mode='lines+markers', name=f'AI 预测 (Exp: {pred_return:.1%})',
            line=dict(color='red', width=3, dash='dash')
        ))

        # 4. 真实走势
        if not future_real.empty:
            fig.add_trace(go.Scatter(
                x=x_pred[:len(future_real)], y=future_real['close_adj'],
                mode='lines', name=f'真实走势 (Real: {real_return:.1%})',
                line=dict(color='green', width=3)
            ))

        # 鼠标悬停显示真实日期
        # 这个稍微复杂点，但为了你看得爽，我们在 hover text 里加真实日期
        real_dates_hist = context_df['date_dt'].dt.strftime('%Y-%m-%d').values
        fig.update_traces(text=real_dates_hist, hovertemplate="Day: %{x}<br>Price: %{y:.2f}<br>Date: %{text}", selector=dict(name='历史价格'))

        fig.update_layout(
            title=f"<b>{name}</b> ({code}): Chronos-2 Forecast (Frequency Fixed)",
            yaxis_title="价格",
            xaxis_title="交易日序列 (Days)",
            template="plotly_white",
            height=500
        )
        fig.show()
    return load_data_subset, test_chronos2_fix_freq


@app.cell
def _(load_data_subset):
    PERFECT_CASES = [
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技"},
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康"},
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材"},
        {"code": "600184_SH", "date": "2025-07-10", "name": "光电股份"},
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升"}
    ]
    perfect_codes = [c['code'] for c in PERFECT_CASES]
    # 1. 加载
    df_pandas = load_data_subset(perfect_codes)
    return PERFECT_CASES, df_pandas


@app.cell
def _(PERFECT_CASES, df_pandas, test_chronos2_fix_freq):
    # ==========================================
    # 3. 运行测试
    # ==========================================
    # 假设 df_pandas 已经就绪
    if 'df_pandas' in locals():
        # 测试前 3 个完美案例
        for case in PERFECT_CASES[:3]:
            test_chronos2_fix_freq(df_pandas, case['code'], case['date'], case['name'])
    else:
        print("⚠️ 请先加载 df_pandas 数据")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
