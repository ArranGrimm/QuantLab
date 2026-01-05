import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pandas as pd
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

    # ==============================================================================
    # 2. 数据加载与探查
    # ==============================================================================
    def investigate_future_returns():
        print("🚀 [Data Probe] 正在调取原始档案，计算未来真实回报...")
    
        # 1. 仅加载这 10 只股票的数据 (为了快)
        target_codes = [c["code"] for c in PERFECT_CASES_CONFIG]
    
        q = (
            pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"), include_file_paths="file_path")
            .filter(pl.col("file_path").str.contains("|".join(target_codes))) # 文件名过滤加速
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
            .select(["code", "date", "close", "high"]) # 只需要 close 和 high
            .rename({"close": "close_adj", "high": "high_adj"})
            .sort(["code", "date"])
        )
    
        df_pool = q.collect().to_pandas()
    
        # 2. 循环计算收益
        results = []
    
        # 定义我们要看的窗口
        windows = [3, 5, 10, 20, 30, 40]
    
        for case in PERFECT_CASES_CONFIG:
            code = case['code']
            signal_date = case['date']
            name = case['name']
        
            # 找到当天的数据索引
            # 注意：这里我们通过 date string 匹配
            mask = (df_pool["code"] == code) & (df_pool["date"].astype(str) == signal_date)
            if not mask.any():
                print(f"❌ 数据缺失: {name} @ {signal_date}")
                continue
            
            base_idx = df_pool[mask].index[0]
            base_close = df_pool.loc[base_idx, "close_adj"]
        
            # 构建这一行的数据
            row_data = {
                "Name": name,
                "Date": signal_date,
                "Base_Price": base_close
            }
        
            # 检查每个时间窗口
            max_idx = len(df_pool) - 1
        
            for w in windows:
                future_idx = base_idx + w
            
                # 确保不越界，且必须是同一只股票
                if future_idx <= max_idx and df_pool.loc[future_idx, "code"] == code:
                    # A. 收盘价收益 (拿住不动的收益)
                    future_close = df_pool.loc[future_idx, "close_adj"]
                    ret_close = (future_close - base_close) / base_close
                
                    # B. 期间最高价收益 (摸到的最高点，用于验证 Label 是否触发)
                    # 切片范围: base_idx + 1 到 future_idx (含)
                    period_highs = df_pool.loc[base_idx+1 : future_idx, "high_adj"]
                    max_high = period_highs.max()
                    ret_max = (max_high - base_close) / base_close
                
                    row_data[f"Hold_{w}d"] = ret_close
                    row_data[f"Max_{w}d"] = ret_max
                else:
                    row_data[f"Hold_{w}d"] = None
                    row_data[f"Max_{w}d"] = None
                
            results.append(row_data)
        
        return pd.DataFrame(results)

    # ==============================================================================
    # 3. 运行并展示
    # ==============================================================================
    df_results = investigate_future_returns()

    # --- 美化展示 (颜色标记) ---
    def highlight_returns(val):
        if pd.isna(val): return ''
        color = 'red' if val > 0.05 else ('orange' if val > 0 else 'green')
        weight = 'bold' if val > 0.05 else 'normal'
        return f'color: {color}; font-weight: {weight}'

    # 设置显示格式为百分比
    print("\n📊 [验尸报告] 完美案例的真实涨幅数据 (Max_10d 是之前 Label 的标准):")
    # 只展示 Max Return (看看是否有机会止盈)
    cols_max = ["Name", "Date"] + [c for c in df_results.columns if "Max_" in c]
    print("-" * 80)
    print(df_results[cols_max].style.format({c: "{:.2%}" for c in df_results.columns if "_" in c}).to_string())

    print("\n📉 [持有体验] 如果死拿不动的收益 (Close Return):")
    cols_hold = ["Name", "Date"] + [c for c in df_results.columns if "Hold_" in c]
    print("-" * 80)
    print(df_results[cols_hold].style.format({c: "{:.2%}" for c in df_results.columns if "_" in c}).to_string())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
