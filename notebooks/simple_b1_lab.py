import marimo

__generated_with = "0.18.4"
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
        {"code": "sh.688799", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "sz.300689", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "sh.600601", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "sh.688321", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "sz.002940", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "sz.301076", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "sh.600184", "date": "2025-07-10", "name": "光电股份(回踩)"},
        {"code": "sz.002074", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "sh.605378", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "sh.600366", "date": "2025-08-06", "name": "宁波韵升(反包)"}
    ]

    perfect_case_list = [item.get("code") for item in PERFECT_CASES_CONFIG] # 获取完美案例 codes 列表


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
    """).pl()


    # print("🚀 [Step 1] 加载原始行情数据...")
    # perfect_case_list = [item.get("code") for item in PERFECT_CASES_CONFIG] # 获取完美案例 codes 列表
    # print("🔗 [Step 2] 合并基础数据...")
    # q_full = (
    #     load_60m_data_adj(conn).filter(
    #         pl.col("code").is_in(perfect_case_list)
    #     )
    # ).collect()
    return PERFECT_CASES_CONFIG, datetime, pd, pl, q_full


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
def _(PERFECT_CASES_CONFIG, datetime, pd, pl, q_full):
    # ==============================================================================
    # 深入分析：洗盘末期卖盘特征
    # ==============================================================================

    def analyze_late_wash_period(df, code, signal_date, days_before=2):
        """
        专门分析洗盘末期（信号日前几天）的卖盘特征
        """
        stock_data = df.filter(pl.col("code") == code).sort("time")
    
        # 找到信号日的第一根K线位置
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        signal_kline = stock_data.filter(
            pl.col("time").dt.date() == signal_dt.date()
        ).select(pl.first("time")).item()
    
        if not signal_kline:
            return None
    
        # 提取洗盘末期（信号日前days_before天）
        start_date = signal_dt - pd.Timedelta(days=days_before)
        wash_end_data = stock_data.filter(
            (pl.col("time").dt.date() >= start_date.date()) &
            (pl.col("time").dt.date() < signal_dt.date())
        )
    
        if wash_end_data.height == 0:
            return None
    
        # 区分阴阳线
        wash_end_data = wash_end_data.with_columns([
            (pl.col("close_adj") < pl.col("open_adj")).alias("is_yin"),
            (pl.col("close_adj") > pl.col("open_adj")).alias("is_yang"),
        ])
    
        # 分离阴线数据
        yin_data = wash_end_data.filter(pl.col("is_yin"))
    
        if yin_data.height == 0:
            return {
                "code": code,
                "signal_date": signal_date,
                "yin_count": 0,
                "yang_count": wash_end_data.height,
                "message": "洗盘末期无阴线"
            }
    
        # 计算更精细的卖盘特征
        result = {
            "code": code,
            "signal_date": signal_date,
            "wash_end_kline_count": wash_end_data.height,
            "yin_count": yin_data.height,
            "yang_count": wash_end_data.height - yin_data.height,
            "yin_percent": yin_data.height / wash_end_data.height,
        
            # 阴线成交量特征
            "yin_vol_avg": yin_data["volume"].mean(),
            "yin_vol_min": yin_data["volume"].min(),
            "yin_vol_max": yin_data["volume"].max(),
        
            # 阴线价格特征
            "yin_amp_avg": ((yin_data["high_adj"] - yin_data["low_adj"]) / yin_data["low_adj"]).mean(),
            "yin_body_avg": (abs(yin_data["close_adj"] - yin_data["open_adj"]) / yin_data["open_adj"]).mean(),
        
            # 对比整个洗盘期的阴线量（需要更多数据）
        }
    
        return result

    # 执行洗盘末期分析
    print("🔍 分析洗盘末期（信号日前2天）卖盘特征...\n")

    late_wash_results = []
    for case in PERFECT_CASES_CONFIG:
        result = analyze_late_wash_period(q_full, case["code"], case["date"], days_before=2)
        if result:
            result["name"] = case["name"]
            late_wash_results.append(result)
        
            if result["yin_count"] > 0:
                print(f"✅ {case['name']:15} | 阴线数: {result['yin_count']}/{result['wash_end_kline_count']} | "
                      f"阴线占比: {result['yin_percent']:.1%} | "
                      f"阴线均量: {result['yin_vol_avg']:.0f} | "
                      f"阴线振幅: {result['yin_amp_avg']*100:.1f}%")
            else:
                print(f"✅ {case['name']:15} | 洗盘末期无阴线！ | "
                      f"阳线数: {result['yang_count']}")

    # 汇总分析
    if late_wash_results:
        late_df = pl.DataFrame([r for r in late_wash_results if "yin_count" in r and r["yin_count"] > 0])
    
        if late_df.height > 0:
            print("\n" + "="*60)
            print("📊 洗盘末期（前2天）阴线特征汇总")
            print("="*60)
        
            print(f"\n📈 阴线出现频率:")
            print(f"  平均阴线数: {late_df['yin_count'].mean():.1f}根")
            print(f"  阴线占比: {late_df['yin_percent'].mean():.1%}")
        
            print(f"\n📈 阴线成交量特征:")
            print(f"  平均成交量: {late_df['yin_vol_avg'].mean():.0f}")
            print(f"  最小成交量: {late_df['yin_vol_min'].min():.0f}")
            print(f"  最大成交量: {late_df['yin_vol_max'].max():.0f}")
        
            print(f"\n📈 阴线价格特征:")
            print(f"  平均振幅: {late_df['yin_amp_avg'].mean()*100:.2f}%")
            print(f"  平均实体: {late_df['yin_body_avg'].mean()*100:.2f}%")
        
            # 识别卖盘枯竭模式
            print("\n🔍 卖盘枯竭模式识别:")
        
            exhausted_cases = []
            for result in late_wash_results:
                if "yin_count" in result:
                    # 枯竭特征：阴线少、成交量小、振幅小
                    is_exhausted = (
                        result["yin_count"] <= 2 and  # 阴线少
                        result.get("yin_vol_avg", 0) < 500000 and  # 成交量小（示例阈值）
                        result.get("yin_amp_avg", 1) < 0.02  # 振幅小
                    )
                
                    if is_exhausted:
                        exhausted_cases.append(result["name"])
                        print(f"  🎯 {result['name']}: 可能卖盘枯竭（阴线少且弱）")
        
            if not exhausted_cases:
                print("  未发现明显的卖盘枯竭模式")

    # ==============================================================================
    # 关键洞察：卖盘枯竭的重新定义
    # ==============================================================================

    print("\n" + "="*60)
    print("🤔 重新思考：什么是真正的卖盘枯竭？")
    print("="*60)

    print("""
    基于数据分析，我们发现：

    1. **阴线占比普遍较高**（平均56.7%，中位数53.5%）
       - 洗盘期确实有很多卖盘（阴线）
       - 但占比高不代表卖压强，还要看价格表现

    2. **卖盘枯竭可能表现为：**
       a) 阴线成交量减少（量比<0.7）
       b) 阴线振幅收窄（下跌无力）
       c) 阴线实体变小（卖盘犹豫）
       d) 阴线数量减少（卖盘减少）

    3. **从数据看，只有部分案例符合：**
       - 华纳药厂：阴线量比0.58 ↘（较好）
       - 微芯生物：阴线量比0.73 ↘（较好）
       - 新瀚新材：阴线量比0.84 ↘（一般）

    4. **其他案例卖盘并未明显枯竭**
       - 有的反而放量下跌（阴线量比>1）
       - 这可能意味着洗盘还未结束，或者...
    """)

    # ==============================================================================
    # 新思路：关注"最后一跌"的特征
    # ==============================================================================

    print("\n🔍 新思路：关注'最后一跌'的特征")

    def find_last_drop(df, code, signal_date, lookback_days=5):
        """
        寻找信号日前的最后一跌
        """
        stock_data = df.filter(pl.col("code") == code).sort("time")
    
        # 找到信号日
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
    
        # 提取信号日前lookback_days天的数据
        start_date = signal_dt - pd.Timedelta(days=lookback_days)
        pre_data = stock_data.filter(
            (pl.col("time").dt.date() >= start_date.date()) &
            (pl.col("time").dt.date() < signal_dt.date())
        )
    
        if pre_data.height == 0:
            return None
    
        # 找出最大的阴线（最后一跌）
        pre_data = pre_data.with_columns([
            (pl.col("close_adj") < pl.col("open_adj")).alias("is_yin"),
            (abs(pl.col("close_adj") - pl.col("open_adj")) / pl.col("open_adj")).alias("body_ratio"),
        ])
    
        yin_data = pre_data.filter(pl.col("is_yin"))
    
        if yin_data.height == 0:
            return {"message": "无阴线"}
    
        # 找到实体最大的阴线（可能是最后一跌）
        max_body_idx = yin_data["body_ratio"].arg_max()
        last_big_yin = yin_data[max_body_idx]
    
        # 找到该阴线之后到信号日的数据
        yin_time = last_big_yin["time"]
        after_yin = pre_data.filter(pl.col("time") > yin_time)
    
        return {
            "last_big_yin_date": yin_time.date(),
            "last_big_yin_body": last_big_yin["body_ratio"],
            "last_big_yin_volume": last_big_yin["volume"],
            "days_before_signal": (signal_dt.date() - yin_time.date()).days,
            "kline_after_yin": after_yin.height,
            "yin_after_count": after_yin.filter(pl.col("is_yin")).height,
        }

    print("\n📊 寻找'最后一跌'模式...")
    for case in PERFECT_CASES_CONFIG[:5]:  # 先看前5个
        result = find_last_drop(q_full, case["code"], case["date"])
        if result and "message" not in result:
            print(f"  {case['name']:15} | 最后一跌: {result['last_big_yin_date']} | "
                  f"实体: {result['last_big_yin_body']*100:.1f}% | "
                  f"距离信号日: {result['days_before_signal']}天")
        elif result:
            print(f"  {case['name']:15} | {result['message']}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
