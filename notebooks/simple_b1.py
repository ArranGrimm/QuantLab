import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import os
    from datetime import datetime
    from utils import load_daily_data_full
    from utils import calc_b1_factors_opt, calc_b1_factors_base, calc_b1_factors_tg
    from utils import run_backtest, print_backtest_report, analyze_yearly_intensity
    from utils import get_st_blacklist_pl
    from utils import export_for_rust

    # ==============================================================================
    # 1. 配置与数据加载 (Clean Version)
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH)

    # ==============================================================================
    # Ztalk 体系核心：只在“活跃市值”强势期开仓
    MANUAL_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),  # 春季躁动
        ("2019-12-16", "2020-03-02"),  # 疫情反弹
        ("2020-06-19", "2020-07-15"),  # 证券带头的疯牛
        ("2020-12-24", "2021-01-25"),  # 新能源抱团主升
        ("2021-04-16", "2021-09-14"),  # 锂电光伏大主升
        ("2022-04-27", "2022-07-05"),  # 427大反弹
        ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO 狂潮
        ("2024-02-06", "2024-03-20"),  # 救市后AI反弹
        # ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
        ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
        ("2026-01-05", "2026-03-31"),  # 2025年慢牛行情延续
    ]

    print("🚀 [Step 1] 加载原始行情数据...")
    st_blacklist = get_st_blacklist_pl('2025-01-27') # 获取ST列表
    print("🔗 [Step 2] 合并基础数据...")
    q_full = (
        load_daily_data_full(conn).filter(
            ~pl.col("code").is_in(st_blacklist)
        )
    )

    # ==============================================================================
    # ⚙️ 策略参数配置 V3.0 (Based on 10 Golden Cases)
    # ==============================================================================
    # 如果想放宽条件增加信号数量
    config_base = {"J_THRESHOLD": 13, "YANGYIN_RATIO": 1.5, "MV_THRESHOLD": 50}
    config_opt = {}
    return (
        analyze_yearly_intensity,
        calc_b1_factors_opt,
        config_opt,
        export_for_rust,
        pl,
        q_full,
    )


@app.cell
def _(calc_b1_factors_opt, config_opt, q_full):
    # 3. 执行计算
    print("⏳ 计算原始 B1 信号...")
    # df_signals = calc_b1_factors_base(q_full, config)
    df_signals = calc_b1_factors_opt(q_full, config_opt)
    return (df_signals,)


@app.cell
def _(df_signals, export_for_rust):
    return_days = [5, 10, 15, 20, 25, 30]

    LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),  # 春季躁动
        ("2019-12-16", "2020-03-02"),  # 疫情反弹
        ("2020-06-19", "2020-07-15"),  # 证券带头的疯牛
        ("2020-12-24", "2021-01-25"),  # 新能源抱团主升
        ("2021-04-16", "2021-09-14"),  # 锂电光伏大主升
        ("2022-04-27", "2022-07-05"),  # 427大反弹
        ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO 狂潮
        ("2024-02-06", "2024-03-20"),  # 救市后AI反弹
        # ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
        ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
        ("2026-01-05", "2026-03-31"),  # 2025年慢牛行情延续
    ]

    # 导出信号供 Rust 使用
    export_for_rust(
        df_signals,
        output_path="data/signals/market_data.parquet",
        loose_periods=LOOSE_PERIODS,
        start_date='2019-01-01',
        # extra_sort_cols=['B1_Final_Score']
    )
    print(f"导出完成")

    # df_result_dynamic = run_backtest(df_signals, return_days, loose_periods=LOOSE_PERIODS, top_n=200, stop_loss_pct=0.03)
    # print_backtest_report(df_result_dynamic, return_days)
    return


@app.cell
def _(analyze_yearly_intensity, df_result_dynamic):
    # ==============================================================================
    # 6. 附录：年度交易频率压力测试 (Stress Test)
    # ==============================================================================
    # 统计 2024 或 2025 年的数据 (取决于你的数据源到哪一天)
    analyze_yearly_intensity(df_result_dynamic, 2024) 
    analyze_yearly_intensity(df_result_dynamic, 2025)
    analyze_yearly_intensity(df_result_dynamic, 2026)
    return


@app.cell
def _(df_signals, pl):
    PERFECT_CASES_CONFIG = [
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂(标准)"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技(蓄势)"},
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物(双底)"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康(压轴)"},
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材(激进)"},
        {"code": "600184_SH", "date": "2025-07-08", "name": "光电股份(回踩)"},
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科(趋势)"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池(突破)"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升(反包)"}
    ]
    def verify_perfect_cases(df_signals: pl.LazyFrame, cases_config: list):
        print("🔍 [Audit] 启动十大完美案例专项验证...")

        # 1. 构造案例查询表
        # 注意：这里我们假设 df_signals 包含了全量计算数据
        # 我们需要重新计算未来 30 天的数据，因为之前的 df_result 可能被 filter 掉了

        # 构造 Polars DataFrame 用于 Join
        target_df = pl.DataFrame(cases_config).with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # 2. 扩展计算未来 30 天的收益数据
        # 我们直接在 df_signals 上通过 code 和 date 关联，不需要全量重算
        # 但为了获取未来数据，我们需要先在 df_signals 里 shift

        print("⏳ 正在回溯历史行情 (T+1 -> T+30)...")

        # 定义评估周期
        horizons = [5, 10, 15, 20, 25, 30]

        # 核心验证逻辑
        audit_df = (
            df_signals
            .sort(["code", "date"])
            .with_columns([
                # 买入价：T+1 开盘价 (实战标准)
                pl.col("open_adj").shift(-1).over("code").alias("audit_buy_price"),

                # 择时状态 (复用之前的逻辑)
                # 这里简单起见，我们直接检查 b1_signal 字段
            ])
        )

        # 动态生成不同周期的收益列
        exprs = []
        for h in horizons:
            # 持有涨幅: (Close_T+N - Buy) / Buy
            exprs.append(
                ((pl.col("close_adj").shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"ret_{h}d")
            )
            # 最高涨幅: (Max_High_T+1_to_T+N - Buy) / Buy
            # rolling_max(h) 往前看，shift(-h) 移到现在
            exprs.append(
                ((pl.col("high_adj").rolling_max(h).shift(-h).over("code") / pl.col("audit_buy_price")) - 1).alias(f"max_{h}d")
            )

        # 执行计算并关联目标
        result = (
            audit_df
            .with_columns(exprs)
            .join(target_df.lazy(), on=["code", "date"], how="inner") # 只保留这10个
            .collect()
        )

        # 3. 输出报表
        print("\n====== ✨ 十大完美案例验证报告 ✨ ======")
        print(f"{'名称':<10} | {'日期':<9} | {'代码':<8} | {'信号?':<5} | {'买入价':<6} | {'5日最高':<8} | {'10日最高':<8} | {'20日最高':<8} | {'30日最高':<8} | {'30日持有':<8}")
        print("-" * 120)

        for row in result.iter_rows(named=True):
            name = row['name']
            code = row['code']
            date_str = row['date']
            is_signal = "✅" if row['b1_signal'] else "❌"
            buy = row['audit_buy_price']

            # 格式化涨幅
            def fmt(val): return f"{val*100:>6.2f}%" if val is not None else "   N/A"

            # 打印核心行
            print(f"{name:<10} | {date_str}  | {code:<8} | {is_signal:<5} | {buy:<6.2f} | {fmt(row['max_5d']):<8} | {fmt(row['max_10d']):<8} | {fmt(row['max_20d']):<8} | {fmt(row['max_30d']):<8} | {fmt(row['ret_30d']):<8}")


                # pl.col("TRIGGER") & 
                # pl.col("J_OK") & 
                # pl.col("LQ") & 
                # pl.col("MVOK") & 
                # pl.col("GOOD28") & 
                # pl.col("MAX28_OK") & 
                # pl.col("YANGYIN_OK") &
                # pl.col("SHAPE_OK") & 
                # pl.col("VOL_SHRINK_OK") &
                # pl.col("ZTALK_GENE_OK")
            # 如果没选出来，打印原因
            if not row['b1_signal']:
                # 简单诊断一下原因
                reasons = []
                if not row['MVOK']: reasons.append("流通市值不在")
                if not row['J_OK']: reasons.append(f"J值({row['J']:.1f})>13")
                if not row['MAX28_OK']: reasons.append("有天量阴")
                if not row['GOOD28']: reasons.append("有坏K线")
                if not row['YANGYIN_OK']: reasons.append(f"红绿比不足, p1: {row["vol_yang_p1"]/row["vol_yin_p1"]}, p2: {row["vol_yang_p2"]/row["vol_yin_p2"]}")
                if not row['TRIGGER']: reasons.append("无关键K")
                if not row['SHAPE_OK']: reasons.append("涨跌幅过大")
                if not row['VOL_SHRINK_OK']: reasons.append("28天缩量不够")
                if not row['ZTALK_GENE_OK']: reasons.append("不符合均线指纹")
                print(f"   ⚠️ 落选原因: {', '.join(reasons)}")

        print("-" * 120)


    # ==============================================================================
    # 执行验证
    # ==============================================================================
    # 假设 df_signals 依然在内存中 (即 run_strategy_b_with_manual_regime 的输入)
    if 'df_signals' in locals():
        verify_perfect_cases(df_signals, PERFECT_CASES_CONFIG)
    else:
        print("⚠️ df_signals 不在内存中，请先运行 Step 3 的 calc_b1_factors")
    return


if __name__ == "__main__":
    app.run()
