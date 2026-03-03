import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import os
    from datetime import datetime
    from utils import load_daily_data_full
    from utils import calc_b1_factors_wmacd
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
        ("2019-01-05", "2026-03-31"),  # 2025年慢牛行情延续
    ]

    print("🚀 [Step 1] 加载原始行情数据...")
    st_blacklist = get_st_blacklist_pl('2026-01-27') # 获取ST列表
    print("🔗 [Step 2] 合并基础数据...")
    q_full = (
        load_daily_data_full(conn).filter(
            ~pl.col("code").is_in(st_blacklist)
        )
    )
    # q_full = (
        # load_daily_data_full(conn)
    # )

    # ==============================================================================
    # ⚙️ 策略参数配置 V3.0 (Based on 10 Golden Cases)
    # ==============================================================================
    # 如果想放宽条件增加信号数量
    config_base = {"J_THRESHOLD": 13, "YANGYIN_RATIO": 1.5, "MV_THRESHOLD": 50}
    return (
        analyze_yearly_intensity,
        calc_b1_factors_wmacd,
        datetime,
        pl,
        print_backtest_report,
        q_full,
        run_backtest,
    )


@app.cell
def _(calc_b1_factors_wmacd, q_full):
    # 3. 执行计算
    print("⏳ 计算原始 B1 信号...")
    config_opt = {"MV_THRESHOLD": 25, "WEEKLY_WL_YL_FILTER": True}
    df_signals = calc_b1_factors_wmacd(q_full, config_opt)
    return (df_signals,)


@app.cell
def _(df_signals, print_backtest_report, run_backtest):
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
        ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
        ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
        ("2026-01-05", "2026-03-31"),  # 2025年慢牛行情延续
    ]

    # 导出信号供 Rust 使用
    # export_for_rust(
    #     df_signals,
    #     output_path="data/signals/market_data_wmacd.parquet",
    #     loose_periods=LOOSE_PERIODS,
    #     start_date='2019-01-01',
    #     extra_sort_cols=['rw_dif_pct']
    # )
    # print(f"导出完成")

    df_result_dynamic = run_backtest(df_signals, return_days=return_days, loose_periods=LOOSE_PERIODS, rank_by="rw_dif_pct", rank_ascending=False, top_n=200, stop_loss_pct=0.03)
    print_backtest_report(df_result_dynamic, return_days)
    return (df_result_dynamic,)


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
def _(datetime, df_result_dynamic, pl):
    df_result_dynamic.filter(
        (pl.col("date") == datetime(2026,1,8)) &
        (pl.col("b1_signal") == 1) 
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
