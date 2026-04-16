import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import polars as pl
    from datetime import datetime
    from utils import load_daily_data_full
    from utils import calc_b1_factors_wmacd
    from utils import run_backtest, print_backtest_report, analyze_yearly_intensity
    from utils import get_st_blacklist_pl
    from utils import export_for_rust
    from utils.signal_export import build_b1_train_run_id, build_feature_hash, get_git_commit

    # ==============================================================================
    # 1. 配置与数据加载 (Clean Version)
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    # ==============================================================================
    # Ztalk 体系核心：只在“活跃市值”强势期开仓
    # RULE_LOOSE_PERIODS = [
    #     ("2019-02-11", "2019-04-10"),  # 春季躁动
    #     ("2019-12-16", "2020-03-02"),  # 疫情反弹
    #     ("2020-06-19", "2020-07-15"),  # 证券带头的疯牛
    #     ("2020-12-24", "2021-01-25"),  # 新能源抱团主升
    #     ("2021-04-16", "2021-09-14"),  # 锂电光伏大主升
    #     ("2022-04-27", "2022-07-05"),  # 427大反弹
    #     ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO 狂潮
    #     ("2024-02-06", "2024-03-20"),  # 救市后AI反弹
    #     ("2024-09-24", "2024-10-15"),  # 924 史诗级暴涨
    #     ("2025-04-09", "2025-09-04"),  # 2025年慢牛行情
    #     ("2026-01-05", "2026-02-02"),  # 2026年春季窗口
    # ]
    RULE_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),
        ("2019-12-16", "2020-03-02"),
        ("2020-06-19", "2020-07-15"),
        ("2020-12-24", "2021-01-25"),
        ("2021-04-20", "2021-06-16"),
        ("2021-07-12", "2021-08-17"),
        ("2021-08-25", "2021-09-16"),
        ("2022-04-28", "2022-07-25"),
        ("2022-10-14", "2022-12-19"),
        ("2023-01-06", "2023-05-12"),
        ("2023-08-01", "2023-08-11"),
        ("2023-08-30", "2023-09-20"),
        ("2023-10-26", "2023-12-20"),
        ("2024-01-02", "2024-01-17"),
        ("2024-01-25", "2024-01-30"),
        ("2024-02-07", "2024-03-25"),
        ("2024-04-18", "2024-05-15"),
        ("2024-07-12", "2024-07-23"),
        ("2024-08-01", "2024-08-12"),
        ("2024-09-02", "2024-11-14"),
        ("2025-01-15", "2025-01-27"),
        ("2025-02-07", "2025-02-28"),
        ("2025-04-09", "2025-04-18"),
        ("2025-05-07", "2025-09-04"),
        ("2026-01-06", "2026-02-02"),
        # ("2026-04-08", "2026-04-30"), # 暂时注释掉，不影响基线回测的结论
    ]
    RULE_SIGNAL_SOURCE = "rule_wmacd"
    RULE_LABEL = "rule_signal"
    RULE_MODEL_NAME = "wmacd_rule"
    RULE_FEATURE_SET_NAME = "rule_wmacd"
    RULE_SORT_FIELD = "rw_dif_pct"
    RULE_SORT_ASCENDING = False
    RULE_FEATURE_COLS = [RULE_SORT_FIELD]
    EXPORT_START_DATE = "2019-01-01"

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
        EXPORT_START_DATE,
        RULE_FEATURE_COLS,
        RULE_FEATURE_SET_NAME,
        RULE_LABEL,
        RULE_LOOSE_PERIODS,
        RULE_MODEL_NAME,
        RULE_SIGNAL_SOURCE,
        RULE_SORT_ASCENDING,
        RULE_SORT_FIELD,
        analyze_yearly_intensity,
        build_b1_train_run_id,
        build_feature_hash,
        calc_b1_factors_wmacd,
        datetime,
        export_for_rust,
        get_git_commit,
        pl,
        q_full,
    )


@app.cell
def _(calc_b1_factors_wmacd, q_full):
    # 3. 执行计算
    print("⏳ 计算原始 B1 信号...")
    config_opt = {"MV_THRESHOLD": 40, 
                  # "WEEKLY_WL_YL_FILTER": True, 
                    # "WAVE_OVERHEAT_FILTER": True,  # 开关 (默认关闭, 需回测调参)
                    # "WAVE_MAX_TURNOVER": 30,        # 中长阳累计换手率阈值 (%)
                    # "WAVE_MAX_GAIN": 0.30,          # 累计涨幅阈值 (30%)
                    # "WAVE_YANG_THRESHOLD": 0.03,    # 中长阳判定: 实体涨幅 >= 3%
                 }
    df_signals = calc_b1_factors_wmacd(q_full, config_opt)
    return config_opt, df_signals


@app.cell
def _(
    EXPORT_START_DATE,
    RULE_FEATURE_COLS,
    RULE_FEATURE_SET_NAME,
    RULE_LABEL,
    RULE_LOOSE_PERIODS,
    RULE_MODEL_NAME,
    RULE_SIGNAL_SOURCE,
    RULE_SORT_ASCENDING,
    RULE_SORT_FIELD,
    build_b1_train_run_id,
    build_feature_hash,
    config_opt,
    datetime,
    df_signals,
    export_for_rust,
    get_git_commit,
):
    train_timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_hash = build_feature_hash(RULE_FEATURE_COLS)
    export_meta = {
        "strategy": "b1",
        "label": RULE_LABEL,
        "model_name": RULE_MODEL_NAME,
        "feature_set_name": RULE_FEATURE_SET_NAME,
        "feature_mode": "rule",
        "feature_hash": feature_hash,
        "features": RULE_FEATURE_COLS,
        "feature_count": len(RULE_FEATURE_COLS),
        "train_timestamp_token": train_timestamp_token,
        "train_run_id": build_b1_train_run_id(
            RULE_LABEL,
            RULE_SIGNAL_SOURCE,
            RULE_MODEL_NAME,
            train_timestamp_token,
            feature_hash,
        ),
        "trained_at": trained_at,
        "git_commit": get_git_commit(),
        "notebook": "notebooks/simple_b1_lab.py",
        "model_params": {
            "rule_config": config_opt,
            "sort_field": RULE_SORT_FIELD,
            "sort_ascending": RULE_SORT_ASCENDING,
        },
        "signal_source": RULE_SIGNAL_SOURCE,
        "sort_field": RULE_SORT_FIELD,
        "sort_ascending": RULE_SORT_ASCENDING,
    }

    export_for_rust(
        df_signals,
        loose_periods=RULE_LOOSE_PERIODS,
        start_date=EXPORT_START_DATE,
        extra_sort_cols=[RULE_SORT_FIELD],
        artifact_metadata=export_meta,
    )
    print("导出完成")
    print(f"规则版 artifact train_run_id: {export_meta['train_run_id']}")
    print(f"规则版 source: {RULE_SIGNAL_SOURCE}")
    print(f"规则版排序字段: {RULE_SORT_FIELD} ({'升序' if RULE_SORT_ASCENDING else '降序'})")

    # df_result_dynamic = run_backtest(df_signals, return_days=return_days, loose_periods=LOOSE_PERIODS, rank_by="rw_dif_pct", rank_ascending=False, top_n=200, stop_loss_pct=0.03)
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
def _(datetime, df_result_dynamic, pl):
    df_result_dynamic.filter(
        (pl.col("date") == datetime(2026,1,8)) &
        (pl.col("b1_signal") == 1) 
    )
    return


if __name__ == "__main__":
    app.run()
