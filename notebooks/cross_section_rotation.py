import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from scipy import stats

    from utils import load_daily_data_full, add_price_limit_cols
    from utils import get_st_blacklist_pl
    from utils.rotation_factors import (
        calc_rotation_factors,
        cross_section_normalize,
        FACTOR_COLS,
        FACTOR_GROUP_LABELS,
        FACTOR_GROUPS,
        FACTOR_TO_GROUP,
    )
    from utils.alpha158_factors import (
        ALPHA158_FACTOR_GROUP_LABELS,
        ALPHA158_FACTOR_GROUPS,
        ALPHA158_FACTOR_TO_GROUP,
        calc_alpha158_factors,
        resolve_alpha158_group_config,
    )
    from utils.factor_analysis import (
        build_daily_ic_frame,
        build_ic_summary_frame,
        compute_factor_decay,
        empty_group_summary_frame,
        empty_ic_summary_frame,
        extract_group_top_factor_cols,
        resolve_decay_factor_cols,
        summarize_factor_groups,
    )

    # ==============================================================================
    # Cell 1: 配置与数据加载
    # ==============================================================================
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Universe 参数
    MV_MIN = 80      # 最小流通市值 (亿)
    MV_MAX = 500     # 最大流通市值 (亿)
    MIN_LIST_DAYS = 60  # 最少上市天数
    START_DATE = "2020-09-01"  # 创业板注册制后
    NORMALIZE_MODE = "zscore"  # 可选: zscore / rank_pct / rank_gauss
    LABEL = "fwd_ret_1d"  # 可选: fwd_ret_{1/2/3/5}d / fwd_ret_{1/2/3/5}d_excess / fwd_ret_{1/2/3/5}d_rank_pct
    FEATURE_MODE = "core_plus_alpha158_top1"  # "all" / "pruned" / "core" / "alpha158" / "core_plus_alpha158" / "core_plus_alpha158_top1" / "all_plus_alpha158"
    ALPHA158_GROUP_MODE = "kbar_shape"  # 可选: all / kbar_shape / price_level / price_trend / trend_regression / range_position / timing_position / price_volume_corr / directionality / volume_dynamics
    ALPHA158_ANALYSIS_GROUP_MODE = "all"  # "all" / "match_training"
    ALPHA_DECAY_SOURCE = "alpha158_top1"  # "rotation" / "alpha158_top1" / "custom_list"
    ALPHA_DECAY_CUSTOM_FACTORS: tuple[str, ...] = ()
    RUN_ROTATION_CORR_DIAGNOSTICS = False
    RUN_ROTATION_CORE_SCREEN = False
    CORE_FEATURES_FROZEN = [
        "ret_max_5d",
        "vol_60d",
        "turnover_rate",
        "atr_14_pct",
        "amplitude",
        "intraday_ret_ma5",
        "disp_bias_20",
        "high_open_pct",
        "vol_std_20d",
        "abnormal_vol",
        "intraday_pos",
        "vol_price_corr_20d",
    ]

    print("🚀 [Step 1] 加载全量日线数据...")
    st_blacklist = get_st_blacklist_pl("2026-03-31")
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
    )

    print(f"✅ 参数: 流通市值 {MV_MIN}~{MV_MAX} 亿, 上市>{MIN_LIST_DAYS}天, 起始={START_DATE}")
    return (
        ALPHA158_ANALYSIS_GROUP_MODE,
        ALPHA158_FACTOR_GROUPS,
        ALPHA158_FACTOR_GROUP_LABELS,
        ALPHA158_FACTOR_TO_GROUP,
        ALPHA158_GROUP_MODE,
        ALPHA_DECAY_CUSTOM_FACTORS,
        ALPHA_DECAY_SOURCE,
        CORE_FEATURES_FROZEN,
        FACTOR_COLS,
        FACTOR_GROUPS,
        FACTOR_GROUP_LABELS,
        FACTOR_TO_GROUP,
        FEATURE_MODE,
        LABEL,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        NORMALIZE_MODE,
        RUN_ROTATION_CORE_SCREEN,
        RUN_ROTATION_CORR_DIAGNOSTICS,
        add_price_limit_cols,
        build_daily_ic_frame,
        build_ic_summary_frame,
        calc_alpha158_factors,
        calc_rotation_factors,
        compute_factor_decay,
        cross_section_normalize,
        empty_group_summary_frame,
        go,
        make_subplots,
        np,
        pl,
        px,
        q_full,
        resolve_alpha158_group_config,
        resolve_decay_factor_cols,
        stats,
        summarize_factor_groups,
    )


@app.cell
def _(
    ALPHA158_ANALYSIS_GROUP_MODE,
    ALPHA158_FACTOR_GROUP_LABELS,
    ALPHA158_GROUP_MODE,
    FACTOR_COLS,
    FEATURE_MODE,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    NORMALIZE_MODE,
    add_price_limit_cols,
    calc_alpha158_factors,
    calc_rotation_factors,
    cross_section_normalize,
    pl,
    q_full,
    resolve_alpha158_group_config,
):
    import os

    os.environ['RUST_BACKTRACE']='full'
    # ==============================================================================
    # Cell 2: 全量因子计算 → 市值过滤 → 截面标准化
    #
    # 关键流程:
    #   q_full (全量) → 因子计算 (连续序列) → forward return (连续序列)
    #   → 市值+上市天数过滤 (确定可交易 universe)
    #   → 截面标准化 (在 universe 内按 NORMALIZE_MODE 变换)
    #
    # 因子必须在连续序列上计算, 市值过滤只决定"哪些股票可交易"
    # ==============================================================================
    print("⏳ [Step 2] 计算训练特征 (全量股票, 保证序列连续)...")

    valid_feature_modes = {
        "all",
        "pruned",
        "core",
        "alpha158",
        "core_plus_alpha158",
        "core_plus_alpha158_top1",
        "all_plus_alpha158",
    }
    if FEATURE_MODE not in valid_feature_modes:
        raise ValueError(
            f"Unsupported FEATURE_MODE: {FEATURE_MODE}. "
            "Expected one of: all, pruned, core, alpha158, core_plus_alpha158, "
            "core_plus_alpha158_top1, all_plus_alpha158"
        )

    need_rotation_factors = FEATURE_MODE in {"all", "pruned", "core", "core_plus_alpha158", "core_plus_alpha158_top1", "all_plus_alpha158"}
    need_alpha158_train_factors = FEATURE_MODE in {"alpha158", "core_plus_alpha158", "core_plus_alpha158_top1", "all_plus_alpha158"}
    alpha158_analysis_mode = (
        ALPHA158_GROUP_MODE
        if str(ALPHA158_ANALYSIS_GROUP_MODE).strip().lower() == "match_training"
        else ALPHA158_ANALYSIS_GROUP_MODE
    )
    need_alpha158_analysis_factors = str(alpha158_analysis_mode).strip().lower() not in {"", "none", "disabled"}
    need_alpha158_factors = need_alpha158_train_factors or need_alpha158_analysis_factors

    df_factors = q_full
    active_factor_cols = []
    active_alpha158_factor_cols = []

    print(f"   特征模式: {FEATURE_MODE}")
    if need_rotation_factors:
        print(f"   计算 Rotation 因子: {len(FACTOR_COLS)} 个")
        df_factors = calc_rotation_factors(df_factors)
        active_factor_cols.extend(FACTOR_COLS)
    else:
        print("   跳过 Rotation 因子计算")

    if need_alpha158_factors:
        alpha158_train_config = resolve_alpha158_group_config(ALPHA158_GROUP_MODE)
        alpha158_analysis_config = resolve_alpha158_group_config(alpha158_analysis_mode)
        merged_group_keys = list(
            dict.fromkeys(
                [
                    *alpha158_train_config["group_keys"],
                    *alpha158_analysis_config["group_keys"],
                ]
            )
        )
        active_alpha158_factor_cols = list(
            dict.fromkeys(
                [
                    *alpha158_train_config["factor_cols"],
                    *alpha158_analysis_config["factor_cols"],
                ]
            )
        )
        merged_price_fields = tuple(
            dict.fromkeys(
                [
                    *alpha158_train_config["price_fields"],
                    *alpha158_analysis_config["price_fields"],
                ]
            )
        )
        merged_include_ops = tuple(
            dict.fromkeys(
                [
                    *(alpha158_train_config["include_ops"] or ()),
                    *(alpha158_analysis_config["include_ops"] or ()),
                ]
            )
        ) or None
        active_group_labels = [
            ALPHA158_FACTOR_GROUP_LABELS.get(group_key, group_key)
            for group_key in merged_group_keys
        ]
        print(f"   Alpha158 训练分组: {alpha158_train_config['group_mode_label']}")
        print(f"   Alpha158 分析分组: {alpha158_analysis_config['group_mode_label']}")
        print(f"   Alpha158 合并分组: {', '.join(active_group_labels)}")
        print(f"   计算 Alpha158 因子: {len(active_alpha158_factor_cols)} 个")
        df_factors = calc_alpha158_factors(
            df_factors,
            use_kbar=bool(alpha158_train_config["use_kbar"] or alpha158_analysis_config["use_kbar"]),
            price_fields=merged_price_fields,
            include=merged_include_ops,
        )
        active_factor_cols.extend(active_alpha158_factor_cols)
    else:
        print("   跳过 Alpha158 因子计算")

    print(f"   合计待标准化: {len(active_factor_cols)} 个")

    # Label: 在连续序列上计算 forward return, 避免因市值过滤产生的缺口
    df_with_label = df_factors.with_columns([
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
        (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
        (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
        (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
    ])

    # 涨跌停标记 (不删行, 仅打标记; 与 Rust bt-core 判定逻辑一致)
    df_with_label = add_price_limit_cols(df_with_label)

    # 市值 + 上市天数过滤 → 确定每日可交易 universe
    df_universe = (
        df_with_label
        .with_columns(
            pl.col("date").cum_count().over("code").alias("_list_days")
        )
        .filter(
            (pl.col("_list_days") >= MIN_LIST_DAYS) &
            (pl.col("market_cap_100m") >= MV_MIN) &
            (pl.col("market_cap_100m") <= MV_MAX)
        )
    )

    # 在可交易 universe 内做截面标准化
    print(f"   截面归一化模式: {NORMALIZE_MODE}")
    df_normalized = cross_section_normalize(
        df_universe,
        active_factor_cols,
        mode=NORMALIZE_MODE,
    )

    # 超额收益标签: 截面去均值 (在 universe 内计算, 消除市场 beta)
    base_label_cols = ["fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"]
    df_normalized = df_normalized.with_columns([
        (pl.col(label_name) - pl.col(label_name).mean().over("date")).alias(f"{label_name}_excess")
        for label_name in base_label_cols
    ])

    # 排序化标签: 将未来收益映射为当日截面分位数 [0, 1]
    rank_pct_exprs = []
    for label_name in base_label_cols:
        valid_mask = pl.col(label_name).is_not_null() & pl.col(label_name).is_not_nan()
        valid_count = (
            pl.when(valid_mask)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .sum()
            .over("date")
            .cast(pl.Float32)
        )
        rank_expr = pl.col(label_name).rank(method="average").over("date").cast(pl.Float32)
        rank_pct_exprs.append(
            pl.when(valid_mask & (valid_count > 1))
            .then((rank_expr - 1.0) / (valid_count - 1.0))
            .otherwise(pl.lit(None, dtype=pl.Float32))
            .alias(f"{label_name}_rank_pct")
        )
    df_normalized = df_normalized.with_columns(rank_pct_exprs)

    final_cols = [
        "code", "date", "open_adj", "high_adj", "low_adj", "close_adj", "vwap_adj",
        "pre_close_adj",
        "volume", "amount", "close_raw", "vwap_raw", "market_cap_100m",
        "circulating_capital",
        "fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d",
        "fwd_ret_1d_excess", "fwd_ret_2d_excess", "fwd_ret_3d_excess", "fwd_ret_5d_excess",
        "fwd_ret_1d_rank_pct", "fwd_ret_2d_rank_pct", "fwd_ret_3d_rank_pct", "fwd_ret_5d_rank_pct",
        "is_limit_up", "is_limit_down",
        *active_factor_cols,
    ]

    print("⏳ [Step 2] Collecting... (全量因子计算, 可能需要更长时间)")
    df_all = df_normalized.select(final_cols).collect()
    n_limit_up = df_all.filter(pl.col("is_limit_up")).height
    n_limit_down = df_all.filter(pl.col("is_limit_down")).height
    print(f"✅ 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    print(f"   日期范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
    print(f"   股票数量: {df_all['code'].n_unique()}")
    print(f"   涨停样本: {n_limit_up:,} ({n_limit_up/df_all.shape[0]*100:.2f}%)")
    print(f"   跌停样本: {n_limit_down:,} ({n_limit_down/df_all.shape[0]*100:.2f}%)")
    return (df_all,)


@app.cell
def _(df_all, pl):
    # ==============================================================================
    # Cell 2b: amount / volume 单位校验
    # 口径说明:
    #   - stock_daily.volume 单位 = 手
    #   - vwap_raw = amount / (volume * 100)
    #   - turnover_rate(%) = volume * 100 / circulating_capital * 100
    # ==============================================================================
    unit_metrics = (
        df_all
        .filter(
            (pl.col("volume") > 0) &
            (pl.col("amount") > 0) &
            (pl.col("close_raw") > 0) &
            pl.col("vwap_raw").is_not_null()
        )
        .select([
            (
                (pl.col("amount") / pl.col("volume") - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_if_volume_is_share"),
            (
                (pl.col("amount") / (pl.col("volume") * 100.0) - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_if_volume_is_lot"),
            (
                (pl.col("vwap_raw") - pl.col("close_raw")).abs()
                / pl.col("close_raw")
            ).median().alias("err_vwap_raw"),
        ])
    )
    unit_metrics_row = unit_metrics.row(0, named=True)
    print("🔎 [Step 2b] amount / volume 单位校验")
    print(f"   假设 volume=股: 相对误差中位数 {unit_metrics_row['err_if_volume_is_share']:.2%}")
    print(f"   假设 volume=手: 相对误差中位数 {unit_metrics_row['err_if_volume_is_lot']:.2%}")
    print(f"   当前 vwap_raw: 相对误差中位数 {unit_metrics_row['err_vwap_raw']:.2%}")
    print("   结论: stock_daily.volume 按“手”解释，turnover_rate 也必须先乘 100 还原股数。")

    if unit_metrics_row["err_vwap_raw"] > 0.02:
        raise ValueError("vwap_raw 与 close_raw 数量级不匹配，请检查 amount / volume 单位口径。")

    unit_sample = (
        df_all
        .filter(
            (pl.col("volume") > 0) &
            (pl.col("amount") > 0) &
            (pl.col("close_raw") > 0)
        )
        .select([
            "code",
            "date",
            "close_raw",
            "volume",
            "amount",
            (pl.col("amount") / pl.col("volume")).alias("px_if_volume_is_share"),
            (pl.col("amount") / (pl.col("volume") * 100.0)).alias("px_if_volume_is_lot"),
            "vwap_raw",
        ])
        .head(5)
    )
    print(unit_sample)
    return


@app.cell
def _(
    ALPHA158_ANALYSIS_GROUP_MODE,
    ALPHA158_FACTOR_GROUPS,
    ALPHA158_FACTOR_GROUP_LABELS,
    ALPHA158_FACTOR_TO_GROUP,
    FACTOR_COLS,
    FACTOR_GROUPS,
    FACTOR_GROUP_LABELS,
    LABEL,
    build_daily_ic_frame,
    build_ic_summary_frame,
    df_all,
    empty_group_summary_frame,
    go,
    make_subplots,
    pl,
    resolve_alpha158_group_config,
    summarize_factor_groups,
):
    # ==============================================================================
    # Cell 3: 因子 IC 分析底座
    # - Rotation: IC 汇总 + 累积 IC 曲线
    # - Alpha158: IC 汇总 + 分组 top1 提取
    # ==============================================================================
    from utils.ic_analysis import calc_factor_ic

    def run_factor_ic_foundation():
        df_valid_local = df_all.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())

        available_rotation_factors = [factor_name for factor_name in FACTOR_COLS if factor_name in df_all.columns]
        if available_rotation_factors:
            rotation_ic_results_local = calc_factor_ic(
                df_valid_local,
                factor_cols=available_rotation_factors,
                label=LABEL,
                min_samples=30,
            )
            rotation_ic_summary_local = build_ic_summary_frame(rotation_ic_results_local)
            rotation_daily_ic_local = build_daily_ic_frame(
                df_valid_local,
                factor_cols=available_rotation_factors,
                label=LABEL,
                min_samples=30,
            )
        else:
            print("ℹ️ [Cell 3] 当前 FEATURE_MODE 未加载 Rotation 因子，跳过 Rotation 因子 IC 分析。")
            rotation_ic_results_local = {}
            rotation_ic_summary_local = build_ic_summary_frame({})
            rotation_daily_ic_local = build_daily_ic_frame(df_valid_local, factor_cols=[], label=LABEL)

        analysis_mode = str(ALPHA158_ANALYSIS_GROUP_MODE).strip().lower()
        if analysis_mode in {"", "none", "disabled"}:
            alpha_group_keys = []
            available_alpha158_factors = []
        elif analysis_mode == "match_training":
            available_alpha158_factors = [
                factor_name for factor_name in ALPHA158_FACTOR_TO_GROUP if factor_name in df_all.columns
            ]
            alpha_group_keys = [
                group_key
                for group_key, factor_cols in ALPHA158_FACTOR_GROUPS.items()
                if any(factor in df_all.columns for factor in factor_cols)
            ]
        else:
            alpha_analysis_config = resolve_alpha158_group_config(ALPHA158_ANALYSIS_GROUP_MODE)
            alpha_group_keys = list(alpha_analysis_config["group_keys"])
            available_alpha158_factors = [
                factor_name
                for factor_name in alpha_analysis_config["factor_cols"]
                if factor_name in df_all.columns
            ]

        alpha_factor_groups = {
            group_key: ALPHA158_FACTOR_GROUPS[group_key]
            for group_key in alpha_group_keys
        }

        if available_alpha158_factors:
            print(f"🧪 [Alpha158 IC] 计算 {len(available_alpha158_factors)} 个 Alpha158 因子的 IC...", flush=True)
            alpha158_ic_results_local = calc_factor_ic(
                df_valid_local,
                factor_cols=available_alpha158_factors,
                label=LABEL,
                min_samples=30,
            )
            alpha158_ic_summary_local = build_ic_summary_frame(alpha158_ic_results_local)
            df_alpha158_group_summary_local = summarize_factor_groups(
                alpha158_ic_results_local,
                alpha_factor_groups,
                ALPHA158_FACTOR_GROUP_LABELS,
            )
        else:
            print("ℹ️ [Cell 3] 当前未加载可分析的 Alpha158 因子，跳过 Alpha158 分组 IC。")
            alpha158_ic_results_local = {}
            alpha158_ic_summary_local = build_ic_summary_frame({})
            df_alpha158_group_summary_local = empty_group_summary_frame()

        df_alpha158_top1_local = (
            df_alpha158_group_summary_local
            .select([
                "group_key",
                "group_name",
                "top_factor",
                "top_ic_mean",
                "top_icir",
                "top_abs_icir",
            ])
            .filter(pl.col("top_factor") != "")
            .sort("top_abs_icir", descending=True)
        )
        alpha158_top1_factor_cols_local = df_alpha158_top1_local["top_factor"].to_list()

        if rotation_ic_summary_local.height > 0 and rotation_daily_ic_local.height > 0:
            top_factors_local = rotation_ic_summary_local["factor"].head(6).to_list()
            fig_ic_local = make_subplots(rows=1, cols=1)
            for factor_name in top_factors_local:
                ic_cum = rotation_daily_ic_local.select(["date", factor_name]).drop_nulls().sort("date")
                fig_ic_local.add_trace(
                    go.Scatter(
                        x=ic_cum["date"].to_list(),
                        y=ic_cum[factor_name].cum_sum().to_list(),
                        name=factor_name,
                        mode="lines",
                    )
                )

            fig_ic_local.update_layout(
                title="Rotation Top 6 因子 — IC 累积曲线",
                xaxis_title="日期",
                yaxis_title="累积 IC",
                height=500,
                template="plotly_dark",
            )
            fig_ic_local.show()

        rotation_group_summary_local = summarize_factor_groups(
            rotation_ic_results_local,
            FACTOR_GROUPS,
            FACTOR_GROUP_LABELS,
        )

        return (
            rotation_group_summary_local,
            rotation_daily_ic_local,
            rotation_ic_results_local,
            rotation_ic_summary_local,
            alpha158_top1_factor_cols_local,
            alpha158_ic_results_local,
            alpha158_ic_summary_local,
            df_alpha158_group_summary_local,
            df_alpha158_top1_local,
        )

    (
        df_group_summary,
        rotation_daily_ic,
        rotation_ic_results,
        rotation_ic_summary,
        alpha158_top1_factor_cols,
        alpha158_ic_results,
        alpha158_ic_summary,
        df_alpha158_group_summary,
        df_alpha158_top1,
    ) = run_factor_ic_foundation()

    df_ic_summary = rotation_ic_summary
    ic_results = rotation_ic_results
    return (
        alpha158_top1_factor_cols,
        df_alpha158_group_summary,
        df_alpha158_top1,
        df_group_summary,
        ic_results,
        rotation_ic_summary,
    )


@app.cell
def _(df_alpha158_group_summary, df_alpha158_top1, df_group_summary):
    # ==============================================================================
    # Cell 3a: 分组总览面板
    # ==============================================================================
    def _print_group_summary(title: str, df_summary):
        if df_summary.height == 0:
            print(f"ℹ️ [{title}] 当前无可展示分组。")
            return

        print(title)
        print("=" * 108)
        print(f"  {'分组':<20} {'数量':>6} {'平均|ICIR|':>12} {'最佳|ICIR|':>12} {'组内最佳因子':<24}")
        print("-" * 108)
        for summary_row in df_summary.iter_rows(named=True):
            print(
                f"  {summary_row['group_name']:<20} {summary_row['n_factors']:>6} "
                f"{summary_row['mean_abs_icir']:>12.4f} {summary_row['max_abs_icir']:>12.4f} "
                f"{summary_row['top_factor']:<24}"
            )
        print("-" * 108)

    _print_group_summary("🧭 [Rotation Groups] Rotation 因子分组概览", df_group_summary)
    _print_group_summary("🧭 [Alpha158 Groups] Alpha158 因子分组概览", df_alpha158_group_summary)

    if df_alpha158_top1.height == 0:
        print("ℹ️ [Alpha158 Top1] 当前无 Alpha158 top1 因子可展示。")
    else:
        print("🏆 [Alpha158 Top1] 各分组 top1 因子 (按 |ICIR|)")
        print("=" * 96)
        print(f"  {'分组':<20} {'top1因子':<18} {'IC Mean':>10} {'ICIR':>10} {'|ICIR|':>10}")
        print("-" * 96)
        for row in df_alpha158_top1.iter_rows(named=True):
            print(
                f"  {row['group_name']:<20} {row['top_factor']:<18} "
                f"{row['top_ic_mean']:>10.4f} {row['top_icir']:>10.4f} {row['top_abs_icir']:>10.4f}"
            )
        print("-" * 96)
    return


@app.cell
def _(
    ALPHA_DECAY_CUSTOM_FACTORS: tuple[str, ...],
    ALPHA_DECAY_SOURCE,
    compute_factor_decay,
    df_all,
    df_alpha158_top1,
    go,
    make_subplots,
    resolve_decay_factor_cols,
    rotation_ic_summary,
):
    # ==============================================================================
    # Cell 3b: Alpha Decay 分析
    # - rotation: 使用 Rotation Top-N 因子
    # - alpha158_top1: 使用 Alpha158 各组 top1
    # - custom_list: 使用手工指定因子列表
    # ==============================================================================
    def run_alpha_decay():
        top_factors = resolve_decay_factor_cols(
            ALPHA_DECAY_SOURCE,
            rotation_ic_summary=rotation_ic_summary,
            alpha158_top1=df_alpha158_top1,
            custom_factor_cols=ALPHA_DECAY_CUSTOM_FACTORS,
            rotation_top_n=15,
        )
        top_factors = [factor for factor in top_factors if factor in df_all.columns]
        if not top_factors:
            print("ℹ️ [Cell 3b] 当前衰减分析没有可用因子，跳过 Alpha Decay。")
            return {}, {}

        print(f"📉 [Alpha Decay] 来源={ALPHA_DECAY_SOURCE}, 因子数={len(top_factors)}...", flush=True)
        decay_summary_local, avg_icir_local = compute_factor_decay(
            df_all,
            factor_cols=top_factors,
        )

        horizons = ["fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"]
        h_days = [1, 2, 3, 5]
        print("\n" + "=" * 100)
        print("  因子 IC 衰减对比")
        print("=" * 100)
        hdr = f"{'因子':<22}"
        for d in h_days:
            hdr += f"  {'IC_'+str(d)+'d':>8} {'ICIR_'+str(d)+'d':>8}"
        print(hdr)
        print("-" * 100)
        for factor_name in top_factors:
            row_str = f"{factor_name:<22}"
            for horizon in horizons:
                dd = decay_summary_local[horizon][factor_name]
                row_str += f"  {dd['ic_mean']:>8.4f} {dd['icir']:>8.4f}"
            print(row_str)
        print("-" * 100)

        print("\n📊 平均 |ICIR| 衰减:")
        for d in h_days:
            print(f"  {d}d: avg |ICIR| = {avg_icir_local.get(d, 0.0):.4f}")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Top 因子 |ICIR| 衰减", "平均 |ICIR| 衰减"],
        )

        for factor_name in top_factors[:8]:
            y_vals = [abs(decay_summary_local[horizon][factor_name]["icir"]) for horizon in horizons]
            fig.add_trace(
                go.Scatter(
                    x=h_days,
                    y=y_vals,
                    name=factor_name,
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

        avg_y = [avg_icir_local.get(d, 0.0) for d in h_days]
        fig.add_trace(
            go.Scatter(
                x=h_days,
                y=avg_y,
                name="平均",
                mode="lines+markers+text",
                text=[f"{v:.3f}" for v in avg_y],
                textposition="top center",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=450,
            template="plotly_dark",
            xaxis_title="持仓天数",
            xaxis2_title="持仓天数",
            yaxis_title="|ICIR|",
            yaxis2_title="平均 |ICIR|",
        )
        fig.show()

        return decay_summary_local, avg_icir_local

    decay_summary, avg_icir_decay = run_alpha_decay()
    return


@app.cell
def _(
    FACTOR_COLS,
    FEATURE_MODE,
    RUN_ROTATION_CORR_DIAGNOSTICS,
    df_all,
    ic_results,
    px,
):
    # ==============================================================================
    # Cell 3c: Rotation 相关性与冗余剪枝 (可选诊断)
    # ==============================================================================
    from utils.ic_analysis import calc_factor_corr, find_redundant_factors, print_corr_clusters

    def run_corr_analysis():
        available_rotation_factors = [factor_name for factor_name in FACTOR_COLS if factor_name in df_all.columns]
        should_run = RUN_ROTATION_CORR_DIAGNOSTICS or FEATURE_MODE == "pruned"
        if not should_run:
            print("ℹ️ [Cell 3c] Rotation 相关性诊断默认关闭，当前仅返回未剪枝因子列表。")
            return None, available_rotation_factors, available_rotation_factors, []
        if len(available_rotation_factors) < 2:
            print("ℹ️ [Cell 3c] 当前 FEATURE_MODE 未加载足够的 Rotation 因子，跳过相关性分析。")
            return None, available_rotation_factors, available_rotation_factors, []

        corr_mat, factor_names = calc_factor_corr(df_all, available_rotation_factors)

        print_corr_clusters(corr_mat, factor_names, threshold=0.7)

        keep, drop, _decisions = find_redundant_factors(
            corr_mat, factor_names, ic_results=ic_results, threshold=0.85
        )

        fig = px.imshow(
            corr_mat,
            x=factor_names,
            y=factor_names,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"因子 Spearman 相关矩阵 ({len(factor_names)} 因子)",
        )
        fig.update_layout(height=800, width=900, template="plotly_dark")
        fig.show()

        return corr_mat, factor_names, keep, drop

    corr_matrix, corr_names, factors_keep, factors_drop = run_corr_analysis()
    return (factors_keep,)


@app.cell
def _(
    CORE_FEATURES_FROZEN,
    FACTOR_COLS,
    FACTOR_GROUPS,
    FACTOR_GROUP_LABELS,
    FACTOR_TO_GROUP,
    RUN_ROTATION_CORE_SCREEN,
    factors_keep,
    ic_results,
):
    # ==============================================================================
    # Cell 3d: Rotation core feature screen (历史治理工具)
    # 默认直接返回冻结的 core_12，不再作为主流程依赖
    # ==============================================================================
    def run_core_factor_screen():
        if not RUN_ROTATION_CORE_SCREEN:
            print("ℹ️ [Cell 3d] 默认不再运行 core feature screen，使用冻结的 core_12 配置。")
            print(f"   core_12 = {', '.join(CORE_FEATURES_FROZEN)}")
            return list(CORE_FEATURES_FROZEN)
        if not ic_results:
            print("ℹ️ [Cell 3d] 当前 FEATURE_MODE 未加载 Rotation 因子，回退到冻结的 core_12 配置。")
            return list(CORE_FEATURES_FROZEN)

        keep_set = set(factors_keep)
        ranked_rows = []
        for factor_name in FACTOR_COLS:
            if factor_name not in ic_results:
                continue
            group_key_local = FACTOR_TO_GROUP.get(factor_name, "ungrouped")
            ranked_rows.append(
                {
                    "factor": factor_name,
                    "group_key": group_key_local,
                    "group_name": FACTOR_GROUP_LABELS.get(group_key_local, group_key_local),
                    "abs_icir": abs(float(ic_results[factor_name]["icir"])),
                    "is_pruned_keep": factor_name in keep_set,
                }
            )

        core_primary = []
        secondary_pool = []

        print("🎯 [Core Factors] 分组核心因子筛查")
        print("=" * 108)
        print(f"  {'分组':<20} {'候选因子':<24} {'|ICIR|':>10} {'保留?':>8} {'角色':>8}")
        print("-" * 108)

        for group_key_local, _factors_local in FACTOR_GROUPS.items():
            group_rows = [r for r in ranked_rows if r["group_key"] == group_key_local]
            group_rows.sort(key=lambda r: r["abs_icir"], reverse=True)
            if not group_rows:
                continue

            kept_rows = [r for r in group_rows if r["is_pruned_keep"]]
            primary = kept_rows[0] if kept_rows else group_rows[0]
            core_primary.append(primary["factor"])

            print(
                f"  {FACTOR_GROUP_LABELS.get(group_key_local, group_key_local):<20} "
                f"{primary['factor']:<24} {primary['abs_icir']:>10.4f} "
                f"{('是' if primary['is_pruned_keep'] else '否'):>8} {'主因子':>8}"
            )

            follow_rows = kept_rows[1:] if primary["is_pruned_keep"] else kept_rows
            for candidate_row in follow_rows:
                if (
                    candidate_row["abs_icir"] >= 0.08
                    and candidate_row["abs_icir"] >= primary["abs_icir"] * 0.60
                ):
                    secondary_pool.append(candidate_row)

        secondary_pool.sort(key=lambda r: r["abs_icir"], reverse=True)
        core_target_size = 12
        extra_slots = max(0, core_target_size - len(core_primary))
        core_factors_local = list(core_primary)
        for candidate_row in secondary_pool:
            if candidate_row["factor"] in core_factors_local:
                continue
            if len(core_factors_local) >= len(core_primary) + extra_slots:
                break
            core_factors_local.append(candidate_row["factor"])
            print(
                f"  {candidate_row['group_name']:<20} {candidate_row['factor']:<24} {candidate_row['abs_icir']:>10.4f} "
                f"{('是' if candidate_row['is_pruned_keep'] else '否'):>8} {'补充':>8}"
            )

        print("-" * 108)
        print(f"  建议 core feature set ({len(core_factors_local)}): {', '.join(core_factors_local)}")
        return core_factors_local

    core_factors = run_core_factor_screen()
    return (core_factors,)


@app.cell
def _():
    # Cell 5: (已移除 — 旧可视化依赖线性回测, 回测逻辑已迁移至 Rust)
    return


@app.cell
def _(
    ALPHA158_ANALYSIS_GROUP_MODE,
    ALPHA158_GROUP_MODE,
    FACTOR_COLS,
    FEATURE_MODE,
    LABEL,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    NORMALIZE_MODE,
    alpha158_top1_factor_cols,
    core_factors,
    df_all,
    factors_keep,
    np,
    pl,
    resolve_alpha158_group_config,
):
    # ==============================================================================
    # Cell 6: LightGBM Walk-Forward 打分
    # 模型只负责打分, 导出与回测解耦
    #
    # FEATURE_MODE:
    #   - "all" = 全部 Rotation 因子
    #   - "pruned" = 相关性剪枝后的 Rotation 因子
    #   - "core" = 分组核心 Rotation 因子
    #   - "alpha158" = 全量 Alpha158
    #   - "core_plus_alpha158" = core_12 + Alpha158
    #   - "core_plus_alpha158_top1" = core_12 + Alpha158 各组 top1
    #   - "all_plus_alpha158" = Rotation 全量 + Alpha158
    # ==============================================================================
    def run_lgbm_scoring():
        from datetime import datetime
        import lightgbm as lgb
        import warnings
        from utils.signal_export import build_feature_hash, build_rotation_train_run_id, get_git_commit
        warnings.filterwarnings("ignore", category=UserWarning)

        TRAIN_WINDOW = 480
        RETRAIN_FREQ = 20
        TOP_N = 20
        alpha158_group_config = resolve_alpha158_group_config(ALPHA158_GROUP_MODE)
        alpha158_feature_cols = list(alpha158_group_config["factor_cols"])
        alpha158_top1_feature_cols = list(
            dict.fromkeys(
                factor_name
                for factor_name in alpha158_top1_factor_cols
                if factor_name in df_all.columns
            )
        )

        if FEATURE_MODE == "pruned":
            feature_cols = list(factors_keep)
        elif FEATURE_MODE == "core":
            feature_cols = list(core_factors)
        elif FEATURE_MODE == "alpha158":
            feature_cols = list(alpha158_feature_cols)
        elif FEATURE_MODE == "core_plus_alpha158":
            feature_cols = list(dict.fromkeys([*core_factors, *alpha158_feature_cols]))
        elif FEATURE_MODE == "core_plus_alpha158_top1":
            if not alpha158_top1_feature_cols:
                raise ValueError(
                    "FEATURE_MODE=core_plus_alpha158_top1 需要先运行 Cell 3，"
                    "并保证 ALPHA158_ANALYSIS_GROUP_MODE 能产出 Alpha158 top1。"
                )
            feature_cols = list(dict.fromkeys([*core_factors, *alpha158_top1_feature_cols]))
        elif FEATURE_MODE == "all_plus_alpha158":
            feature_cols = list(dict.fromkeys([*FACTOR_COLS, *alpha158_feature_cols]))
        else:
            feature_cols = list(FACTOR_COLS)

        lgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 100,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        if FEATURE_MODE == "pruned":
            mode_desc = f"pruned ({len(feature_cols)})"
        elif FEATURE_MODE == "core":
            mode_desc = f"core ({len(feature_cols)})"
        elif FEATURE_MODE == "alpha158":
            mode_desc = f"alpha158 ({len(feature_cols)})"
        elif FEATURE_MODE == "core_plus_alpha158":
            mode_desc = f"core_plus_alpha158 ({len(feature_cols)})"
        elif FEATURE_MODE == "core_plus_alpha158_top1":
            mode_desc = f"core_plus_alpha158_top1 ({len(feature_cols)})"
        elif FEATURE_MODE == "all_plus_alpha158":
            mode_desc = f"all_plus_alpha158 ({len(feature_cols)})"
        else:
            mode_desc = f"all ({len(feature_cols)})"
        print("🤖 LightGBM Walk-Forward 打分", flush=True)
        print(f"   训练窗口: {TRAIN_WINDOW}天, 重训: 每{RETRAIN_FREQ}天, 标签: {LABEL}", flush=True)
        print(f"   特征模式: {mode_desc}, Top-{TOP_N}", flush=True)
        if FEATURE_MODE in {"alpha158", "core_plus_alpha158", "all_plus_alpha158"}:
            print(f"   Alpha158 分组: {alpha158_group_config['group_mode_label']}", flush=True)
        elif FEATURE_MODE == "core_plus_alpha158_top1":
            print(f"   Alpha158 Top1 来源: {ALPHA158_ANALYSIS_GROUP_MODE}", flush=True)
            print(f"   Alpha158 Top1 因子数: {len(alpha158_top1_feature_cols)}", flush=True)
        print(f"   截面归一化: {NORMALIZE_MODE}", flush=True)

        df_valid_ml = (
            df_all
            .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())
            .sort("date")
        )

        X_all_np = df_valid_ml.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
        y_all_np = df_valid_ml[LABEL].to_numpy().astype(np.float32)
        dates_np = df_valid_ml["date"].to_numpy()
        codes_np = df_valid_ml["code"].to_numpy()
        is_limit_up_np = df_valid_ml["is_limit_up"].fill_null(False).to_numpy()

        unique_dates_ml = np.unique(dates_np)
        unique_dates_ml.sort()
        n_dates = len(unique_dates_ml)

        date_start = np.searchsorted(dates_np, unique_dates_ml, side="left")
        date_end = np.searchsorted(dates_np, unique_dates_ml, side="right")

        score_dates = []
        score_codes = []
        score_values = []
        model = None
        last_train_idx = -RETRAIN_FREQ

        for i in range(TRAIN_WINDOW, n_dates):
            cur_date = unique_dates_ml[i]

            if i - last_train_idx >= RETRAIN_FREQ or model is None:
                ts = date_start[i - TRAIN_WINDOW]
                te = date_end[i - 1]
                X_tr = X_all_np[ts:te]
                y_tr = y_all_np[ts:te]

                valid = np.isfinite(y_tr) & ~is_limit_up_np[ts:te]
                if valid.sum() < 1000:
                    continue

                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_tr[valid], y_tr[valid], feature_name=feature_cols)
                last_train_idx = i

                pct = (i - TRAIN_WINDOW) / (n_dates - TRAIN_WINDOW) * 100
                print(f"   [{cur_date}] 重训 ({pct:.0f}%), 样本: {valid.sum():,}", flush=True)

            if model is None:
                continue

            s, e = date_start[i], date_end[i]
            X_te = X_all_np[s:e]
            codes_te = codes_np[s:e]
            n_stocks = e - s

            if n_stocks < TOP_N:
                continue

            preds = model.predict(X_te)
            cur_date_py = cur_date.astype("datetime64[D]").item()
            score_dates.extend([cur_date_py] * n_stocks)
            score_codes.extend(codes_te.tolist())
            score_values.extend(preds.tolist())

        print(f"\n   ✅ 打分完成: {len(score_values):,} 条记录", flush=True)

        # ── Build raw scores DataFrame ──
        df_scores_raw = pl.DataFrame({
            "date": score_dates,
            "code": score_codes,
            "score": score_values,
        })

        # ── Feature importance (最后一个模型) ──
        if model is not None:
            imp_vals = model.feature_importances_
            imp_max = max(imp_vals) if max(imp_vals) > 0 else 1
            imp_df = pl.DataFrame({
                "factor": feature_cols,
                "importance": imp_vals.tolist(),
            }).sort("importance", descending=True)

            print("\n" + "=" * 55)
            print(f"  LightGBM 特征重要性 (全部 {len(feature_cols)} 个)")
            print("=" * 55)
            for imp_row in imp_df.iter_rows(named=True):
                bar_len = int(imp_row["importance"] / imp_max * 30)
                bar = "█" * bar_len
                print(f"  {imp_row['factor']:<22} {imp_row['importance']:>6} {bar}")
            print("=" * 55)

        train_timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feature_hash = build_feature_hash(feature_cols)
        rotation_train_meta = {
            "strategy": "rotation",
            "label": LABEL,
            "model_name": "lightgbm",
            "feature_mode": FEATURE_MODE,
            "alpha158_group_mode": alpha158_group_config["group_mode_label"]
            if FEATURE_MODE in {"alpha158", "core_plus_alpha158", "all_plus_alpha158"}
            else None,
            "alpha158_analysis_group_mode": ALPHA158_ANALYSIS_GROUP_MODE
            if FEATURE_MODE == "core_plus_alpha158_top1"
            else None,
            "alpha158_top1_factors": alpha158_top1_feature_cols
            if FEATURE_MODE == "core_plus_alpha158_top1"
            else None,
            "normalize_mode": NORMALIZE_MODE,
            "feature_hash": feature_hash,
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "train_timestamp_token": train_timestamp_token,
            "train_run_id": build_rotation_train_run_id(
                LABEL, "lightgbm", train_timestamp_token, feature_hash
            ),
            "trained_at": trained_at,
            "git_commit": get_git_commit(),
            "notebook": "notebooks/cross_section_rotation.py",
            "model_params": lgb_params,
            "train_window": TRAIN_WINDOW,
            "retrain_freq": RETRAIN_FREQ,
            "universe": {
                "mv_min": MV_MIN,
                "mv_max": MV_MAX,
                "min_list_days": MIN_LIST_DAYS,
            },
        }

        return df_scores_raw, rotation_train_meta

    df_scores_raw, rotation_train_meta = run_lgbm_scoring()
    return df_scores_raw, rotation_train_meta


@app.cell
def _(df_scores_raw, pl, q_full, rotation_train_meta):
    # ==============================================================================
    # Cell 6b: 导出 Rotation 分数 → Rust 回测
    #
    # 基于 Cell 6 输出的原始分数 (df_scores_raw), 独立控制导出侧 EMA.
    # 修改 EXPORT_EMA_ALPHA 只需重跑本 Cell, 无需重新训练模型.
    #
    # 关键: Parquet 必须包含所有"曾被评分"股票在整个回测期间的价格数据,
    # 即使某天该股票不在 universe 内 (市值越界等), 也要保留价格行,
    # 否则 Rust 引擎无法对该仓位执行止损/排名退出等检查 → "幽灵仓位"
    # ==============================================================================
    from utils.signal_export import export_rotation_scores

    def run_export():
        EXPORT_TOP_N = 20
        EXPORT_EMA_ALPHA = 0.30  # 导出 parquet 用的分数平滑; 改这里仅需重跑本 Cell

        df_scores_export = df_scores_raw

        if EXPORT_EMA_ALPHA < 1.0:
            df_scores_export = (
                df_scores_raw
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score")
                      .ewm_mean(alpha=EXPORT_EMA_ALPHA)
                      .over("code")
                      .alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"⏳ [Step 6b] 导出分数 EMA 平滑: α={EXPORT_EMA_ALPHA}", flush=True)
        else:
            print("⏳ [Step 6b] 导出使用原始分数 (无 EMA)", flush=True)

        ever_scored_codes = df_scores_export["code"].unique().to_list()
        score_date_min = df_scores_export["date"].min()
        score_date_max = df_scores_export["date"].max()

        print(f"   📦 补全价格: {len(ever_scored_codes)} 只股票, "
              f"{score_date_min} ~ {score_date_max}", flush=True)

        price_cols = ["date", "code", "open_adj", "high_adj", "low_adj",
                      "close_adj", "volume", "market_cap_100m"]
        q_full_cols = q_full.collect_schema().names()
        df_full_prices = (
            q_full
            .filter(pl.col("code").is_in(ever_scored_codes))
            .filter(pl.col("date") >= score_date_min)
            .filter(pl.col("date") <= score_date_max)
            .select([c for c in price_cols if c in q_full_cols])
            .collect()
        )

        df_expanded = df_full_prices.join(
            df_scores_export, on=["date", "code"], how="left"
        ).with_columns(
            pl.col("score").fill_null(-999.0),
        )

        n_scored = df_scores_export.height
        n_total = df_expanded.height
        n_padded = n_total - n_scored
        print(f"   评分行: {n_scored:,}, 补全行: {n_padded:,}, 总计: {n_total:,}", flush=True)

        export_meta = {
            **rotation_train_meta,
            "export_ema_alpha": EXPORT_EMA_ALPHA,
            "export_token": f"e{str(EXPORT_EMA_ALPHA).replace('.', 'p')}_t{EXPORT_TOP_N}",
        }
        scores_path = export_rotation_scores(
            df_expanded,
            top_n=EXPORT_TOP_N,
            raw_scores=df_scores_raw,
            artifact_metadata=export_meta,
        )
        return scores_path

    run_export()
    return


@app.cell
def _(LABEL, df_all, df_scores_raw, go, make_subplots, np, pl, stats):
    # ==============================================================================
    # Cell 7: Signal Quality Analysis — 模型信号统计检验
    #
    # 基于 Cell 6 输出的原始分数 (df_scores_raw), 独立做 EMA 平滑.
    # 修改 EMA_ALPHA 只需重跑本 Cell, 无需重新训练模型.
    # 评估口径拆分:
    #   - 目标标签诊断: 跟随 LABEL, 看模型是否学到训练目标
    #   - 经济效果评估: 固定使用 fwd_ret_1d, 保证不同训练目标之间可比
    #
    # 包含:
    #   7a. Target IC/ICIR + t检验 + 累积IC曲线
    #   7b. Economic Quintile Long-Short 分层收益 (固定 fwd_ret_1d)
    #   7c. Prediction Turnover (Top-20 日间重叠率)
    # ==============================================================================
    def run_signal_quality():
        EMA_ALPHA = 0.3  # Score 时序平滑 (1.0 = 不平滑, 仅影响分析, 不影响训练)
        target_label_col = LABEL
        eval_label_col = "fwd_ret_1d"
        label_cols = ["date", "code", target_label_col]
        if eval_label_col != target_label_col:
            label_cols.append(eval_label_col)

        # ── 合并原始 score 与目标标签/经济评估标签 ──
        df_signal = (
            df_scores_raw
            .select(["date", "code", "score"])
            .join(
                df_all.select(label_cols),
                on=["date", "code"],
                how="inner",
            )
            .filter(
                pl.col(target_label_col).is_not_null()
                & pl.col(target_label_col).is_not_nan()
                & pl.col(eval_label_col).is_not_null()
                & pl.col(eval_label_col).is_not_nan()
            )
            .sort(["date", "code"])
        )

        if EMA_ALPHA < 1.0:
            df_signal = (
                df_signal
                .sort(["code", "date"])
                .with_columns(
                    pl.col("score")
                      .ewm_mean(alpha=EMA_ALPHA)
                      .over("code")
                      .alias("score")
                )
                .sort(["date", "code"])
            )
            print(f"   ⚡ Score EMA 平滑: α={EMA_ALPHA}")
        else:
            print("   ⚡ 无 EMA 平滑 (原始分数)")

        dates_np = df_signal["date"].to_numpy()
        scores_np = df_signal["score"].to_numpy().astype(np.float64)
        target_rets_np = df_signal[target_label_col].to_numpy().astype(np.float64)
        eval_rets_np = df_signal[eval_label_col].to_numpy().astype(np.float64)
        codes_np = df_signal["code"].to_numpy()

        unique_dates = np.unique(dates_np)
        unique_dates.sort()
        n_days = len(unique_dates)
        date_start = np.searchsorted(dates_np, unique_dates, side="left")
        date_end = np.searchsorted(dates_np, unique_dates, side="right")

        print("📊 Signal Quality Analysis")
        print(f"   目标标签: {target_label_col}")
        print(f"   经济评估标签: {eval_label_col}")
        print(f"   样本: {len(dates_np):,} 条, {n_days} 个交易日\n")

        # ================================================================
        # 7a. Target IC / ICIR / t-test
        # ================================================================
        daily_ic = np.full(n_days, np.nan)
        daily_n_stocks = np.zeros(n_days, dtype=int)

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = target_rets_np[s:e]
            mask = np.isfinite(sc) & np.isfinite(rt)
            cnt = mask.sum()
            daily_n_stocks[i] = cnt
            if cnt >= 30:
                ic, _ = stats.spearmanr(sc[mask], rt[mask])
                if np.isfinite(ic):
                    daily_ic[i] = ic

        valid_ic = daily_ic[np.isfinite(daily_ic)]
        ic_mean = float(np.mean(valid_ic))
        ic_std = float(np.std(valid_ic))
        icir = ic_mean / max(ic_std, 1e-8)
        t_stat = ic_mean / max(ic_std, 1e-8) * np.sqrt(len(valid_ic))
        ic_pos_pct = float(np.mean(valid_ic > 0)) * 100
        print("=" * 65)
        print("  7a. Target Label IC Analysis")
        print("=" * 65)
        print(f"  标签列:        {target_label_col}")
        print(f"  IC Mean:       {ic_mean:+.4f}")
        print(f"  IC Std:        {ic_std:.4f}")
        print(f"  ICIR:          {icir:+.4f}")
        print(f"  t-stat:        {t_stat:+.2f}  {'✅ 显著 (>2)' if abs(t_stat) > 2 else '❌ 不显著 (<2)'}")
        print(f"  IC > 0 占比:   {ic_pos_pct:.1f}%")
        print(f"  有效天数:      {len(valid_ic)} / {n_days}")
        print(f"  日均股票数:    {int(np.mean(daily_n_stocks))}")
        print("-" * 65)

        # ================================================================
        # 7b. Economic Quintile Analysis (分层收益)
        # ================================================================
        N_Q = 5
        quintile_daily = {q: [] for q in range(1, N_Q + 1)}
        ls_daily = []

        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            sc = scores_np[s:e]
            rt = eval_rets_np[s:e]
            mask = np.isfinite(sc) & np.isfinite(rt)
            if mask.sum() < N_Q * 10:
                continue

            sc_v = sc[mask]
            rt_v = rt[mask]
            order = np.argsort(-sc_v)
            n = len(order)
            group_size = n // N_Q

            for q in range(N_Q):
                start_idx = q * group_size
                end_idx = (q + 1) * group_size if q < N_Q - 1 else n
                grp_ret = float(np.mean(rt_v[order[start_idx:end_idx]]))
                quintile_daily[q + 1].append(grp_ret)

            q1_ret = float(np.mean(rt_v[order[:group_size]]))
            q5_ret = float(np.mean(rt_v[order[-(n - (N_Q - 1) * group_size):]]))
            ls_daily.append(q1_ret - q5_ret)

        ls_arr = np.array(ls_daily)
        ls_mean = float(np.mean(ls_arr))
        ls_std = float(np.std(ls_arr))
        ls_sharpe = ls_mean / max(ls_std, 1e-8) * np.sqrt(242)
        ls_t = ls_mean / max(ls_std, 1e-8) * np.sqrt(len(ls_arr))
        ls_hit = float(np.mean(ls_arr > 0)) * 100

        print("\n" + "=" * 65)
        print("  7b. Economic Quintile Long-Short Analysis (Q1 做多 - Q5 做空)")
        print("=" * 65)
        print(f"  经济评估标签:  {eval_label_col}")
        for q in range(1, N_Q + 1):
            arr = np.array(quintile_daily[q])
            qm = float(np.mean(arr)) * 100
            print(f"  Q{q} 日均收益: {qm:+.3f}%")
        print("  ---")
        print(f"  L/S 日均收益:  {ls_mean * 100:+.4f}%")
        print(f"  L/S 年化Sharpe: {ls_sharpe:.2f}")
        print(f"  L/S t-stat:    {ls_t:+.2f}  {'✅ 显著' if abs(ls_t) > 2 else '❌ 不显著'}")
        print(f"  L/S 胜率:      {ls_hit:.1f}%")
        cum_ls = np.cumsum(ls_arr)
        ls_dd = float(np.max(np.maximum.accumulate(cum_ls) - cum_ls))
        print(f"  L/S 最大回撤:  {ls_dd * 100:.2f}%")
        print("-" * 65)

        # ================================================================
        # 7c. Prediction Turnover (Top-N 日间重叠率)
        # ================================================================
        for top_n in [20, 50]:
            prev_top = None
            overlaps = []
            for i in range(n_days):
                s, e = date_start[i], date_end[i]
                if e - s < top_n:
                    continue
                sc = scores_np[s:e]
                cd = codes_np[s:e]
                top_idx = np.argsort(-sc)[:top_n]
                top_codes = set(cd[top_idx])
                if prev_top is not None:
                    overlap = len(top_codes & prev_top) / top_n
                    overlaps.append(overlap)
                prev_top = top_codes

            ov_arr = np.array(overlaps)
            est_daily_turnover = (1 - np.mean(ov_arr)) * 2
            print(f"\n  Top-{top_n} 日均重叠率: {np.mean(ov_arr) * 100:.1f}%, "
                  f"日均双边换手: {est_daily_turnover * 100:.1f}%, "
                  f"年化换手: {est_daily_turnover * 242:.0f}x")

        # ================================================================
        # 7d. 分年分析
        # ================================================================
        print("\n" + "=" * 65)
        print("  7d. 分年 Target IC / Economic L-S 统计")
        print("=" * 65)
        print(f"  {'年份':<6} {'IC_mean':>8} {'ICIR':>8} {'t-stat':>8} "
              f"{'L/S日均':>10} {'L/S Sharpe':>10} {'显著?':>6}")
        print("-" * 65)

        ic_dates = unique_dates[np.isfinite(daily_ic)]
        ic_valid_vals = daily_ic[np.isfinite(daily_ic)]

        years = sorted(set(d.astype("datetime64[Y]").item().year
                          for d in ic_dates))

        for yr in years:
            yr_mask_ic = np.array([d.astype("datetime64[Y]").item().year == yr
                                   for d in ic_dates])
            yr_ic = ic_valid_vals[yr_mask_ic]

            if len(yr_ic) < 20:
                continue

            yr_ic_mean = float(np.mean(yr_ic))
            yr_ic_std = float(np.std(yr_ic))
            yr_icir = yr_ic_mean / max(yr_ic_std, 1e-8)
            yr_t = yr_ic_mean / max(yr_ic_std, 1e-8) * np.sqrt(len(yr_ic))

            yr_ls = [ls_daily[j] for j in range(len(ls_daily))
                     if j < len(unique_dates) and
                     unique_dates[j].astype("datetime64[Y]").item().year == yr]
            if yr_ls:
                yr_ls_arr = np.array(yr_ls)
                yr_ls_mean = float(np.mean(yr_ls_arr))
                yr_ls_sharpe = yr_ls_mean / max(float(np.std(yr_ls_arr)), 1e-8) * np.sqrt(242)
            else:
                yr_ls_mean = 0.0
                yr_ls_sharpe = 0.0

            sig = "✅" if abs(yr_t) > 2 else "❌"
            print(f"  {yr:<6} {yr_ic_mean:>+8.4f} {yr_icir:>+8.4f} {yr_t:>+8.2f} "
                  f"{yr_ls_mean * 100:>+10.4f}% {yr_ls_sharpe:>10.2f} {sig:>6}")
        print("-" * 65)

        # ================================================================
        # Visualization
        # ================================================================
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"累积 Target IC 曲线 ({target_label_col})",
                f"分层日均收益 (按 {eval_label_col} 评估)",
                f"L/S 累积收益曲线 ({eval_label_col})",
                "Top-20 日间重叠率 (滚动20日均)",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # 1. Cumulative IC
        valid_dates = unique_dates[np.isfinite(daily_ic)]
        cum_ic_valid = np.nancumsum(daily_ic[np.isfinite(daily_ic)])
        fig.add_trace(go.Scatter(
            x=valid_dates.astype("datetime64[D]").tolist(),
            y=cum_ic_valid.tolist(),
            name="累积Target IC", line=dict(color="#00d4aa"),
        ), row=1, col=1)

        # 2. Quintile bar chart
        q_means = [float(np.mean(quintile_daily[q])) * 100 for q in range(1, N_Q + 1)]
        colors = ["#00d4aa" if m > 0 else "#ff6b6b" for m in q_means]
        fig.add_trace(go.Bar(
            x=[f"Q{q}" for q in range(1, N_Q + 1)],
            y=q_means,
            marker_color=colors,
            name=f"{eval_label_col} 日均收益%",
        ), row=1, col=2)

        # 3. L/S cumulative return
        cum_ls_arr = np.cumsum(ls_arr)
        fig.add_trace(go.Scatter(
            y=cum_ls_arr.tolist(),
            name=f"{eval_label_col} L/S累积收益", line=dict(color="#ffa500"),
        ), row=2, col=1)

        # 4. Top-20 overlap rolling mean
        prev_top = None
        overlap_series = []
        for i in range(n_days):
            s, e = date_start[i], date_end[i]
            if e - s < 20:
                continue
            sc = scores_np[s:e]
            cd = codes_np[s:e]
            top_idx = np.argsort(-sc)[:20]
            top_codes = set(cd[top_idx])
            if prev_top is not None:
                overlap_series.append(len(top_codes & prev_top) / 20)
            prev_top = top_codes

        if len(overlap_series) > 20:
            ov_rolling = np.convolve(overlap_series,
                                     np.ones(20) / 20, mode="valid")
            fig.add_trace(go.Scatter(
                y=(ov_rolling * 100).tolist(),
                name="重叠率%(20日均)", line=dict(color="#9b59b6"),
            ), row=2, col=2)

        fig.update_layout(
            height=700, template="plotly_dark",
            showlegend=False,
            yaxis_title="累积Target IC", yaxis2_title="日均收益(%)",
            yaxis3_title=f"{eval_label_col} 累积L/S收益", yaxis4_title="重叠率(%)",
        )
        fig.show()

        return {
            "target_label": target_label_col,
            "eval_label": eval_label_col,
            "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
            "t_stat": t_stat, "ic_pos_pct": ic_pos_pct,
            "ls_mean": ls_mean, "ls_sharpe": ls_sharpe, "ls_t": ls_t,
            "ls_hit": ls_hit,
        }

    run_signal_quality()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
