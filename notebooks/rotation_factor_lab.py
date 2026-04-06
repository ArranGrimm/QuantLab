import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import numpy as np
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from manifests.rotation_feature_sets import (
        CORE_12_FEATURES,
        list_rotation_feature_sets,
    )
    from utils import add_price_limit_cols, get_st_blacklist_pl, load_daily_data_full
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
        resolve_decay_factor_cols,
        summarize_factor_groups,
    )
    from utils.ic_analysis import (
        calc_factor_corr,
        calc_factor_ic,
        find_redundant_factors,
        print_corr_clusters,
    )
    from utils.rotation_factors import (
        FACTOR_COLS,
        FACTOR_GROUP_LABELS,
        FACTOR_GROUPS,
        FACTOR_TO_GROUP,
        calc_rotation_factors,
        cross_section_normalize,
    )

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    conn = duckdb.connect(DB_PATH, read_only=True)

    MV_MIN = 80
    MV_MAX = 500
    MIN_LIST_DAYS = 60
    START_DATE = "2020-09-01"
    NORMALIZE_MODE = "zscore"
    LABEL = "fwd_ret_1d"
    ALPHA158_ANALYSIS_GROUP_MODE = "all"
    ALPHA_DECAY_SOURCE = "alpha158_top1"
    ALPHA_DECAY_CUSTOM_FACTORS: tuple[str, ...] = ()
    RUN_ROTATION_CORR_DIAGNOSTICS = False
    RUN_ROTATION_CORE_SCREEN = False

    print("🚀 [Lab Step 1] 加载 Rotation Factor Lab 数据底座...")
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
        ALPHA_DECAY_CUSTOM_FACTORS,
        ALPHA158_FACTOR_GROUPS,
        ALPHA158_FACTOR_GROUP_LABELS,
        ALPHA158_FACTOR_TO_GROUP,
        ALPHA_DECAY_SOURCE,
        CORE_12_FEATURES,
        FACTOR_COLS,
        FACTOR_GROUPS,
        FACTOR_GROUP_LABELS,
        FACTOR_TO_GROUP,
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
        calc_factor_corr,
        calc_factor_ic,
        calc_rotation_factors,
        compute_factor_decay,
        cross_section_normalize,
        empty_group_summary_frame,
        find_redundant_factors,
        go,
        list_rotation_feature_sets,
        make_subplots,
        np,
        pl,
        print_corr_clusters,
        px,
        q_full,
        resolve_alpha158_group_config,
        summarize_factor_groups,
    )


@app.cell
def _(list_rotation_feature_sets):
    print("🗂️ [Manifest Snapshot] 当前 Rotation 特征集清单")
    for spec in list_rotation_feature_sets(include_unselectable=True):
        selectable = "训练可用" if spec.selectable else "仅分析"
        print(
            f"  - {spec.name:<30} [{spec.status:<12}] "
            f"{selectable:<8} {spec.feature_count:>3} 因子"
        )
    return


@app.cell
def _(
    ALPHA158_ANALYSIS_GROUP_MODE,
    ALPHA158_FACTOR_GROUPS,
    ALPHA158_FACTOR_GROUP_LABELS,
    ALPHA158_FACTOR_TO_GROUP,
    FACTOR_COLS,
    LABEL,
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
    print("⏳ [Lab Step 2] 计算分析特征矩阵...")

    df_factors = calc_rotation_factors(q_full)
    active_factor_cols = list(FACTOR_COLS)

    analysis_mode = str(ALPHA158_ANALYSIS_GROUP_MODE).strip().lower()
    if analysis_mode not in {"", "none", "disabled"}:
        alpha_analysis_config = resolve_alpha158_group_config(ALPHA158_ANALYSIS_GROUP_MODE)
        alpha_factor_cols = list(alpha_analysis_config["factor_cols"])
        print(f"   Alpha158 分析分组: {alpha_analysis_config['group_mode_label']}")
        print(f"   计算 Alpha158 因子: {len(alpha_factor_cols)} 个")
        df_factors = calc_alpha158_factors(
            df_factors,
            use_kbar=bool(alpha_analysis_config["use_kbar"]),
            price_fields=alpha_analysis_config["price_fields"],
            include=alpha_analysis_config["include_ops"],
        )
        active_factor_cols.extend(alpha_factor_cols)

    df_with_label = df_factors.with_columns([
        (pl.col("close_adj").shift(-1).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_1d"),
        (pl.col("close_adj").shift(-2).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_2d"),
        (pl.col("close_adj").shift(-3).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_3d"),
        (pl.col("close_adj").shift(-5).over("code") / pl.col("close_adj") - 1).alias("fwd_ret_5d"),
    ])
    df_with_label = add_price_limit_cols(df_with_label)

    df_universe = (
        df_with_label
        .with_columns(pl.col("date").cum_count().over("code").alias("_list_days"))
        .filter(
            (pl.col("_list_days") >= MIN_LIST_DAYS)
            & (pl.col("market_cap_100m") >= MV_MIN)
            & (pl.col("market_cap_100m") <= MV_MAX)
        )
    )

    df_normalized = cross_section_normalize(
        df_universe,
        active_factor_cols,
        mode=NORMALIZE_MODE,
    )
    df_normalized = df_normalized.with_columns([
        (pl.col(label_name) - pl.col(label_name).mean().over("date")).alias(f"{label_name}_excess")
        for label_name in ("fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d")
    ])

    final_cols = [
        "code",
        "date",
        "close_adj",
        "close_raw",
        "vwap_raw",
        "amount",
        "volume",
        "market_cap_100m",
        "circulating_capital",
        "fwd_ret_1d",
        "fwd_ret_2d",
        "fwd_ret_3d",
        "fwd_ret_5d",
        "fwd_ret_1d_excess",
        "fwd_ret_2d_excess",
        "fwd_ret_3d_excess",
        "fwd_ret_5d_excess",
        "is_limit_up",
        "is_limit_down",
        *active_factor_cols,
    ]
    df_all = df_normalized.select(final_cols).collect()

    print(f"✅ Factor Lab 数据集: {df_all.shape[0]:,} 行 x {df_all.shape[1]} 列")
    return (df_all,)


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
    calc_factor_ic,
    df_all,
    empty_group_summary_frame,
    go,
    make_subplots,
    pl,
    resolve_alpha158_group_config,
    summarize_factor_groups,
):
    df_valid = df_all.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_not_nan())

    available_rotation_factors = [factor_name for factor_name in FACTOR_COLS if factor_name in df_all.columns]
    if available_rotation_factors:
        rotation_ic_results = calc_factor_ic(
            df_valid,
            factor_cols=available_rotation_factors,
            label=LABEL,
            min_samples=30,
        )
        rotation_ic_summary = build_ic_summary_frame(rotation_ic_results)
        rotation_daily_ic = build_daily_ic_frame(
            df_valid,
            factor_cols=available_rotation_factors,
            label=LABEL,
            min_samples=30,
        )
    else:
        rotation_ic_results = {}
        rotation_ic_summary = build_ic_summary_frame({})
        rotation_daily_ic = build_daily_ic_frame(df_valid, factor_cols=[], label=LABEL)

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
        alpha158_ic_results = calc_factor_ic(
            df_valid,
            factor_cols=available_alpha158_factors,
            label=LABEL,
            min_samples=30,
        )
        alpha158_ic_summary = build_ic_summary_frame(alpha158_ic_results)
        df_alpha158_group_summary = summarize_factor_groups(
            alpha158_ic_results,
            alpha_factor_groups,
            ALPHA158_FACTOR_GROUP_LABELS,
        )
    else:
        alpha158_ic_results = {}
        alpha158_ic_summary = build_ic_summary_frame({})
        df_alpha158_group_summary = empty_group_summary_frame()

    df_alpha158_top1 = (
        df_alpha158_group_summary
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
    alpha158_top1_factor_cols = df_alpha158_top1["top_factor"].to_list()

    if rotation_ic_summary.height > 0 and rotation_daily_ic.height > 0:
        top_factors = rotation_ic_summary["factor"].head(6).to_list()
        fig_ic = make_subplots(rows=1, cols=1)
        for factor_name in top_factors:
            ic_cum = rotation_daily_ic.select(["date", factor_name]).drop_nulls().sort("date")
            fig_ic.add_trace(
                go.Scatter(
                    x=ic_cum["date"].to_list(),
                    y=ic_cum[factor_name].cum_sum().to_list(),
                    name=factor_name,
                    mode="lines",
                )
            )

        fig_ic.update_layout(
            title="Rotation Top 6 因子 — IC 累积曲线",
            xaxis_title="日期",
            yaxis_title="累积 IC",
            height=500,
            template="plotly_dark",
        )
        fig_ic.show()

    df_group_summary = summarize_factor_groups(
        rotation_ic_results,
        FACTOR_GROUPS,
        FACTOR_GROUP_LABELS,
    )

    ic_results = rotation_ic_results
    return (
        alpha158_ic_summary,
        alpha158_top1_factor_cols,
        df_alpha158_group_summary,
        df_alpha158_top1,
        df_group_summary,
        ic_results,
        rotation_ic_summary,
    )


@app.cell
def _(df_alpha158_group_summary, df_alpha158_top1, df_group_summary):
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
        print(f"  {'分组':<20} {'top1因子':<24} {'IC Mean':>10} {'ICIR':>10} {'|ICIR|':>10}")
        print("-" * 96)
        for row in df_alpha158_top1.iter_rows(named=True):
            print(
                f"  {row['group_name']:<20} {row['top_factor']:<24} "
                f"{row['top_ic_mean']:>10.4f} {row['top_icir']:>10.4f} {row['top_abs_icir']:>10.4f}"
            )
        print("-" * 96)
    return


@app.cell
def _(
    ALPHA_DECAY_CUSTOM_FACTORS,
    ALPHA_DECAY_SOURCE,
    compute_factor_decay,
    df_all,
    df_alpha158_top1,
    go,
    make_subplots,
    resolve_decay_factor_cols,
    rotation_ic_summary,
):
    top_factors = resolve_decay_factor_cols(
        ALPHA_DECAY_SOURCE,
        rotation_ic_summary=rotation_ic_summary,
        alpha158_top1=df_alpha158_top1,
        custom_factor_cols=ALPHA_DECAY_CUSTOM_FACTORS,
        rotation_top_n=15,
    )
    top_factors = [factor for factor in top_factors if factor in df_all.columns]
    if not top_factors:
        print("ℹ️ [Alpha Decay] 当前无可用因子。")
        return

    decay_summary, avg_icir = compute_factor_decay(
        df_all,
        factor_cols=top_factors,
    )

    horizons = ["fwd_ret_1d", "fwd_ret_2d", "fwd_ret_3d", "fwd_ret_5d"]
    h_days = [1, 2, 3, 5]
    print("📉 [Alpha Decay] 因子 IC 衰减对比")
    for factor_name in top_factors:
        row = [factor_name]
        for horizon in horizons:
            dd = decay_summary[horizon][factor_name]
            row.append(f"{dd['ic_mean']:+.4f}/{dd['icir']:+.4f}")
        print("  " + " | ".join(row))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Top 因子 |ICIR| 衰减", "平均 |ICIR| 衰减"],
    )
    for factor_name in top_factors[:8]:
        y_vals = [abs(decay_summary[horizon][factor_name]["icir"]) for horizon in horizons]
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

    avg_y = [avg_icir.get(d, 0.0) for d in h_days]
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
    return


@app.cell
def _(
    FACTOR_COLS,
    FACTOR_GROUPS,
    FACTOR_GROUP_LABELS,
    FACTOR_TO_GROUP,
    CORE_12_FEATURES,
    RUN_ROTATION_CORE_SCREEN,
    RUN_ROTATION_CORR_DIAGNOSTICS,
    calc_factor_corr,
    df_all,
    find_redundant_factors,
    ic_results,
    print_corr_clusters,
    px,
):
    available_rotation_factors = [factor_name for factor_name in FACTOR_COLS if factor_name in df_all.columns]
    factors_keep = list(available_rotation_factors)

    should_run_corr = RUN_ROTATION_CORR_DIAGNOSTICS
    if should_run_corr and len(available_rotation_factors) >= 2:
        corr_mat, factor_names = calc_factor_corr(df_all, available_rotation_factors)
        print_corr_clusters(corr_mat, factor_names, threshold=0.7)
        factors_keep, factors_drop, _ = find_redundant_factors(
            corr_mat,
            factor_names,
            ic_results=ic_results,
            threshold=0.85,
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
    else:
        print("ℹ️ [Rotation Corr] 默认不跑相关性剪枝，可按需手动打开。")

    if not RUN_ROTATION_CORE_SCREEN:
        print("ℹ️ [Core Screen] 默认展示冻结的 core_12。")
        print(f"   core_12 = {', '.join(CORE_12_FEATURES)}")
        return

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
    for group_key_local in FACTOR_GROUPS:
        group_rows = [r for r in ranked_rows if r["group_key"] == group_key_local]
        group_rows.sort(key=lambda r: r["abs_icir"], reverse=True)
        if not group_rows:
            continue

        kept_rows = [r for r in group_rows if r["is_pruned_keep"]]
        primary = kept_rows[0] if kept_rows else group_rows[0]
        core_primary.append(primary["factor"])
        print(
            f"  {FACTOR_GROUP_LABELS.get(group_key_local, group_key_local):<20} "
            f"{primary['factor']:<24} {primary['abs_icir']:>10.4f}"
        )

        follow_rows = kept_rows[1:] if primary["is_pruned_keep"] else kept_rows
        for candidate_row in follow_rows:
            if (
                candidate_row["abs_icir"] >= 0.08
                and candidate_row["abs_icir"] >= primary["abs_icir"] * 0.60
            ):
                secondary_pool.append(candidate_row)

    secondary_pool.sort(key=lambda r: r["abs_icir"], reverse=True)
    core_factors = list(core_primary)
    for candidate_row in secondary_pool:
        if candidate_row["factor"] in core_factors:
            continue
        if len(core_factors) >= 12:
            break
        core_factors.append(candidate_row["factor"])
    print(f"  建议 core feature set ({len(core_factors)}): {', '.join(core_factors)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
