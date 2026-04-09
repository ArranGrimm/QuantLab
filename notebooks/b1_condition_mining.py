import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import numpy as np
    import polars as pl

    from utils import (
        B1_MINING_CORE_FEATURE_COLS,
        B1_MINING_FEATURE_COLS,
        B1_MINING_SECOND_BATCH_FEATURE_COLS,
        build_b1_research_frame,
        get_st_blacklist_pl,
        load_daily_data_full,
    )

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2019-01-01"
    END_DATE = "2026-03-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0

    LABEL_COL = "fwd_mfe_10d"
    POSITIVE_MFE_THRESHOLD = 0.08
    N_BINS = 5
    ACTIVE_SEED_COL = "seed_mid"
    USE_BULL_ONLY = True

    LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),
        ("2019-12-16", "2020-03-02"),
        ("2020-06-19", "2020-07-15"),
        ("2020-12-24", "2021-01-25"),
        ("2021-04-16", "2021-09-14"),
        ("2022-04-27", "2022-07-05"),
        ("2023-01-15", "2023-04-15"),
        ("2024-02-06", "2024-03-20"),
        ("2024-09-24", "2024-10-15"),
        ("2025-04-09", "2025-09-04"),
        ("2026-01-05", "2026-02-02"),
    ]

    CORE_FEATURE_COLS = list(B1_MINING_CORE_FEATURE_COLS)
    SECOND_BATCH_FEATURE_COLS = list(B1_MINING_SECOND_BATCH_FEATURE_COLS)
    FEATURE_COLS = list(B1_MINING_FEATURE_COLS)
    return (
        ACTIVE_SEED_COL,
        CORE_FEATURE_COLS,
        DB_PATH,
        END_DATE,
        FEATURE_COLS,
        LABEL_COL,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        N_BINS,
        POSITIVE_MFE_THRESHOLD,
        SECOND_BATCH_FEATURE_COLS,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        USE_BULL_ONLY,
        build_b1_research_frame,
        duckdb,
        get_st_blacklist_pl,
        load_daily_data_full,
        np,
        pl,
    )


@app.cell
def _(
    ACTIVE_SEED_COL,
    CORE_FEATURE_COLS,
    FEATURE_COLS,
    LABEL_COL,
    N_BINS,
    POSITIVE_MFE_THRESHOLD,
    SECOND_BATCH_FEATURE_COLS,
    SEED_J_MAX,
    USE_BULL_ONLY,
):
    print("=" * 72)
    print("  B1 条件挖掘面板")
    print("=" * 72)
    print("  这本 notebook 只回答 3 个问题:")
    print("  1. 三档 seed pool 里，哪一档更适合做第一轮条件挖掘")
    print("  2. 哪些连续特征单独看就已经有明显增量")
    print("  3. 能不能提炼出几条值得继续验证的简单规则")
    print("")
    print("  建议阅读顺序: Step 3 -> Step 4 -> Step 5 -> Step 6 -> Step 7 -> Step 7b -> Step 8")
    print("  当前配置:")
    print(f"    active_seed:        {ACTIVE_SEED_COL}")
    print(f"    bull_regime_only:   {USE_BULL_ONLY}")
    print(f"    label_col:          {LABEL_COL}")
    print(f"    positive_threshold: {POSITIVE_MFE_THRESHOLD:.0%}")
    print(f"    n_bins:             {N_BINS}")
    print(f"    seed_j_max:         {SEED_J_MAX}")
    print(f"    feature_count:      {len(FEATURE_COLS)}")
    print(f"    core_feature_count: {len(CORE_FEATURE_COLS)}")
    print(f"    batch2_feature_cnt: {len(SECOND_BATCH_FEATURE_COLS)}")
    print("    review_feature:     在 Step 6 cell 内直接修改")
    print("")
    print("  快速术语:")
    print("    mfe10_mean   : 未来 10 日最大上涨空间均值，越高越好")
    print(f"    mfe_hit_rate : fwd_mfe_10d >= {POSITIVE_MFE_THRESHOLD:.0%} 的占比")
    print("    best_bin_lift: 最优分箱相对整体样本的增量")
    print("    monotonicity : 分箱从低到高是否更单调地变好，越接近 1 越顺")
    print("=" * 72)
    return


@app.cell
def _(
    DB_PATH,
    END_DATE,
    START_DATE,
    ST_SNAPSHOT_DATE,
    duckdb,
    get_st_blacklist_pl,
    load_daily_data_full,
    pl,
):
    _conn = duckdb.connect(DB_PATH, read_only=True)
    st_blacklist = get_st_blacklist_pl(ST_SNAPSHOT_DATE)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()

    q_full = (
        load_daily_data_full(_conn)
        .join(st_blacklist_df, on="code", how="anti")
        .filter(
            pl.col("date").is_between(
                pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
                pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"),
            )
        )
    )

    data_load_summary = pl.DataFrame(
        [
            {"item": "date_range", "value": f"{START_DATE} ~ {END_DATE}"},
            {"item": "st_snapshot_date", "value": ST_SNAPSHOT_DATE},
            {"item": "st_excluded_count", "value": str(len(st_blacklist))},
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 1. 数据范围与 ST 过滤快照")
    print("=" * 72)
    print(data_load_summary)
    return (q_full,)


@app.cell
def _(
    FEATURE_COLS,
    LABEL_COL,
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    SEED_J_MAX,
    build_b1_research_frame,
    pl,
    q_full,
):
    df_all = build_b1_research_frame(
        q_full,
        mv_min=MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
    )

    valid_feature_cols = [col for col in FEATURE_COLS if col in df_all.columns]
    dataset_summary = pl.DataFrame(
        [
            {"item": "rows", "value": f"{df_all.shape[0]:,}"},
            {"item": "columns", "value": str(df_all.shape[1])},
            {"item": "date_min", "value": str(df_all["date"].min())},
            {"item": "date_max", "value": str(df_all["date"].max())},
            {"item": "n_codes", "value": str(df_all["code"].n_unique())},
            {"item": "feature_count", "value": str(len(valid_feature_cols))},
            {"item": "label_col", "value": LABEL_COL},
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 2. B1 主链因子与研究特征")
    print("=" * 72)
    print(dataset_summary)
    print("\n  当前特征池:")
    print(", ".join(valid_feature_cols))
    return df_all, valid_feature_cols


@app.cell
def _(POSITIVE_MFE_THRESHOLD, df_all, pl):
    _seed_cols = ["seed_loose", "seed_mid", "seed_strict"]
    _scope_defs = [
        ("all", pl.lit(True)),
        ("bull_only", pl.col("is_manual_bull")),
    ]

    _summary_rows = []
    for _scope_name, _scope_expr in _scope_defs:
        _df_scope = df_all.filter(_scope_expr)
        for _seed_col in _seed_cols:
            _df_seed = _df_scope.filter(pl.col(_seed_col))
            if _df_seed.is_empty():
                continue

            _date_count = max(_df_seed["date"].n_unique(), 1)
            _summary_rows.append(
                {
                    "scope": _scope_name,
                    "seed": _seed_col,
                    "rows": _df_seed.height,
                    "dates": _date_count,
                    "avg_candidates_per_day": _df_seed.height / _date_count,
                    "mfe10_mean": float(_df_seed["fwd_mfe_10d"].mean()),
                    "mae10_mean": float(_df_seed["fwd_mae_10d"].mean()),
                    "ret1_mean": float(_df_seed["fwd_ret_1d"].mean()),
                    "mfe_hit_rate": float((_df_seed["fwd_mfe_10d"] >= POSITIVE_MFE_THRESHOLD).mean()),
                    "ret1_win_rate": float((_df_seed["fwd_ret_1d"] > 0).mean()),
                }
            )

    if _summary_rows:
        seed_summary = pl.DataFrame(_summary_rows).sort(["scope", "seed"])
    else:
        seed_summary = pl.DataFrame(
            schema={
                "scope": pl.Utf8,
                "seed": pl.Utf8,
                "rows": pl.Int64,
                "dates": pl.Int64,
                "avg_candidates_per_day": pl.Float64,
                "mfe10_mean": pl.Float64,
                "mae10_mean": pl.Float64,
                "ret1_mean": pl.Float64,
                "mfe_hit_rate": pl.Float64,
                "ret1_win_rate": pl.Float64,
            }
        )
    seed_summary_view = seed_summary.with_columns(
        [
            pl.col("avg_candidates_per_day").round(2),
            pl.col("mfe10_mean").round(4),
            pl.col("mae10_mean").round(4),
            pl.col("ret1_mean").round(4),
            pl.col("mfe_hit_rate").round(4),
            pl.col("ret1_win_rate").round(4),
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 3. Seed Pool 概览")
    print("=" * 72)
    print(seed_summary_view)
    if seed_summary.height == 0:
        print("  结论: 当前没有可用 seed 样本。")
    else:
        _seed_focus = seed_summary.filter(pl.col("scope") == "bull_only")
        if _seed_focus.height == 0:
            _seed_focus = seed_summary
        _best_seed_row = _seed_focus.sort(["mfe_hit_rate", "mfe10_mean"], descending=[True, True]).row(0, named=True)
        print(
            "  结论: 当前最值得优先看的 seed 是 "
            f"`{_best_seed_row['seed']}` ({_best_seed_row['scope']})，"
            f"mfe_hit_rate={_best_seed_row['mfe_hit_rate']:.2%}, "
            f"mfe10_mean={_best_seed_row['mfe10_mean']:+.4f}"
        )
    return


@app.cell
def _(
    ACTIVE_SEED_COL,
    LABEL_COL,
    POSITIVE_MFE_THRESHOLD,
    USE_BULL_ONLY,
    df_all,
    pl,
):
    _mining_mask = pl.col(ACTIVE_SEED_COL)
    if USE_BULL_ONLY:
        _mining_mask = _mining_mask & pl.col("is_manual_bull")

    df_mining = df_all.filter(
        _mining_mask & pl.col(LABEL_COL).is_not_null() & pl.col(LABEL_COL).is_not_nan()
    )

    if df_mining.is_empty():
        sample_stats = {
            "rows": 0,
            "dates": 0,
            "codes": 0,
            "label_mean": 0.0,
            "positive_rate": 0.0,
        }
    else:
        sample_stats = {
            "rows": df_mining.height,
            "dates": df_mining["date"].n_unique(),
            "codes": df_mining["code"].n_unique(),
            "label_mean": float(df_mining[LABEL_COL].mean()),
            "positive_rate": float((df_mining[LABEL_COL] >= POSITIVE_MFE_THRESHOLD).mean()),
        }

    mining_sample_view = pl.DataFrame(
        [
            {"item": "active_seed", "value": ACTIVE_SEED_COL},
            {"item": "bull_only", "value": str(USE_BULL_ONLY)},
            {"item": "rows", "value": f"{sample_stats['rows']:,}"},
            {"item": "dates", "value": str(sample_stats["dates"])},
            {"item": "codes", "value": str(sample_stats["codes"])},
            {"item": "label_mean", "value": f"{sample_stats['label_mean']:+.4f}"},
            {"item": "positive_rate", "value": f"{sample_stats['positive_rate']:.2%}"},
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 4. 当前挖掘样本")
    print("=" * 72)
    print(mining_sample_view)
    return (df_mining,)


@app.cell
def _(
    LABEL_COL,
    N_BINS,
    POSITIVE_MFE_THRESHOLD,
    df_mining,
    np,
    pl,
    valid_feature_cols,
):
    def _build_bin_edges(values, n_bins):
        _edges_local = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        return np.unique(_edges_local)

    _feature_rows = []
    base_mean = float(df_mining[LABEL_COL].mean()) if not df_mining.is_empty() else 0.0
    base_hit = float((df_mining[LABEL_COL] >= POSITIVE_MFE_THRESHOLD).mean()) if not df_mining.is_empty() else 0.0

    for _feature in valid_feature_cols:
        _df_feature = (
            df_mining.select([_feature, LABEL_COL])
            .filter(pl.col(_feature).is_not_null() & pl.col(LABEL_COL).is_not_null())
        )
        if _df_feature.height < 200:
            continue

        _x_vals = _df_feature[_feature].to_numpy().astype(np.float64)
        _y_vals = _df_feature[LABEL_COL].to_numpy().astype(np.float64)

        if not np.isfinite(_x_vals).all() or np.nanstd(_x_vals) < 1e-12:
            continue

        _edges = _build_bin_edges(_x_vals, N_BINS)
        if len(_edges) < 3:
            continue

        _bin_ids = np.digitize(_x_vals, _edges[1:-1], right=True)
        _bin_stats = []
        for _bin_idx in range(len(_edges) - 1):
            _mask = _bin_ids == _bin_idx
            _count = int(_mask.sum())
            if _count < 30:
                continue
            _bin_stats.append(
                {
                    "bin_idx": _bin_idx + 1,
                    "count": _count,
                    "mean_label": float(np.mean(_y_vals[_mask])),
                    "hit_rate": float(np.mean(_y_vals[_mask] >= POSITIVE_MFE_THRESHOLD)),
                    "left_edge": float(_edges[_bin_idx]),
                    "right_edge": float(_edges[_bin_idx + 1]),
                }
            )

        if len(_bin_stats) < 2:
            continue

        _best_bin = max(_bin_stats, key=lambda item: item["mean_label"])
        _worst_bin = min(_bin_stats, key=lambda item: item["mean_label"])
        _mean_series = np.array([item["mean_label"] for item in _bin_stats], dtype=np.float64)
        _order_series = np.arange(1, len(_bin_stats) + 1, dtype=np.float64)
        _monotonicity = (
            float(np.corrcoef(_order_series, _mean_series)[0, 1]) if len(_bin_stats) >= 3 else 0.0
        )

        _feature_rows.append(
            {
                "feature": _feature,
                "base_mean": base_mean,
                "base_hit_rate": base_hit,
                "best_bin_mean": _best_bin["mean_label"],
                "best_bin_hit_rate": _best_bin["hit_rate"],
                "best_bin_range": f"[{_best_bin['left_edge']:.4f}, {_best_bin['right_edge']:.4f}]",
                "best_bin_lift": _best_bin["mean_label"] - base_mean,
                "worst_bin_mean": _worst_bin["mean_label"],
                "worst_bin_range": f"[{_worst_bin['left_edge']:.4f}, {_worst_bin['right_edge']:.4f}]",
                "worst_bin_lift": _worst_bin["mean_label"] - base_mean,
                "monotonicity": _monotonicity,
            }
        )

    if _feature_rows:
        feature_scoreboard = pl.DataFrame(_feature_rows).sort(
            ["best_bin_lift", "monotonicity"], descending=[True, True]
        )
    else:
        feature_scoreboard = pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "base_mean": pl.Float64,
                "base_hit_rate": pl.Float64,
                "best_bin_mean": pl.Float64,
                "best_bin_hit_rate": pl.Float64,
                "best_bin_range": pl.Utf8,
                "best_bin_lift": pl.Float64,
                "worst_bin_mean": pl.Float64,
                "worst_bin_range": pl.Utf8,
                "worst_bin_lift": pl.Float64,
                "monotonicity": pl.Float64,
            }
        )
    feature_scoreboard_view = feature_scoreboard.select(
        [
            pl.col("feature"),
            pl.col("best_bin_range"),
            pl.col("best_bin_mean").round(4).alias("best_mfe10_mean"),
            pl.col("best_bin_hit_rate").round(4).alias("best_hit_rate"),
            pl.col("best_bin_lift").round(4).alias("lift_vs_base"),
            pl.col("worst_bin_range"),
            pl.col("worst_bin_lift").round(4),
            pl.col("monotonicity").round(4),
        ]
    )
    print("\n" + "=" * 72)
    print("  Step 5. 单变量分箱得分榜")
    print("=" * 72)
    print(f"  基线命中率: {base_hit:.2%}")
    if feature_scoreboard.height == 0:
        print("  结论: 当前没有足够样本生成特征得分榜。")
    else:
        print(feature_scoreboard_view)
        _top_rows = feature_scoreboard.head(3).iter_rows(named=True)
        print("  结论:")
        for _row in _top_rows:
            print(
                f"    - {_row['feature']}: best_range={_row['best_bin_range']}, "
                f"lift={_row['best_bin_lift']:+.4f}, monotonicity={_row['monotonicity']:+.4f}"
            )
    return base_hit, feature_scoreboard


@app.cell
def _(
    LABEL_COL,
    N_BINS,
    POSITIVE_MFE_THRESHOLD,
    df_mining,
    feature_scoreboard,
    np,
    pl,
):
    # Bias_WL_YL
    # rw_dif_pct
    # Bias_C_YL

    review_feature = "Bias_WL_YL"
    if feature_scoreboard.height > 0 and review_feature not in feature_scoreboard["feature"].to_list():
        review_feature = feature_scoreboard["feature"][0]

    _df_review = (
        df_mining.select([review_feature, LABEL_COL])
        .filter(pl.col(review_feature).is_not_null() & pl.col(LABEL_COL).is_not_null())
    )

    _review_rows = []
    if _df_review.height >= 200:
        _x_review = _df_review[review_feature].to_numpy().astype(np.float64)
        _y_review = _df_review[LABEL_COL].to_numpy().astype(np.float64)
        _edges_review = np.unique(np.quantile(_x_review, np.linspace(0, 1, N_BINS + 1)))
        if len(_edges_review) >= 3:
            _bin_ids_review = np.digitize(_x_review, _edges_review[1:-1], right=True)
            for _bin_idx_review in range(len(_edges_review) - 1):
                _mask_review = _bin_ids_review == _bin_idx_review
                _count_review = int(_mask_review.sum())
                if _count_review < 30:
                    continue
                _review_rows.append(
                    {
                        "feature": review_feature,
                        "bin": _bin_idx_review + 1,
                        "range": f"[{_edges_review[_bin_idx_review]:.4f}, {_edges_review[_bin_idx_review + 1]:.4f}]",
                        "count": _count_review,
                        "mfe10_mean": float(np.mean(_y_review[_mask_review])),
                        "mfe_hit_rate": float(np.mean(_y_review[_mask_review] >= POSITIVE_MFE_THRESHOLD)),
                    }
                )

    if _review_rows:
        review_table = pl.DataFrame(_review_rows)
    else:
        review_table = pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "bin": pl.Int64,
                "range": pl.Utf8,
                "count": pl.Int64,
                "mfe10_mean": pl.Float64,
                "mfe_hit_rate": pl.Float64,
            }
        )
    review_table_view = review_table.with_columns(
        [
            pl.col("mfe10_mean").round(4),
            pl.col("mfe_hit_rate").round(4),
        ]
    )
    print("\n" + "=" * 72)
    print(f"  Step 6. 特征深挖: {review_feature}")
    print("=" * 72)
    if review_table_view.height == 0:
        print("  结论: 当前样本不足，暂时无法对这个特征做稳定分箱。")
    else:
        print(review_table_view)
        _best_review_row = review_table.sort("mfe10_mean", descending=True).row(0, named=True)
        print(
            "  结论: 当前最优分箱为 "
            f"{_best_review_row['range']}, "
            f"mfe10_mean={_best_review_row['mfe10_mean']:+.4f}, "
            f"mfe_hit_rate={_best_review_row['mfe_hit_rate']:.2%}"
        )
    return (review_feature,)


@app.cell
def _(
    LABEL_COL,
    POSITIVE_MFE_THRESHOLD,
    base_hit,
    df_mining,
    feature_scoreboard,
    np,
    pl,
):
    candidate_rules = pl.DataFrame(
        schema={
            "rule": pl.Utf8,
            "samples": pl.Int64,
            "positive_rate": pl.Float64,
            "lift_vs_base": pl.Float64,
        }
    )
    tree_text = "skipped"
    print("\n" + "=" * 72)
    print("  Step 7. 浅树候选规则")
    print("=" * 72)

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception as exc:
        print(f"⚠️ [Step 7] sklearn 不可用，跳过浅树规则提取: {exc}")
    else:
        _tree_features = feature_scoreboard["feature"].head(6).to_list()
        if not _tree_features:
            print("⚠️ [Step 7] 没有足够的候选特征，跳过浅树规则提取")
        else:
            _df_tree = (
                df_mining.select(_tree_features + [LABEL_COL])
                .filter(pl.all_horizontal(*[pl.col(col).is_not_null() for col in _tree_features]))
                .filter(pl.col(LABEL_COL).is_not_null())
            )

            if _df_tree.height < 300:
                print("⚠️ [Step 7] 样本不足，跳过浅树规则提取")
            else:
                _x_tree = _df_tree.select(_tree_features).to_numpy().astype(np.float64)
                _y_tree = (_df_tree[LABEL_COL].to_numpy().astype(np.float64) >= POSITIVE_MFE_THRESHOLD).astype(
                    np.int32
                )

                if len(np.unique(_y_tree)) < 2:
                    print("⚠️ [Step 7] 标签只有单一类别，跳过浅树规则提取")
                else:
                    _tree_model = DecisionTreeClassifier(
                        max_depth=2,
                        min_samples_leaf=max(50, _df_tree.height // 20),
                        random_state=42,
                    )
                    _tree_model.fit(_x_tree, _y_tree)

                    tree_text = export_text(_tree_model, feature_names=_tree_features, decimals=4)

                    _rule_rows = []
                    _tree_state = _tree_model.tree_

                    def _walk(node_id, clauses):
                        _left_id = _tree_state.children_left[node_id]
                        _right_id = _tree_state.children_right[node_id]
                        _is_leaf = _left_id == _right_id

                        if _is_leaf:
                            _sample_count = int(_tree_state.n_node_samples[node_id])
                            _values = _tree_state.value[node_id][0]
                            _positive_rate = float(_values[1] / _values.sum())
                            if _sample_count >= 50 and _positive_rate > base_hit + 0.03:
                                _rule_rows.append(
                                    {
                                        "rule": " & ".join(clauses) if clauses else "ALL",
                                        "samples": _sample_count,
                                        "positive_rate": _positive_rate,
                                        "lift_vs_base": _positive_rate - base_hit,
                                    }
                                )
                            return

                        _feature_name = _tree_features[_tree_state.feature[node_id]]
                        _threshold = float(_tree_state.threshold[node_id])
                        _walk(_left_id, clauses + [f"{_feature_name} <= {_threshold:.4f}"])
                        _walk(_right_id, clauses + [f"{_feature_name} > {_threshold:.4f}"])

                    _walk(0, [])

                    if _rule_rows:
                        candidate_rules = pl.DataFrame(_rule_rows).sort("lift_vs_base", descending=True)
    candidate_rules_view = candidate_rules.with_columns(
        [
            pl.col("positive_rate").round(4),
            pl.col("lift_vs_base").round(4),
        ]
    )
    if tree_text != "skipped":
        print("  浅树文本:")
        print(tree_text)
    if candidate_rules.height == 0:
        print("  结论: 当前浅树没有提炼出明显优于基线的候选规则。")
    else:
        print(candidate_rules_view)
        _best_rule = candidate_rules.row(0, named=True)
        print(
            "  结论: 第一候选规则为 "
            f"{_best_rule['rule']}, "
            f"samples={_best_rule['samples']}, "
            f"positive_rate={_best_rule['positive_rate']:.2%}, "
            f"lift_vs_base={_best_rule['lift_vs_base']:+.4f}"
        )
    return (candidate_rules,)


@app.cell
def _(LABEL_COL, POSITIVE_MFE_THRESHOLD, df_mining, pl):
    print("\n" + "=" * 72)
    print("  Step 7b. 手工候选规则验证")
    print("=" * 72)

    _rule_defs = [
        (
            "Bias_WL_YL > 9 & rw_dif_pct > 8.5 & Bias_C_YL > 7.65",
            (pl.col("Bias_WL_YL") > 9) & (pl.col("rw_dif_pct") > 8.5) & (pl.col("Bias_C_YL") > 7.65),
        ),
        (
            "Bias_WL_YL > 9 & rw_dif_pct > 9 & Bias_C_YL > 7.65",
            (pl.col("Bias_WL_YL") > 9) & (pl.col("rw_dif_pct") > 9) & (pl.col("Bias_C_YL") > 7.65),
        ),
        (
            "Bias_WL_YL > 9 & rw_dif_pct > 10 & Bias_C_YL > 7.65",
            (pl.col("Bias_WL_YL") > 9) & (pl.col("rw_dif_pct") > 10) & (pl.col("Bias_C_YL") > 7.65),
        ),
    ]

    manual_rule_table = pl.DataFrame(
        schema={
            "rule": pl.Utf8,
            "samples": pl.Int64,
            "dates": pl.Int64,
            "mfe10_mean": pl.Float64,
            "positive_rate": pl.Float64,
            "mfe10_lift": pl.Float64,
            "hit_lift": pl.Float64,
        }
    )
    if df_mining.is_empty():
        print("  结论: 当前 df_mining 为空，无法验证手工候选规则。")
    else:
        _base_mean = float(df_mining[LABEL_COL].mean())
        _base_hit = float((df_mining[LABEL_COL] >= POSITIVE_MFE_THRESHOLD).mean())
        print(f"  基线 positive_rate: {_base_hit:.2%}")
        print(f"  基线 mfe10_mean:    {_base_mean:+.4f}")

        _rule_rows = []
        for _rule_name, _rule_expr in _rule_defs:
            _df_rule = df_mining.filter(_rule_expr & pl.col(LABEL_COL).is_not_null())
            if _df_rule.is_empty():
                continue

            _rule_mean = float(_df_rule[LABEL_COL].mean())
            _rule_hit = float((_df_rule[LABEL_COL] >= POSITIVE_MFE_THRESHOLD).mean())
            _rule_rows.append(
                {
                    "rule": _rule_name,
                    "samples": _df_rule.height,
                    "dates": _df_rule["date"].n_unique(),
                    "mfe10_mean": _rule_mean,
                    "positive_rate": _rule_hit,
                    "mfe10_lift": _rule_mean - _base_mean,
                    "hit_lift": _rule_hit - _base_hit,
                }
            )

        if not _rule_rows:
            print("  结论: 当前没有手工候选规则命中样本。")
        else:
            manual_rule_table = (
                pl.DataFrame(_rule_rows)
                .with_columns(
                    [
                        pl.col("mfe10_mean").round(4),
                        pl.col("positive_rate").round(4),
                        pl.col("mfe10_lift").round(4),
                        pl.col("hit_lift").round(4),
                    ]
                )
                .sort(["hit_lift", "mfe10_lift"], descending=[True, True])
            )
            print(manual_rule_table)

            _best_manual_rule = manual_rule_table.row(0, named=True)
            print(
                "  结论: 当前最强的手工候选规则是 "
                f"{_best_manual_rule['rule']}, "
                f"samples={_best_manual_rule['samples']}, "
                f"positive_rate={_best_manual_rule['positive_rate']:.2%}, "
                f"hit_lift={_best_manual_rule['hit_lift']:+.4f}, "
                f"mfe10_lift={_best_manual_rule['mfe10_lift']:+.4f}"
            )
    return (manual_rule_table,)


@app.cell
def _(candidate_rules, manual_rule_table, pl):
    print("\n" + "=" * 72)
    print("  Step 8. 候选条件收敛")
    print("=" * 72)

    _shortlist_rows = []
    if candidate_rules.height > 0:
        for _row in candidate_rules.head(2).iter_rows(named=True):
            _shortlist_rows.append(
                {
                    "source": "tree",
                    "rule": _row["rule"],
                    "samples": _row["samples"],
                    "positive_rate": _row["positive_rate"],
                    "hit_lift": _row["lift_vs_base"],
                    "mfe10_lift": None,
                }
            )
    if manual_rule_table.height > 0:
        for _row in manual_rule_table.head(3).iter_rows(named=True):
            _shortlist_rows.append(
                {
                    "source": "manual",
                    "rule": _row["rule"],
                    "samples": _row["samples"],
                    "positive_rate": _row["positive_rate"],
                    "hit_lift": _row["hit_lift"],
                    "mfe10_lift": _row["mfe10_lift"],
                }
            )

    if not _shortlist_rows:
        candidate_rule_shortlist = pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "rule": pl.Utf8,
                "samples": pl.Int64,
                "positive_rate": pl.Float64,
                "hit_lift": pl.Float64,
                "mfe10_lift": pl.Float64,
            }
        )
        print("  结论: 当前还没有收敛出可进入下一轮验证的候选条件。")
    else:
        candidate_rule_shortlist = (
            pl.DataFrame(_shortlist_rows)
            .unique(subset=["rule"], keep="first", maintain_order=True)
            .with_columns(
                [
                    pl.col("positive_rate").round(4),
                    pl.col("hit_lift").round(4),
                    pl.col("mfe10_lift").round(4),
                ]
            )
            .sort(
                ["hit_lift", "mfe10_lift", "samples"],
                descending=[True, True, True],
                nulls_last=True,
            )
            .head(5)
        )
        print(candidate_rule_shortlist)
        print(
            "  结论: 当前已收敛出 "
            f"{candidate_rule_shortlist.height} 条候选条件，可直接进入规则增强或 seed 内纯模型对照。"
        )
    return (candidate_rule_shortlist,)


@app.cell
def _(
    ACTIVE_SEED_COL,
    candidate_rule_shortlist,
    feature_scoreboard,
    review_feature,
):
    print("\n" + "=" * 72)
    print("  Step 9. 当前阅读结论")
    print("=" * 72)
    if feature_scoreboard.height == 0:
        print("  当前还没有形成可解释的条件候选，先检查样本量或切换 seed。")
    else:
        _top_feature = feature_scoreboard["feature"][0]
        _top_rule_text = (
            candidate_rule_shortlist["rule"][0]
            if candidate_rule_shortlist.height > 0
            else "当前候选规则仍需继续挖掘"
        )
        print(
            f"  当前最该看的主线: 以 {ACTIVE_SEED_COL} 为样本，"
            f"先盯 {_top_feature} 这种单变量已显出增量的特征，"
            f"再看 {review_feature} 的分箱形状；当前第一优先候选条件是 {_top_rule_text}。"
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
