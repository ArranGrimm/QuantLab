import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

"""
Textbook Case Classifier (方向 C, 2026-04-18)
=============================================

背景:
- `perfect_top10b1_analyze.py` 已实证 V1 / V2 / V3 三版 textbook_b1_score 都救不了反向富集
  - V1 等权 mean similarity (textbook14):           Top10% enrichment = 0.74x
  - V2 max-archetype:                                Top10% enrichment = 0.79x
  - V3 共线去重 + 高 |Cohen's d| + hard rule (10+2): Top10% enrichment = 0.84x
- 三种修复路径 (聚合 H1 / 标签 R1 / 特征 H2) 全部部分有效但不足以翻盘

方向 C 的假设:
- 把 "教科书相似度" 从无监督距离 (mean of similarity) 升级到有监督判别模型
- 让 LightGBM 自己学 P(is_textbook_case | features), 而不是手工设计 score 公式
- 评估口径完全对齐 V3, 在 6 档分位 / Top10% enrichment 上比对

核心风险:
- 正样本极少 (11 base 或 51 加 expanded) vs 负样本 ~180k
- 容易过拟合 (模型可能直接记住 11 个 (code, date))
- 缓解: 浅树 + min_data_in_leaf 大 + bagging + class_weight=balanced + 不做 leakage 评估

输出:
- `textbook_b1_prob` - LightGBM 输出的 case 概率
- 评估 [G1]/[G2]/[G3] 表 (与 perfect notebook Step B 同口径), 直接对比 V3
"""


@app.cell
def _():
    """Cell 0: imports + 数据/模型配置。"""
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    from manifests import B1_BASE_TEXTBOOK_CASES
    from manifests.b1_expanded_textbook_cases import EXPANDED_TEXTBOOK_CASES
    from utils import build_b1_research_frame, get_st_blacklist_pl, load_daily_data_full

    pl.Config(tbl_rows=-1, tbl_cols=-1)

    # ── 数据范围 (与 perfect_top10b1_analyze 对齐) ──────────────
    DB_PATH = r"../QuantData/Ashare/qmt_data.duckdb"
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    ST_SNAPSHOT_DATE = "2026-03-31"

    MV_MIN = 40
    MV_MAX = 1500
    CASE_MV_MIN = 0
    MIN_LIST_DAYS = 60
    SEED_J_MAX = 20.0
    ACTIVE_SEED_COL = "seed_mid"
    USE_BULL_ONLY = False
    INCLUDE_ROTATION_KBAR_FEATURES = True

    # ── 评估 horizon ────────────────────────────────────────────
    ACTIVE_HORIZON = 20  # 与 V3 实验对齐

    # ── case 来源 ───────────────────────────────────────────────
    # "base" = 11 个手挑 textbook cases
    # "base_plus_expanded" = base + 40 个 mining 扩容 cases (51 个正样本)
    CASE_SOURCE = "base_plus_expanded"

    # ── LightGBM 参数 (强正则避免过拟合 11 个正样本) ────────────
    LGB_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 80,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1,
    }

    LOOSE_PERIODS = [
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
    ]
    return (
        ACTIVE_HORIZON,
        ACTIVE_SEED_COL,
        B1_BASE_TEXTBOOK_CASES,
        CASE_MV_MIN,
        CASE_SOURCE,
        DB_PATH,
        END_DATE,
        EXPANDED_TEXTBOOK_CASES,
        INCLUDE_ROTATION_KBAR_FEATURES,
        LGB_PARAMS,
        LOOSE_PERIODS,
        MIN_LIST_DAYS,
        MV_MAX,
        MV_MIN,
        SEED_J_MAX,
        START_DATE,
        ST_SNAPSHOT_DATE,
        USE_BULL_ONLY,
        build_b1_research_frame,
        duckdb,
        get_st_blacklist_pl,
        load_daily_data_full,
        mo,
        np,
        pl,
    )


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
    """Cell 1: 加载 q_full + ST 黑名单。"""
    conn = duckdb.connect(DB_PATH, read_only=True)
    st_blacklist = get_st_blacklist_pl(ST_SNAPSHOT_DATE)
    st_blacklist_df = pl.DataFrame({"code": st_blacklist}).lazy()
    q_full = (
        load_daily_data_full(conn)
        .filter(
            (pl.col("date") >= pl.lit(START_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
            & (pl.col("date") <= pl.lit(END_DATE).str.strptime(pl.Date, "%Y-%m-%d"))
        )
        .join(st_blacklist_df, on="code", how="anti")
    )
    print(f"q_full prepared, st_blacklist size = {len(st_blacklist):,}")
    return (q_full,)


@app.cell
def _(
    ACTIVE_SEED_COL,
    B1_BASE_TEXTBOOK_CASES,
    CASE_MV_MIN,
    CASE_SOURCE,
    EXPANDED_TEXTBOOK_CASES,
    INCLUDE_ROTATION_KBAR_FEATURES,
    LOOSE_PERIODS,
    MIN_LIST_DAYS,
    MV_MAX,
    MV_MIN,
    SEED_J_MAX,
    USE_BULL_ONLY,
    build_b1_research_frame,
    pl,
    q_full,
):
    """Cell 2: build research frame (seed pool + case pool, 与 perfect notebook 同口径)。
    textbook_score_version='v1' 仅为底表稳定, 我们在 Cell 4 重新打 case 标签, 不依赖 v1 的
    is_textbook_b1 / textbook_b1_score。"""
    df_all = build_b1_research_frame(
        q_full,
        mv_min=MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
        include_rotation_kbar_features=INCLUDE_ROTATION_KBAR_FEATURES,
        textbook_score_version="v1",
    )
    df_case_source = build_b1_research_frame(
        q_full,
        mv_min=CASE_MV_MIN,
        mv_max=MV_MAX,
        min_list_days=MIN_LIST_DAYS,
        seed_j_max=SEED_J_MAX,
        loose_periods=LOOSE_PERIODS,
        include_rotation_kbar_features=INCLUDE_ROTATION_KBAR_FEATURES,
        textbook_score_version="v1",
    )
    seed_filter = pl.col(ACTIVE_SEED_COL)
    if USE_BULL_ONLY:
        seed_filter = seed_filter & pl.col("is_manual_bull")

    df_seed_base = df_all.filter(seed_filter)

    case_records: list[dict] = []
    if CASE_SOURCE in {"base", "base_plus_expanded"}:
        case_records += list(B1_BASE_TEXTBOOK_CASES)
    if CASE_SOURCE == "base_plus_expanded":
        case_records += [
            {"code": c["code"], "date": c["date"], "name": c["name"]}
            for c in EXPANDED_TEXTBOOK_CASES
        ]

    case_df = (
        pl.DataFrame(case_records)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .select(["code", "date", "name"])
        .unique(subset=["code", "date"])
    )
    print(
        f"case source = {CASE_SOURCE}, total cases = {case_df.height} "
        f"(base = {len(B1_BASE_TEXTBOOK_CASES)}, expanded = {len(EXPANDED_TEXTBOOK_CASES)})"
    )

    df_cases_base = (
        df_case_source.join(case_df.drop("name"), on=["code", "date"], how="inner")
    )
    print(
        f"df_seed_base = {df_seed_base.height:,} rows,  "
        f"df_cases_base = {df_cases_base.height} rows  "
        f"(case 命中 / case 总数 = {df_cases_base.height} / {case_df.height})"
    )
    return case_df, df_all, df_cases_base, df_seed_base


@app.cell
def _(ACTIVE_HORIZON, df_all, df_cases_base, df_seed_base, pl):
    """Cell 3: 给 df_seed / df_cases 补齐 ACTIVE_HORIZON 的前瞻标签
    (build_b1_research_frame 内置只有 10d, 这里按需扩到 ACTIVE_HORIZON)。"""
    def _run():
        _h = ACTIVE_HORIZON
        if f"fwd_mfe_{_h}d" in df_all.columns:
            print(f"  df_all 已含 fwd_mfe_{_h}d, 直接复用")
            return df_seed_base, df_cases_base

        future_high_cols = [
            pl.col("high_adj").shift(-step).over("code").alias(f"_fwd_high_{step}")
            for step in range(1, _h + 1)
        ]
        future_low_cols = [
            pl.col("low_adj").shift(-step).over("code").alias(f"_fwd_low_{step}")
            for step in range(1, _h + 1)
        ]
        future_high_names = [f"_fwd_high_{step}" for step in range(1, _h + 1)]
        future_low_names = [f"_fwd_low_{step}" for step in range(1, _h + 1)]

        df_with_h = (
            df_all.lazy()
            .with_columns(future_high_cols + future_low_cols)
            .with_columns(
                [
                    (pl.max_horizontal(future_high_names) / pl.col("close_adj") - 1).alias(f"fwd_mfe_{_h}d"),
                    (pl.min_horizontal(future_low_names) / pl.col("close_adj") - 1).alias(f"fwd_mae_{_h}d"),
                ]
            )
            .with_columns(
                (pl.col(f"fwd_mfe_{_h}d") / (1 + pl.col(f"fwd_mae_{_h}d").abs())).alias(
                    f"fwd_mfe_risk_adj_{_h}d"
                )
            )
            .drop(future_high_names + future_low_names)
            .collect()
        )

        join_cols = ["code", "date", f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_mfe_risk_adj_{_h}d"]
        _seed_aug = df_seed_base.join(
            df_with_h.select(join_cols), on=["code", "date"], how="left"
        )
        _cases_aug = df_cases_base.join(
            df_with_h.select(join_cols), on=["code", "date"], how="left"
        )
        print(
            f"  注入 horizon={_h}d 标签完成, df_seed_aug = {_seed_aug.height:,} rows, "
            f"df_cases_aug = {_cases_aug.height} rows"
        )
        return _seed_aug, _cases_aug

    df_seed_aug, _df_cases_aug = _run()
    return (df_seed_aug,)


@app.cell
def _(case_df, df_seed_aug, pl):
    """Cell 4: 把 case 标签合并到 df_seed_aug, 形成训练集 (X, y)。
    y = 1: 该 (code, date) 在 case_df 中
    y = 0: 否则
    注意: 训练集 = seed_mid 全体 (case 默认在 seed_mid 里, 但兜底 join 一下)。"""
    case_flag = case_df.select(["code", "date"]).with_columns(
        pl.lit(1).alias("y_is_case")
    )
    df_train = (
        df_seed_aug.join(case_flag, on=["code", "date"], how="left")
        .with_columns(pl.col("y_is_case").fill_null(0).cast(pl.Int8))
    )

    excluded_prefixes = ("fwd_", "_", "is_", "textbook_", "case", "seed_", "y_")
    excluded_exact = {
        "code", "date",
        "open_adj", "high_adj", "low_adj", "close_adj",
        "volume", "amount",
        "market_cap_100m",
        "TRIGGER", "KEY_K", "KEY_K_EXIST", "PLRY_CNT",
        "GOOD28", "MAX28_OK", "YANGYIN_OK",
        "is_manual_bull",
    }
    feature_cols = []
    for col, dtype in zip(df_train.columns, df_train.dtypes):
        if col in excluded_exact:
            continue
        if any(col.startswith(p) for p in excluded_prefixes):
            continue
        if dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean):
            continue
        feature_cols.append(col)

    n_pos = int(df_train["y_is_case"].sum())
    n_neg = df_train.height - n_pos
    print(f"  train rows = {df_train.height:,},  features = {len(feature_cols)}")
    print(f"  positives (y=1) = {n_pos},  negatives (y=0) = {n_neg:,},  pos ratio = {n_pos/df_train.height:.5%}")
    return df_train, feature_cols


@app.cell
def _(LGB_PARAMS, df_train, feature_cols, np, pl):
    """Cell 5: 训练 LightGBM 二分类。
    全样本训练 (no walk-forward 因为 n_pos 太少); 在 Cell 6 用全样本 prob 做评估,
    可以理解为 "in-sample" 的 ranking 能力检验; 真正的 generalization 要做 LOOCV。"""
    try:
        from lightgbm import LGBMClassifier
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=UserWarning)
    except ImportError as exc:
        raise RuntimeError(f"lightgbm not installed: {exc}") from exc

    df_xy = df_train.select(feature_cols + ["y_is_case"]).fill_nan(None).drop_nulls()
    print(f"  rows after dropna = {df_xy.height:,} (lost {df_train.height - df_xy.height:,} rows)")

    X = df_xy.select(feature_cols).to_numpy()
    y = df_xy["y_is_case"].to_numpy().astype(np.int8)
    print(f"  X shape = {X.shape},  y positives after dropna = {int(y.sum())}")

    clf = LGBMClassifier(**LGB_PARAMS)
    clf.fit(X, y)
    print("\n  LightGBM trained.")
    print(f"  best_iteration_ = {clf.best_iteration_}, n_features_in_ = {clf.n_features_in_}")

    df_pred_input = df_train.select(["code", "date"] + feature_cols).fill_nan(None)
    df_pred_ready = df_pred_input.with_columns(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int8) for c in feature_cols]).alias("_n_null")
    ).filter(pl.col("_n_null") == 0).drop("_n_null")
    X_pred = df_pred_ready.select(feature_cols).to_numpy()
    prob = clf.predict_proba(X_pred)[:, 1]

    df_prob = df_pred_ready.select(["code", "date"]).with_columns(
        pl.Series("textbook_b1_prob", prob)
    )
    print(f"\n  predicted on {df_prob.height:,} rows, "
          f"prob mean = {prob.mean():.4f}, std = {prob.std():.4f}, "
          f"min = {prob.min():.4f}, max = {prob.max():.4f}")
    return clf, df_prob


@app.cell
def _(LGB_PARAMS, df_train, feature_cols, np, pl):
    """Cell 5b: 5-fold Stratified CV → Out-Of-Fold prob (OOF)。
    每行的 prob 都来自它**作为 holdout** 时的预测, 完全消除 case 自身泄漏。
    OOF prob 的 enrichment 才是 C 是否真有泛化能力的判官。"""
    try:
        from lightgbm import LGBMClassifier as _LGBMClassifier
        from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
        import warnings as _warnings_mod
        _warnings_mod.filterwarnings("ignore", category=UserWarning)
    except ImportError as exc:
        raise RuntimeError(f"sklearn not installed: {exc}") from exc

    def _run():
        _df_xy = (
            df_train
            .select(["code", "date"] + feature_cols + ["y_is_case"])
            .fill_nan(None)
            .drop_nulls()
        )
        _X = _df_xy.select(feature_cols).to_numpy()
        _y = _df_xy["y_is_case"].to_numpy().astype(np.int8)
        print(f"  X shape = {_X.shape}, y positives = {int(_y.sum())}")

        _K = 5
        _skf = _StratifiedKFold(n_splits=_K, shuffle=True, random_state=42)
        _oof = np.zeros(len(_y), dtype=np.float64)

        _fold_pos_recovered = []
        for _fold_idx, (_tr_idx, _va_idx) in enumerate(_skf.split(_X, _y)):
            _n_pos_tr = int(_y[_tr_idx].sum())
            _n_pos_va = int(_y[_va_idx].sum())
            _clf_fold = _LGBMClassifier(**LGB_PARAMS)
            _clf_fold.fit(_X[_tr_idx], _y[_tr_idx])
            _oof_va = _clf_fold.predict_proba(_X[_va_idx])[:, 1]
            _oof[_va_idx] = _oof_va
            _case_oof = _oof_va[_y[_va_idx] == 1]
            _case_oof_mean = float(_case_oof.mean()) if _case_oof.size else float("nan")
            _fold_pos_recovered.append(int((_case_oof >= 0.5).sum()))
            print(f"  fold {_fold_idx+1}/{_K}: train pos={_n_pos_tr}, val pos={_n_pos_va}, "
                  f"holdout case mean prob = {_case_oof_mean:.4f}, "
                  f"hold case prob>=0.5: {_fold_pos_recovered[-1]}/{_n_pos_va}")

        _df_oof = _df_xy.select(["code", "date", "y_is_case"]).with_columns(
            pl.Series("textbook_b1_prob_oof", _oof)
        )
        _case_oof_all = _oof[_y == 1]
        _base_oof_all = _oof[_y == 0]
        print("\n" + "=" * 72)
        print("  OOF prob 总览 — 关键泛化诊断")
        print("=" * 72)
        print(f"  case (y=1):    n={_case_oof_all.size}, mean={_case_oof_all.mean():.4f}, "
              f"median={np.median(_case_oof_all):.4f}, q25={np.quantile(_case_oof_all, 0.25):.4f}")
        print(f"  baseline(y=0): n={_base_oof_all.size:,}, mean={_base_oof_all.mean():.4f}, "
              f"q90={np.quantile(_base_oof_all, 0.90):.4f}, q99={np.quantile(_base_oof_all, 0.99):.4f}")
        print(f"  case 在 holdout 上恢复 (prob>=0.5) 总计 = {sum(_fold_pos_recovered)}/{int(_y.sum())}")
        return _df_oof

    df_oof = _run()
    return (df_oof,)


@app.cell
def _(ACTIVE_HORIZON, df_oof, df_train, np, pl):
    """Cell 6b: OOF 评估 — 重跑 [G1]/[G2]/[G3], 但 prob 来自 5-fold OOF。
    这才是 C 真正的考核, in-sample 的 5.48x 不算。"""
    def _run():
        _h = ACTIVE_HORIZON

        df_eval = (
            df_train
            .select([
                "code", "date",
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .join(df_oof, on=["code", "date"], how="inner")
            .filter(pl.col(f"fwd_mfe_risk_adj_{_h}d").is_not_null())
        )
        print(f"  eval rows (OOF) = {df_eval.height:,}")

        case_probs = df_eval.filter(pl.col("y_is_case") == 1)["textbook_b1_prob_oof"].to_numpy()
        if case_probs.size == 0:
            clf_threshold = 0.5
        else:
            clf_threshold = float(np.quantile(case_probs, 0.20))
            clf_threshold = max(min(clf_threshold, 0.95), 0.01)
            print(f"  case OOF prob: mean={case_probs.mean():.4f} q20={clf_threshold:.4f} "
                  f"min={case_probs.min():.4f} max={case_probs.max():.4f}")

        df_eval = df_eval.with_columns(
            (pl.col("textbook_b1_prob_oof") >= pl.lit(clf_threshold)).alias("is_textbook_b1_oof")
        )
        n_oof_pos = int(df_eval["is_textbook_b1_oof"].sum())
        print(f"  threshold = {clf_threshold:.4f}, is_textbook_b1_oof=True 样本数 = {n_oof_pos:,}")

        def _agg_block(df, label):
            return {
                "group": label,
                "rows": df.height,
                f"mean_mfe_{_h}d": round(float(df[f"fwd_mfe_{_h}d"].mean() or 0.0), 4),
                f"mean_risk_adj_{_h}d": round(float(df[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4),
                "hit_15pct": round(float((df[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0), 4),
            }

        print("\n" + "=" * 72)
        print(f"  [G1-OOF] 平均前瞻 ({_h}d), prob 来自 5-fold OOF (无 leakage)")
        print("=" * 72)
        print(pl.DataFrame([
            _agg_block(df_eval, "seed_mid 全体"),
            _agg_block(df_eval.filter(pl.col("is_textbook_b1_oof")), "形态像B1 (clf-OOF)"),
            _agg_block(df_eval.filter(~pl.col("is_textbook_b1_oof")), "非B1 (clf-OOF)"),
            _agg_block(
                df_eval.filter((pl.col("is_textbook_b1_oof")) & (pl.col("y_is_case") == 0)),
                "形态像B1 - case自身 (纯泛化)",
            ),
        ]))

        bins = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.75), (0.75, 1.01)]
        g2_rows = []
        for lo, hi in bins:
            sub = df_eval.filter(
                (pl.col("textbook_b1_prob_oof") >= pl.lit(lo))
                & (pl.col("textbook_b1_prob_oof") < pl.lit(hi))
            )
            g2_rows.append({
                "prob_oof_range": f"[{lo:.2f}, {hi:.2f})",
                "rows": sub.height,
                f"mean_mfe_{_h}d": round(float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0), 4) if sub.height else None,
                f"mean_risk_adj_{_h}d": round(float(sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4) if sub.height else None,
            })
        print("\n" + "=" * 72)
        print(f"  [G2-OOF] textbook_b1_prob_oof 6 档分箱 ({_h}d)")
        print("=" * 72)
        print(pl.DataFrame(g2_rows))

        baseline_pos = int(df_eval["is_textbook_b1_oof"].sum())
        baseline_total = df_eval.height
        baseline_ratio = baseline_pos / baseline_total if baseline_total else 0.0

        median_thr = float(df_eval[f"fwd_mfe_risk_adj_{_h}d"].median() or 0.0)
        top10_thr = float(df_eval[f"fwd_mfe_risk_adj_{_h}d"].quantile(0.90) or 0.0)
        df_above_med = df_eval.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") > median_thr)
        df_top10 = df_eval.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)

        def _enrich_row(df_sub, label):
            if df_sub.height == 0:
                return {"sample": label, "total": 0, "b1_like": 0, "b1_ratio": "0.00%",
                        "enrichment": "n/a", f"mean_risk_adj_{_h}d": None}
            pos = int(df_sub["is_textbook_b1_oof"].sum())
            ratio = pos / df_sub.height
            return {
                "sample": label,
                "total": df_sub.height,
                "b1_like": pos,
                "b1_ratio": f"{ratio:.2%}",
                "enrichment": f"{(ratio / baseline_ratio):.2f}x" if baseline_ratio else "n/a",
                f"mean_risk_adj_{_h}d": round(float(df_sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4),
            }

        print("\n" + "=" * 72)
        print(f"  [G3-OOF] 强表现样本中 is_textbook_b1_oof 占比 (enrichment, {_h}d)")
        print("=" * 72)
        print(pl.DataFrame([
            _enrich_row(df_eval, "seed_mid 全体 (baseline)"),
            _enrich_row(df_above_med, f"高于中位数 risk_adj_{_h}d (>{median_thr:.4f})"),
            _enrich_row(df_top10, f"Top 10% risk_adj_{_h}d (>={top10_thr:.4f})"),
        ]))
        print(f"\n  baseline B1-like ratio = {baseline_ratio:.2%}")
        print("\n  对比目标:")
        print("    V3 (perfect notebook)  Top10% enrichment = 0.84x  ← 反向, 需翻盘")
        print("    C in-sample (Cell 6)   Top10% enrichment = 5.48x  ← leakage 可疑")
        print("    C OOF (本格)           Top10% enrichment = ?      ← 真正的考核")
        print("    > 1.0x → C 有泛化能力, 方向 C 成功")
        print("    < 1.0x → C 也是 leakage 假象, textbook 路径全方位证伪")

    _run()
    return


@app.cell
def _(ACTIVE_HORIZON, df_prob, df_train, np, pl):
    """Cell 6: 评估 — 与 perfect notebook Step B 完全同口径
       [G1] is_textbook_b1_clf vs baseline 平均前瞻表现
       [G2] textbook_b1_prob 6 档分箱 (单调性)
       [G3] 强表现样本中 is_textbook_b1_clf 占比 (enrichment)
    口径: is_textbook_b1_clf = (textbook_b1_prob >= threshold), threshold 取 case 上 prob 的 q20
          (对齐 perfect notebook v1/v3 的 threshold 推导方式)。"""
    def _run():
        _h = ACTIVE_HORIZON

        df_eval = (
            df_train
            .select([
                "code", "date", "y_is_case",
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .join(df_prob, on=["code", "date"], how="inner")
            .filter(pl.col(f"fwd_mfe_risk_adj_{_h}d").is_not_null())
        )
        print(f"  eval rows = {df_eval.height:,}")

        case_probs = df_eval.filter(pl.col("y_is_case") == 1)["textbook_b1_prob"].to_numpy()
        if case_probs.size == 0:
            clf_threshold = 0.5
            print("  [WARN] case 样本在 eval 集中为空, threshold 回退到 0.5")
        else:
            clf_threshold = float(np.quantile(case_probs, 0.20))
            clf_threshold = max(min(clf_threshold, 0.95), 0.05)
            print(f"  case prob: mean={case_probs.mean():.4f} q20={clf_threshold:.4f} "
                  f"min={case_probs.min():.4f} max={case_probs.max():.4f}")

        df_eval = df_eval.with_columns(
            (pl.col("textbook_b1_prob") >= pl.lit(clf_threshold)).alias("is_textbook_b1_clf")
        )

        print(f"\n  threshold = {clf_threshold:.4f}, "
              f"is_textbook_b1_clf=True 样本数 = {int(df_eval['is_textbook_b1_clf'].sum()):,}")

        def _agg_block(df, label):
            return {
                "group": label,
                "rows": df.height,
                f"mean_mfe_{_h}d": round(float(df[f"fwd_mfe_{_h}d"].mean() or 0.0), 4),
                f"mean_mae_{_h}d": round(float(df[f"fwd_mae_{_h}d"].mean() or 0.0), 4),
                f"mean_risk_adj_{_h}d": round(float(df[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4),
                "hit_15pct": round(float((df[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0), 4),
            }

        g1 = pl.DataFrame([
            _agg_block(df_eval, "seed_mid 全体"),
            _agg_block(df_eval.filter(pl.col("is_textbook_b1_clf")), "形态像B1 (clf)"),
            _agg_block(df_eval.filter(~pl.col("is_textbook_b1_clf")), "非B1 (clf)"),
        ])
        print("\n" + "=" * 72)
        print(f"  [G1] 平均前瞻表现 ({_h}d), 判别模型版 textbook_b1_prob")
        print("=" * 72)
        print(g1)

        bins = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.75), (0.75, 1.01)]
        g2_rows = []
        for lo, hi in bins:
            sub = df_eval.filter(
                (pl.col("textbook_b1_prob") >= pl.lit(lo))
                & (pl.col("textbook_b1_prob") < pl.lit(hi))
            )
            g2_rows.append({
                "prob_range": f"[{lo:.2f}, {hi:.2f})",
                "rows": sub.height,
                f"mean_mfe_{_h}d": round(float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0), 4) if sub.height else None,
                f"mean_risk_adj_{_h}d": round(float(sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4) if sub.height else None,
            })
        g2 = pl.DataFrame(g2_rows)
        print("\n" + "=" * 72)
        print(f"  [G2] textbook_b1_prob 6 档分箱平均前瞻 ({_h}d) — 越高 prob 应该越强收益")
        print("=" * 72)
        print(g2)

        baseline_pos = int(df_eval["is_textbook_b1_clf"].sum())
        baseline_total = df_eval.height
        baseline_ratio = baseline_pos / baseline_total if baseline_total else 0.0

        median_thr = float(df_eval[f"fwd_mfe_risk_adj_{_h}d"].median() or 0.0)
        top10_thr = float(df_eval[f"fwd_mfe_risk_adj_{_h}d"].quantile(0.90) or 0.0)
        df_above_med = df_eval.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") > median_thr)
        df_top10 = df_eval.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)

        def _enrich_row(df_sub, label):
            if df_sub.height == 0:
                return {"sample": label, "total": 0, "b1_like": 0, "b1_ratio": "0.00%",
                        "enrichment": "n/a", f"mean_risk_adj_{_h}d": None}
            pos = int(df_sub["is_textbook_b1_clf"].sum())
            ratio = pos / df_sub.height
            return {
                "sample": label,
                "total": df_sub.height,
                "b1_like": pos,
                "b1_ratio": f"{ratio:.2%}",
                "enrichment": f"{(ratio / baseline_ratio):.2f}x" if baseline_ratio else "n/a",
                f"mean_risk_adj_{_h}d": round(float(df_sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0), 4),
            }

        g3 = pl.DataFrame([
            _enrich_row(df_eval, "seed_mid 全体 (baseline)"),
            _enrich_row(df_above_med, f"高于中位数 risk_adj_{_h}d (>{median_thr:.4f})"),
            _enrich_row(df_top10, f"Top 10% risk_adj_{_h}d (>={top10_thr:.4f})"),
        ])
        print("\n" + "=" * 72)
        print(f"  [G3] 强表现样本中 is_textbook_b1_clf 占比 (enrichment, {_h}d)")
        print("=" * 72)
        print(g3)
        print(f"\n  baseline B1-like ratio = {baseline_ratio:.2%}")
        print("\n  对比目标 (perfect notebook):")
        print("    V1 Top10% enrichment = 0.74x  (reverse, 14 textbook 软相似度)")
        print("    V2 Top10% enrichment = 0.79x  (reverse, max-archetype)")
        print("    V3 Top10% enrichment = 0.84x  (reverse, 10 高 |d| + hard rule)")
        print("    ≥ 1.0x  → 判别模型成功翻盘")
        print("    > 0.84x → C 比 V3 更接近翻盘, 但未跨越")
        print("    ≤ 0.84x → C 也救不了 textbook 路径, 应转 D 或 A")

    _run()
    return


@app.cell
def _(clf, feature_cols, pl):
    """Cell 7: feature importance — 看模型自己挑了什么 (gain + split)。"""
    importances_gain = clf.booster_.feature_importance(importance_type="gain")
    importances_split = clf.booster_.feature_importance(importance_type="split")
    df_imp = (
        pl.DataFrame({
            "feature": feature_cols,
            "gain": importances_gain,
            "split": importances_split,
        })
        .with_columns(
            (pl.col("gain") / pl.col("gain").sum() * 100).round(2).alias("gain_pct"),
        )
        .sort("gain", descending=True)
    )
    print("\n" + "=" * 72)
    print("  [G4] LightGBM feature_importance (Top 20 by gain)")
    print("=" * 72)
    print(df_imp.head(20))
    print(f"\n  total gain top 10 占比 = {df_imp.head(10)['gain_pct'].sum():.1f}%")
    print(f"  used (gain > 0) features = {df_imp.filter(pl.col('gain') > 0).height} / {len(feature_cols)}")
    return


@app.cell
def _(ACTIVE_HORIZON, df_all, pl):
    """Cell 8 (Step H): stage-0 alpha 检验 — seed_mid 三个条件分别贡献多少 alpha?
    动机: textbook 路径 (V1/V2/V3/C) 都救不活, 但我们一直拿 seed_mid 当 baseline。
          seed_mid 已经包含了 J<=20 + WL>YL + close>YL 三个相当强的过滤,
          这层 stage-0 setup 可能才是 B1 真正的 alpha 来源。
          本格在 df_all (全市场, 仅 mv/list_days/ST 过滤) 上做层层加压, 看每个条件的边际贡献。"""
    def _run():
        _h = ACTIVE_HORIZON

        future_high_cols = [
            pl.col("high_adj").shift(-step).over("code").alias(f"_fwd_high_{step}")
            for step in range(1, _h + 1)
        ]
        future_low_cols = [
            pl.col("low_adj").shift(-step).over("code").alias(f"_fwd_low_{step}")
            for step in range(1, _h + 1)
        ]
        future_high_names = [f"_fwd_high_{step}" for step in range(1, _h + 1)]
        future_low_names = [f"_fwd_low_{step}" for step in range(1, _h + 1)]

        df_all_h = (
            df_all.lazy()
            .with_columns(future_high_cols + future_low_cols)
            .with_columns([
                (pl.max_horizontal(future_high_names) / pl.col("close_adj") - 1).alias(f"fwd_mfe_{_h}d"),
                (pl.min_horizontal(future_low_names) / pl.col("close_adj") - 1).alias(f"fwd_mae_{_h}d"),
                (pl.col("close_adj").shift(-_h).over("code") / pl.col("close_adj") - 1).alias(f"fwd_ret_{_h}d"),
            ])
            .with_columns(
                (pl.col(f"fwd_mfe_{_h}d") / (1 + pl.col(f"fwd_mae_{_h}d").abs())).alias(
                    f"fwd_mfe_risk_adj_{_h}d"
                )
            )
            .drop(future_high_names + future_low_names)
            .filter(pl.col(f"fwd_mfe_risk_adj_{_h}d").is_not_null())
            .filter(pl.col(f"fwd_ret_{_h}d").is_not_null())
            .select([
                "code", "date", "J", "WL", "YL", "close_adj",
                f"fwd_mfe_{_h}d", f"fwd_mae_{_h}d", f"fwd_ret_{_h}d", f"fwd_mfe_risk_adj_{_h}d",
            ])
            .collect()
        )
        print(f"  df_all_h 准备完成, rows = {df_all_h.height:,}")

        cond_J = pl.col("J") <= 20
        cond_WL = pl.col("WL") > pl.col("YL")
        cond_close = pl.col("close_adj") > pl.col("YL")

        strata = [
            ("L0 全市场 (df_all, 仅 mv/list/ST)",       pl.lit(True)),
            ("L1 +J<=20 only (KDJ 超跌)",                cond_J),
            ("L1 +WL>YL only (中长期多头)",               cond_WL),
            ("L1 +close>YL only (个股在多空线上方)",      cond_close),
            ("L2 +J<=20 & WL>YL (= seed_loose)",         cond_J & cond_WL),
            ("L2 +J<=20 & close>YL",                     cond_J & cond_close),
            ("L2 +WL>YL & close>YL",                     cond_WL & cond_close),
            ("L3 三条件全开 (= seed_mid)",                 cond_J & cond_WL & cond_close),
        ]

        rows = []
        n_total = df_all_h.height
        baseline_mfe = float(df_all_h[f"fwd_mfe_{_h}d"].mean() or 0.0)
        baseline_risk = float(df_all_h[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            mfe = float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0)
            mae = float(sub[f"fwd_mae_{_h}d"].mean() or 0.0)
            risk = float(sub[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            hit15 = float((sub[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0)
            rows.append({
                "stratum": label,
                "rows": sub.height,
                "share_of_total": f"{sub.height / n_total:.2%}",
                f"mean_mfe_{_h}d": round(mfe, 4),
                f"mean_mae_{_h}d": round(mae, 4),
                f"mean_risk_adj_{_h}d": round(risk, 4),
                "hit_15pct": round(hit15, 4),
                "mfe_lift_vs_L0": f"{(mfe - baseline_mfe)*100:+.2f}pp",
                "risk_lift_vs_L0": f"{(risk - baseline_risk)*100:+.2f}pp",
            })

        print("\n" + "=" * 72)
        print(f"  [H1] stage-0 alpha 层层加压 (horizon = {_h}d)")
        print("=" * 72)
        print(pl.DataFrame(rows))

        h2_rows = []
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height < 100:
                continue
            top10_thr = float(sub[f"fwd_mfe_risk_adj_{_h}d"].quantile(0.90) or 0.0)
            mean_top = float(sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            mean_bot = float(sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") < top10_thr)[f"fwd_mfe_risk_adj_{_h}d"].mean() or 0.0)
            hit15_top = float((sub.filter(pl.col(f"fwd_mfe_risk_adj_{_h}d") >= top10_thr)[f"fwd_mfe_{_h}d"] >= 0.15).mean() or 0.0)
            h2_rows.append({
                "stratum": label,
                "rows": sub.height,
                f"top10_thr": round(top10_thr, 4),
                f"top10_mean_risk_adj": round(mean_top, 4),
                f"top10_hit_15pct": round(hit15_top, 4),
                f"bot90_mean_risk_adj": round(mean_bot, 4),
            })
        print("\n" + "=" * 72)
        print(f"  [H2] 各层级内部右尾 (Top 10% risk_adj_{_h}d) 强度")
        print("=" * 72)
        print(pl.DataFrame(h2_rows))

        h3_rows = []
        baseline_ret = float(df_all_h[f"fwd_ret_{_h}d"].mean() or 0.0)
        baseline_win = float((df_all_h[f"fwd_ret_{_h}d"] > 0).mean() or 0.0)
        for label, cond in strata:
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            ret_mean = float(sub[f"fwd_ret_{_h}d"].mean() or 0.0)
            ret_median = float(sub[f"fwd_ret_{_h}d"].median() or 0.0)
            ret_std = float(sub[f"fwd_ret_{_h}d"].std() or 0.0)
            win_rate = float((sub[f"fwd_ret_{_h}d"] > 0).mean() or 0.0)
            mfe_mean = float(sub[f"fwd_mfe_{_h}d"].mean() or 0.0)
            mfe_to_ret = (ret_mean / mfe_mean) if mfe_mean else 0.0
            h3_rows.append({
                "stratum": label,
                "rows": sub.height,
                f"mean_ret_{_h}d": round(ret_mean, 4),
                f"median_ret_{_h}d": round(ret_median, 4),
                f"std_ret_{_h}d": round(ret_std, 4),
                "win_rate(>0)": f"{win_rate:.2%}",
                "ret_lift_vs_L0": f"{(ret_mean - baseline_ret)*100:+.2f}pp",
                "win_lift_vs_L0": f"{(win_rate - baseline_win)*100:+.2f}pp",
                "ret/mfe": f"{mfe_to_ret:.2f}",
            })
        print("\n" + "=" * 72)
        print(f"  [H3] 真实持有收益 fwd_ret_{_h}d (close-to-close), 不是 MFE!")
        print("=" * 72)
        print(pl.DataFrame(h3_rows))

        print("\n  ───────────────────── 判读 ─────────────────────")
        print(f"  • [H1/H2 是 MFE-based] mfe = '20 天内最大浮盈', 天然右偏, baseline = 11% 不等于赚 11%")
        print(f"  • [H3 才是真实可交易收益] mean_ret_{_h}d 才是 'close 到 20 天后 close' 的实际期望")
        print(f"  • ret/mfe 比值 ≈ 实际能拿到 mfe 的多少 (1.0=完美抓顶, <0.3=只能吃到零头)")
        print(f"  • L0 mean_ret_{_h}d ≈ 0 → 全市场平均持有 20 天接近零和, 11% MFE 几乎全被回撤吃掉")
        print(f"  • 真正的 alpha = 各层级 ret_lift_vs_L0, 不是 mfe_lift")
        print(f"  • 若 L2 (WL>YL & close>YL) 在 ret 上 lift 仍最大 → H1 结论可信, J<=20 确实是负 alpha")
        print(f"  • 若 lift 在 ret 上接近 0 → MFE-based 的 alpha 是统计幻觉, 实际不可交易")

        return df_all_h

    df_all_h = _run()
    return (df_all_h,)


@app.cell
def _(ACTIVE_HORIZON, df_all_h, np, pl):
    """Cell 9 (Step I): 统计学严格证明 B1 信号 alpha
    动机: backtest 实测 B1 是正收益, 但 close-to-close mean_ret 是负 alpha (H3)。
          说明 B1 是 '波动结构 alpha' 而非 '方向 alpha'。
          本格用 4 个独立的统计检验, 严格证明这种 alpha 在统计学上显著存在,
          脱离 backtest 引擎依赖, 直接给出 p-value / CI。

    检验:
      I1. 截面百分位 alpha + t-test (是否显著 > 50%)
      I2. hit_15pct 二项 z-test (是否显著 > L0)
      I3. 时序月度 alpha t-test (alpha 是否时间稳定)
      I4. Bootstrap 95% CI (lift 是否显著 != 0)
    """
    def _run():
        _h = ACTIVE_HORIZON
        from scipy import stats as _stats

        cond_J = pl.col("J") <= 20
        cond_WL = pl.col("WL") > pl.col("YL")
        cond_close = pl.col("close_adj") > pl.col("YL")
        strata = {
            "L1 +J<=20":          cond_J,
            "L1 +WL>YL":          cond_WL,
            "L1 +close>YL":       cond_close,
            "L2 +WL & close":     cond_WL & cond_close,
            "L3 seed_mid":        cond_J & cond_WL & cond_close,
        }

        ric = f"fwd_mfe_risk_adj_{_h}d"
        mfe = f"fwd_mfe_{_h}d"
        mae = f"fwd_mae_{_h}d"
        ret = f"fwd_ret_{_h}d"

        # ── I1. 截面百分位 alpha ─────────────────────────────────
        # 按 date 分组, 给每个 (date, stock) 算当天 risk_adj 在截面里的百分位
        # 然后看每个 stratum 的样本平均百分位是否显著 > 0.5
        df_pct = df_all_h.with_columns(
            pl.col(ric).rank(method="average").over("date").alias("_rank"),
            pl.len().over("date").alias("_n_day"),
        ).with_columns(
            (pl.col("_rank") / pl.col("_n_day")).alias("_pct")
        )

        i1_rows = []
        for label, cond in strata.items():
            sub = df_pct.filter(cond)
            if sub.height == 0:
                continue
            pcts = sub["_pct"].to_numpy()
            t_stat, p_val = _stats.ttest_1samp(pcts, 0.5)
            i1_rows.append({
                "stratum": label,
                "rows": sub.height,
                "mean_pct": round(float(pcts.mean()), 4),
                "lift_vs_50%": f"{(pcts.mean() - 0.5) * 100:+.2f}pp",
                "t_stat": round(float(t_stat), 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著" if p_val < 0.01 and pcts.mean() > 0.5 else (
                    "⚠ 显著反向" if p_val < 0.01 else "✗ 不显著"),
            })
        print("=" * 72)
        print(f"  [I1] 截面百分位 alpha (在每天截面 {ric} 中的平均排名)")
        print("=" * 72)
        print(pl.DataFrame(i1_rows))

        # ── I2. hit_15pct 二项 z-test ────────────────────────────
        baseline_hit = float((df_all_h[mfe] >= 0.15).mean() or 0.0)
        n_total = df_all_h.height
        i2_rows = []
        for label, cond in strata.items():
            sub = df_all_h.filter(cond)
            if sub.height == 0:
                continue
            n_sub = sub.height
            p_sub = float((sub[mfe] >= 0.15).mean() or 0.0)
            se = (baseline_hit * (1 - baseline_hit) / n_sub) ** 0.5
            z = (p_sub - baseline_hit) / se if se > 0 else 0.0
            p_val = 2 * (1 - _stats.norm.cdf(abs(z)))
            i2_rows.append({
                "stratum": label,
                "rows": sub.height,
                "hit_15pct": f"{p_sub:.4%}",
                "baseline_hit": f"{baseline_hit:.4%}",
                "lift_pp": f"{(p_sub - baseline_hit) * 100:+.2f}pp",
                "z_stat": round(z, 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著" if p_val < 0.01 and p_sub > baseline_hit else (
                    "⚠ 显著反向" if p_val < 0.01 else "✗ 不显著"),
            })
        print("\n" + "=" * 72)
        print(f"  [I2] hit_15pct 二项 z-test (P(fwd_mfe_{_h}d >= 15%) 是否显著高于全市场)")
        print("=" * 72)
        print(pl.DataFrame(i2_rows))

        # ── I3. 时序月度 alpha t-test ────────────────────────────
        # 按 month 算每月 stratum 的 mean(risk_adj) - 全市场 mean(risk_adj)
        # 看月度 alpha 序列是否 t-test 显著 > 0
        df_month = df_all_h.with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("_ym")
        )
        baseline_by_month = df_month.group_by("_ym").agg(
            pl.col(ric).mean().alias("L0_mean")
        )

        i3_rows = []
        for label, cond in strata.items():
            sub_month = df_month.filter(cond).group_by("_ym").agg(
                pl.col(ric).mean().alias("sub_mean"),
                pl.len().alias("sub_n"),
            )
            joined = sub_month.join(baseline_by_month, on="_ym", how="inner").filter(
                pl.col("sub_n") >= 30
            ).with_columns(
                (pl.col("sub_mean") - pl.col("L0_mean")).alias("alpha")
            )
            if joined.height == 0:
                continue
            alphas = joined["alpha"].to_numpy()
            t_stat, p_val = _stats.ttest_1samp(alphas, 0.0)
            pos_months = int((alphas > 0).sum())
            i3_rows.append({
                "stratum": label,
                "n_months": joined.height,
                "mean_monthly_alpha": round(float(alphas.mean()), 5),
                "alpha_in_pp": f"{alphas.mean() * 100:+.3f}pp",
                "pos_month_ratio": f"{pos_months/joined.height:.0%}",
                "t_stat": round(float(t_stat), 2),
                "p_value": f"{p_val:.2e}",
                "verdict": "✓ 显著且稳定" if p_val < 0.05 and alphas.mean() > 0 and pos_months/joined.height > 0.55 else (
                    "✓ 显著但不稳定" if p_val < 0.05 and alphas.mean() > 0 else "✗ 不显著"),
            })
        print("\n" + "=" * 72)
        print(f"  [I3] 月度时序 alpha t-test (月度 stratum_mean - L0_mean 是否显著 > 0)")
        print("=" * 72)
        print(pl.DataFrame(i3_rows))

        # ── I4. Bootstrap 95% CI for mean lift ───────────────────
        baseline_mfe = float(df_all_h[mfe].mean() or 0.0)
        baseline_ric = float(df_all_h[ric].mean() or 0.0)

        rng = np.random.default_rng(42)
        N_BOOT = 1000
        i4_rows = []
        for label, cond in strata.items():
            sub = df_all_h.filter(cond)
            if sub.height < 100:
                continue
            sub_mfe = sub[mfe].to_numpy()
            sub_ric = sub[ric].to_numpy()
            n = sub.height

            mfe_lifts = np.empty(N_BOOT)
            ric_lifts = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                mfe_lifts[b] = sub_mfe[idx].mean() - baseline_mfe
                ric_lifts[b] = sub_ric[idx].mean() - baseline_ric

            mfe_ci = (np.quantile(mfe_lifts, 0.025), np.quantile(mfe_lifts, 0.975))
            ric_ci = (np.quantile(ric_lifts, 0.025), np.quantile(ric_lifts, 0.975))
            i4_rows.append({
                "stratum": label,
                "rows": sub.height,
                "mfe_lift": f"{mfe_lifts.mean()*100:+.3f}pp",
                "mfe_95%_CI": f"[{mfe_ci[0]*100:+.3f}, {mfe_ci[1]*100:+.3f}]pp",
                "mfe_sig": "✓" if mfe_ci[0] > 0 or mfe_ci[1] < 0 else "✗",
                "risk_adj_lift": f"{ric_lifts.mean()*100:+.3f}pp",
                "risk_adj_95%_CI": f"[{ric_ci[0]*100:+.3f}, {ric_ci[1]*100:+.3f}]pp",
                "ric_sig": "✓" if ric_ci[0] > 0 or ric_ci[1] < 0 else "✗",
            })
        print("\n" + "=" * 72)
        print(f"  [I4] Bootstrap 95% CI ({N_BOOT} resamples, mfe_lift / risk_adj_lift)")
        print("=" * 72)
        print(pl.DataFrame(i4_rows))

        print("\n  ───────────────────── 总判读 ─────────────────────")
        print("  四个检验全 ✓ → B1 信号在 '波动结构 alpha' 维度上严格证明")
        print("  I1 显著 → 信号能把样本筛到截面 risk_adj 排名上端 (相对 alpha)")
        print("  I2 显著 → 信号能放大 '出现 15% 以上浮盈' 的概率 (right-tail alpha)")
        print("  I3 显著 → alpha 不是来自某段时间的偶然 (时间稳定性)")
        print("  I4 CI 不跨 0 → mean lift 在 1000 次重采样下稳定 (估计精度)")

    _run()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook 总览 — Textbook Case Classifier (方向 C) + Stage-0 检验

    - **Cell 0**: 配置 (CASE_SOURCE / ACTIVE_HORIZON / LightGBM 参数)
    - **Cell 1**: q_full 加载
    - **Cell 2**: build_b1_research_frame + 选 case (base / base+expanded)
    - **Cell 3**: 注入 ACTIVE_HORIZON 的前瞻标签
    - **Cell 4**: 构造训练集 (X = 数值特征, y = is_textbook_case)
    - **Cell 5**: 训练 LightGBM 二分类 + 全样本预测 prob (in-sample)
    - **Cell 5b**: 5-fold Stratified CV → OOF prob (无 leakage)
    - **Cell 6**: 评估 [G1]/[G2]/[G3] 三表 (与 V1/V2/V3 同口径对比, in-sample)
    - **Cell 6b**: 评估 [G1-OOF]/[G2-OOF]/[G3-OOF] (无 leakage 的真实泛化考核)
    - **Cell 7**: feature_importance (Top 20)
    - **Cell 8 (Step H)**: stage-0 alpha 检验 — J<=20 / WL>YL / close>YL 三条件 + seed_mid 全开
    - **Cell 9 (Step I)**: 统计学严格证明 (4 检验: 截面百分位 + 二项 z + 月度 t + bootstrap CI)

    判读阈值:
    - V1=0.74x / V2=0.79x / V3=0.84x (反向富集)
    - C ≥ 1.0x → 判别模型翻盘, textbook 路径活了
    - 0.84x < C < 1.0x → C 改善了但仍反向, 需要思考下一步
    - C ≤ 0.84x → C 也救不了, textbook 路径全方位证伪, 转方向 D
    - **Step H seed_mid lift > +5pp vs 全市场 → B1 stage-0 本身就是 alpha 池**

    切换 CASE_SOURCE:
    - "base" = 11 个手挑案例 (极少正样本, 模型几乎一定过拟合)
    - "base_plus_expanded" = 51 个 (base 11 + expanded 40, 默认)
    """)
    return


if __name__ == "__main__":
    app.run()
