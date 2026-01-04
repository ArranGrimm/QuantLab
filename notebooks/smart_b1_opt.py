import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime
    # 引入机器学习库
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, precision_score, recall_score
    return (
        CatBoostClassifier,
        Pool,
        classification_report,
        go,
        make_subplots,
        np,
        os,
        pd,
        pl,
        precision_score,
        recall_score,
    )


@app.cell
def _():
    # ==============================================================================
    # 1. 配置与数据加载
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    print("🚀 [Smart B1] 启动! 正在加载全量数据...")
    return (DATA_ROOT,)


@app.cell
def _(DATA_ROOT, os, pl):
    # (A) 加载数据 (复用之前的逻辑)
    # 为了演示方便，这里直接写加载逻辑。如果 simple_b1 里有共享模块也可以 import
    def load_data():
        # (A) 加载复前权行情
        q_adj = (
            pl.scan_parquet(
                os.path.join(DATA_ROOT, "stock_day_adj", "*.parquet"),
                include_file_paths="file_path"
            )
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
            .select(["code", "date", "open", "high", "low", "close", "volume", "amount"])
            .rename({"close": "close_adj", "high": "high_adj", "low": "low_adj", "open": "open_adj"})
            .filter(pl.col("volume") > 0)
        )

        # (B) 加载 Raw (不复权) 和 Capital (股本)
        q_raw = (
            pl.scan_parquet(os.path.join(DATA_ROOT, "stock_day_raw", "*.parquet"), include_file_paths="file_path")
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.from_epoch("time", time_unit="ms").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").dt.date().alias("date")
            ])
            .select(["code", "date", "close"]).rename({"close": "close_raw"})
        )

        q_cap = (
            pl.scan_parquet(os.path.join(DATA_ROOT, "finance_capital", "*.parquet"), include_file_paths="file_path")
            .with_columns([
                pl.col("file_path").str.extract(r"(\d{6}_[A-Z]{2})", 1).alias("code"),
                pl.col("m_anntime").str.strptime(pl.Date, "%Y%m%d").alias("date"),
                pl.col("total_capital").cast(pl.Float64)
            ])
            .select(["code", "date", "total_capital"]).sort(["code", "date"])
        )

        return q_adj.join(q_raw, on=["code", "date"]).sort(["code", "date"]).join_asof(q_cap, on="date", by="code", strategy="backward").with_columns([
            (pl.col("close_raw") * pl.col("total_capital") / 1e8).alias("market_cap_100m")
        ])
    return (load_data,)


@app.cell
def _(
    CatBoostClassifier,
    Pool,
    classification_report,
    pd,
    pl,
    precision_score,
):
    # ==============================================================================
    # 1. 配置：Ztalk 体系核心“天道” (活跃市值多头区域)
    # ==============================================================================
    MANUAL_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),  # 春季躁动
        ("2019-12-16", "2020-03-02"),  # 疫情反弹
        ("2020-06-19", "2020-07-15"),  # 证券疯牛
        ("2020-12-24", "2021-01-25"),  # 新能源抱团
        ("2021-04-16", "2021-09-14"),  # 锂电光伏
        ("2022-04-27", "2022-07-05"),  # 427大反弹
        ("2023-01-15", "2023-04-15"),  # ChatGPT/CPO
        ("2024-02-06", "2024-03-20"),  # 救市AI反弹
        ("2024-09-24", "2024-10-15"),  # 924 史诗暴涨
        ("2025-04-08", "2025-09-04"),  # 2025年慢牛
    ]

    # 定义未来 N 天的窗口
    FUTURE_WINDOW = 10
    # ==============================================================================
    # 2. 特征工程：全能工厂 (形态 + 天道 + 目标)
    # ==============================================================================
    def prepare_morphology_dataset(df: pl.LazyFrame) -> pl.DataFrame:
        print("🛠️ [Feature] 正在构建全量特征 (形态学 + 天道注入)...")

        # --- 辅助函数：构建天道表达式 ---
        # 在 Polars 中高效匹配日期区间
        regime_expr = pl.lit(False)
        for start, end in MANUAL_LOOSE_PERIODS:
            # 逻辑：只要落在任意一个区间内，就是 True
            regime_expr = regime_expr | (
                (pl.col("date") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d")) & 
                (pl.col("date") <= pl.lit(end).str.strptime(pl.Date, "%Y-%m-%d"))
            )

        return (
            df.sort(["code", "date"])
            # 1. 基础数据准备
            .with_columns([
                pl.col("close_adj").shift(1).over("code").alias("prev_close"),
                pl.col("high_adj").rolling_max(9).over("code").alias("high_9d"),
                pl.col("low_adj").rolling_min(9).over("code").alias("low_9d"),
            # 黄白线 (Trend Lines)
                pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
                ((pl.col("close_adj").rolling_mean(14).over("code") + 
                  pl.col("close_adj").rolling_mean(28).over("code") + 
                  pl.col("close_adj").rolling_mean(57).over("code") + 
                  pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),
            ])
            # 2. KDJ 计算
            .with_columns([
                (pl.col("high_9d") - pl.col("low_9d")).alias("kdj_den"),
            ])
            .with_columns([
                pl.when(pl.col("kdj_den") == 0).then(50.0)
                .otherwise((pl.col("close_adj") - pl.col("low_9d")) / pl.col("kdj_den") * 100).alias("rsv"),
            ])
            .with_columns([
                pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K")
            ])
            .with_columns([
                pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D")
            ])
            .with_columns([
                (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
            ])
            # 3. 🔥 核心特征构建 (Micro & Macro)
            .with_columns(
                # --- Macro: 天道 (直接在此处注入) ---
                regime_expr.cast(pl.Int32).alias("market_regime")
            )
            # --- 3.1 基础窗口计算 ---
            .with_columns([
                # 过去20天的最高价（用于定位N型顶点）
                pl.col("high_adj").shift(1).rolling_max(20).over("code").alias("high_20d"),
                # 过去20天的最大成交量（用于对比极致缩量）
                pl.col("volume").shift(1).rolling_max(20).over("code").alias("vol_max_20d"),
                # 5日均量
                pl.col("volume").shift(1).rolling_mean(5).over("code").alias("vol_ma5"),
                # 过去20天的最低价（用于计算涨幅）
                pl.col("low_adj").shift(1).rolling_min(20).over("code").alias("low_20d"),
            ])
            # --- 3.1.5 🔥 插入：关键K (Violent K) 骨架特征 ---
            .with_columns([
                # 第一步：标记每一天是否发生了“暴力行为”
                # 定义：当日涨幅 > 5% 且 量比 > 2.0 (倍量)
                (
                    ((pl.col("close_adj") - pl.col("prev_close")) / pl.col("prev_close") > 0.05) & 
                    (pl.col("volume") / (pl.col("vol_ma5") + 1.0) > 1.8)
                ).cast(pl.Int8).alias("is_violent_day")
            ])
            .with_columns([
                # 第二步：寻找“骨头”
                # 过去20天内，是否出现过至少一次暴力K？(1=有骨头，0=软脚虾)
                # 这就是 B1 的“入场门票”
                pl.col("is_violent_day").rolling_max(40).over("code").alias("has_violent_k"),
                # 第三步：计算距离最近一次暴力K大约过了多久 (用衰减法近似)
                # 这里我们用一个 trick：如果20天内有暴力K，计算其累计强度
                pl.col("is_violent_day").rolling_sum(40).over("code").alias("feat_violent_count"),
            ])
            # --- 3.2 核心量化因子构造 ---
            .with_columns([
                # [特征1: 极致缩量]
                # 当日量能相对于近期天量的比例。越小越好 (例如 0.1 - 0.3)
                (pl.col("volume") / (pl.col("vol_max_20d") + 1.0)).alias("feat_vol_shrink_ratio"),
                # 当日量能相对于5日均量的比例。 < 1.0 代表缩量
                (pl.col("volume") / (pl.col("vol_ma5") + 1.0)).alias("feat_vol_rel_ma5"),
                # [特征2: N型结构位置]
                # 回调深度：0代表高点，1代表跌回20日低点。完美B1通常在 0.3 - 0.6 之间
                ((pl.col("high_20d") - pl.col("close_adj")) / (pl.col("high_20d") - pl.col("low_20d") + 1e-6)).alias("feat_retrace_ratio"),
                # 前期爆发力：过去20天的最大涨幅。如果没涨过，B1无意义。
                # ((pl.col("high_20d") - pl.col("low_20d")) / pl.col("low_20d")).alias("feat_impulse_strength"),
                # [特征3: 双线逻辑] (核心！)
                # 股价距离黄线(大哥线)的距离。 0.0 ~ 0.05 是最佳回踩区间。
                ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL")).alias("feat_dist_to_yellow"),
                # 白黄线开口程度。 > 0 代表多头趋势。
                ((pl.col("WL") - pl.col("YL")) / pl.col("YL")).alias("feat_trend_gap"),
                # [特征4: K线形态]
                # K线实体大小。完美B1通常是十字星或小阴小阳，数值应很小 (< 0.02)
                ((pl.col("close_adj") - pl.col("open_adj")).abs() / pl.col("prev_close")).alias("feat_body_size"),
                # 是否收红 (1=阳线, 0=阴线) - 某些完美图偏好假阴真阳
                (pl.col("close_adj") > pl.col("open_adj")).cast(pl.Int8).alias("feat_is_red"),
                # [特征5: 情绪 J值]
                # J值本身 (直接使用之前算好的 J)
                pl.col("J").alias("feat_J_val"),
                # J值变化率 (勾头向上)
                (pl.col("J") - pl.col("J").shift(1).over("code")).alias("feat_J_delta"),
                # 暴力K既然发生了，当且仅当收盘价在 20日均线之上，且没有跌得太深，才算支撑有效。
                # 这个特征用来惩罚那些“跌穿地板”的伪B1。
                ((pl.col("close_adj") - pl.col("low_20d")) / (pl.col("high_20d") - pl.col("low_20d") + 0.001)).alias("feat_position_in_range"),
                # 乖离率的变种：不仅看距离黄线，还要看距离 20日线（暴力K的生命线）
                ((pl.col("close_adj") - pl.col("close_adj").rolling_mean(20).over("code")) / pl.col("close_adj")).alias("feat_dist_ma20")
            ])
            .with_columns([
                # [特征 X1: 下影线力度] - 视觉锚点的核心
                # 逻辑：下影线越长，说明下方的支撑越强（有人抄底）。
                # 公式：(min(open, close) - low) / close
                ((pl.min_horizontal(["open_adj", "close_adj"]) - pl.col("low_adj")) / pl.col("close_adj")).alias("feat_lower_shadow"),

                # [特征 X2: 上影线压力] 
                # 逻辑：上影线太长不好，说明抛压重。B1 最好是光头或者短上影。
                ((pl.col("high_adj") - pl.max_horizontal(["open_adj", "close_adj"])) / pl.col("close_adj")).alias("feat_upper_shadow"),
            ])
            .with_columns([
                # [特征 X3: 黄线精准踩踏 (Golden Touch)] - Ztalk 绝学
                # 逻辑：最低价 击穿了 黄线，但 收盘价 站回了 黄线之上。
                # 这是最强的 B1 信号——"破而后立"。
                (
                    (pl.col("low_adj") < pl.col("YL")) &    # 盘中跌破黄线
                    (pl.col("close_adj") > pl.col("YL"))    # 收盘站回黄线
                ).cast(pl.Int8).alias("feat_touch_yellow_rebound"),

                # [特征 X4: 实体极小化 (Doji)]
                # 完美B1通常是十字星。实体越小，代表多空平衡，即将变盘。
                # 我们把你之前的 feat_body_size 再次强化一下权重逻辑
                # 这里不用新造，只是提醒你这个特征和下影线结合非常重要
            ])
            .with_columns([
                # 未来N天的最高价
                pl.col("close_adj").shift(-FUTURE_WINDOW).rolling_max(FUTURE_WINDOW).over("code").alias("future_max"),
                # 未来N天的最低价
                pl.col("close_adj").shift(-FUTURE_WINDOW).rolling_min(FUTURE_WINDOW).over("code").alias("future_min"),
            ])
            .with_columns([
                # 计算未来最大收益率
                ((pl.col("future_max") - pl.col("close_adj")) / pl.col("close_adj")).alias("max_return"),
                # 计算未来最大回撤
                ((pl.col("future_min") - pl.col("close_adj")) / pl.col("close_adj")).alias("max_drawdown"),
            ])
            .with_columns([
                # 盈亏比因子
                (pl.col("future_max") / (pl.col("future_min").abs() + 0.001)).alias("risk_reward_ratio"),
                pl.when(
                    # 情况1: 暴利 (涨超 8%，回撤可接受)
                    ((pl.col("max_return") > 0.08) & (pl.col("max_drawdown") > -0.06)) |
                    # 情况2: 稳健 (涨超 4%，几乎无回撤，典型的完美B1)
                    ((pl.col("max_return") > 0.04) & (pl.col("max_drawdown") > -0.02))
                ).then(1).otherwise(0).alias("label")
            ])
            .filter(
                (pl.col("market_regime") == 1) & # 只训练活跃市值多头区域的数据
                (pl.col("has_violent_k") == 1) &         # 没有暴力K的直接去除
                (pl.col("feat_dist_ma20") > -0.1) &     # 不能跌的太狠
                (pl.col("feat_dist_to_yellow") > -0.1) & # 不能跌的太狠
                (pl.col("feat_J_val") < 14) &           # J值处于潜伏区
                (pl.col("feat_trend_gap") > -0.05) &    # 趋势不能坏得太离谱
                (pl.col("volume") > 0)                  # 排除停牌
            )
            # 过滤掉缺失值 (上市前几天的票)
            .drop_nulls()
        )

    # ==============================================================================
    # 3. 训练：一致性训练 (Consistency Training)
    # ==============================================================================
    def train_morphology_model(df_ml: pl.DataFrame):
        print("🚀 [Training] 启动 B1 形态学训练...")
        # 转换为 Pandas 供 CatBoost 使用 (内存足够时 Pandas 对索引支持更好)
        # 注意：这里我们只取 filter 后的数据进行训练
        df_train_pool = df_ml.collect().to_pandas()

        print(f"📊 [Class Balance] 正样本(Label=1) 占比: {df_train_pool['label'].mean():.2%}")
        # -------------------------------------------------------------------------
        # 3.2 定义特征列 (Features)
        # -------------------------------------------------------------------------
        # 自动提取所有以 'feat_' 开头的列作为特征
        feature_cols = [c for c in df_train_pool.columns if c.startswith("feat_")]
        target_col = "label"

        # -------------------------------------------------------------------------
        # 3. 🔥 核心修正：基于日期的严格切分 (Strict Date Split)
        # -------------------------------------------------------------------------
        # 逻辑：找出所有独立的交易日，取前 70% 的日期做训练，后 30% 做验证
        unique_dates = df_train_pool['date'].sort_values().unique()
        split_date_idx = int(len(unique_dates) * 0.7)
        split_date = unique_dates[split_date_idx]

        print(f"⏳ [Time Split] 切分日期锚点: {split_date}")

        # 训练集：日期 < split_date
        train_mask = df_train_pool['date'] < split_date
        # 测试集：日期 >= split_date
        test_mask = df_train_pool['date'] >= split_date

        X_train = df_train_pool.loc[train_mask, feature_cols]
        y_train = df_train_pool.loc[train_mask, target_col]

        X_test = df_train_pool.loc[test_mask, feature_cols]
        y_test = df_train_pool.loc[test_mask, target_col]

        df_test = df_train_pool.loc[test_mask, df_train_pool.columns]

        # 验证一下切分结果
        print(f"✅ [Train Set] {X_train.shape[0]} 样本 | 区间: {df_train_pool.loc[train_mask, 'date'].min().date()} -> {df_train_pool.loc[train_mask, 'date'].max().date()}")
        print(f"✅ [Test Set ] {X_test.shape[0]} 样本 | 区间: {df_train_pool.loc[test_mask, 'date'].min().date()} -> {df_train_pool.loc[test_mask, 'date'].max().date()}")
        # -------------------------------------------------------------------------
        # 3.4 CatBoost 模型训练
        # -------------------------------------------------------------------------
        # 这里的参数是为了捕捉“形态”特意调整的
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.001,
            depth=8,
            l2_leaf_reg=8,            # 正则化，防止过拟合
            loss_function='Logloss',
            eval_metric='Logloss',
            auto_class_weights='Balanced', # 当前是类别不平衡的
            # scale_pos_weight=2.0,           # <-- 加上这行，降低对正样本的渴望，只吃精肉
            early_stopping_rounds=500, # 给他 500 轮的耐心，别 50 轮就跑
            # 增加一点随机性，防止在局部最优解死循环
            bagging_temperature=1,
            random_seed=42,
            verbose=100,
            allow_writing_files=False
        )

        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

        model.fit(
            train_pool,
            eval_set=test_pool,
            use_best_model=True
        )

        # -------------------------------------------------------------------------
        # 3.5 评估与归因
        # -------------------------------------------------------------------------
        print("\n🧐 [Evaluation] 测试集表现:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # 重点看精准率 (Precision)：我们要的是“不出手则已，出手必中”
        precision = precision_score(y_test, y_pred)
        print(f"🎯 B1 信号精准率 (Precision): {precision:.2%}")

        # -------------------------------------------------------------------------
        # 3.6 特征重要性可视化 (看看机器眼中的“完美图”长啥样)
        # -------------------------------------------------------------------------
        fea_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        print("\n🌟 [Feature Importance] Ztalk 体系核心因子排名:")
        for row in fea_imp.itertuples():
            # row.feature 和 row.importance 对应列名
            print(f"{row.feature:<30} | {row.importance:.6f}")

        return model, feature_cols, df_test
    return prepare_morphology_dataset, train_morphology_model


@app.cell
def _(load_data, prepare_morphology_dataset, train_morphology_model):
    # ==============================================================================
    # 4. 执行
    # ==============================================================================
    q_full = load_data()
    df_features = prepare_morphology_dataset(q_full)
    model, feature_cols, df_test = train_morphology_model(df_features)
    return df_features, df_test, feature_cols, model


@app.cell
def _(
    df_test,
    feature_cols,
    go,
    make_subplots,
    model,
    np,
    precision_score,
    recall_score,
):
    def analyze_thresholds_plotly(model, df_test, feature_cols, target_col="label"):
        """
        使用 Plotly 绘制交互式阈值扫描图 (Sniper Mode)
        """
        print("🚀 [Sniper Mode] 正在启动交互式阈值扫描...")

        # -----------------------------------------------------------
        # 1. 确保拿到测试集 (如果你之前的 X_test 还在内存里，可以跳过这一步)
        # -----------------------------------------------------------

        X_test = df_test[feature_cols]
        y_test = df_test[target_col]

        # 2. 获取预测概率 (Probability)
        # CatBoost 的 predict_proba 返回 [class0_prob, class1_prob]
        y_proba = model.predict_proba(X_test)[:, 1]

        # 3. 扫描阈值
        thresholds = np.arange(0.3, 0.98, 0.02) # 颗粒度细一点
        precisions = []
        recalls = []
        counts = []

        print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Signal Count':<12}")
        print("-" * 55)

        for thr in thresholds:
            y_pred_thr = (y_proba >= thr).astype(int)

            # 计算指标
            p = precision_score(y_test, y_pred_thr, zero_division=0)
            r = recall_score(y_test, y_pred_thr, zero_division=0)
            c = np.sum(y_pred_thr)

            precisions.append(p)
            recalls.append(r)
            counts.append(c)

            # 只打印关键节点，避免刷屏
            if int(thr * 100) % 10 == 0: 
                print(f"{thr:.2f}       | {p:.2%}     | {r:.2%}     | {c}")

        # -----------------------------------------------------------
        # 4. Plotly 可视化 (双轴图：左轴胜率/召回，右轴信号数量)
        # -----------------------------------------------------------
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 胜率曲线 (红色 - 核心关注)
        fig.add_trace(
            go.Scatter(x=thresholds, y=precisions, name="Precision (胜率)",
                       mode='lines+markers', line=dict(color='firebrick', width=3),
                       marker=dict(size=6),
                       hovertemplate="阈值: %{x:.2f}<br>胜率: %{y:.2%}<extra></extra>"),
            secondary_y=False,
        )

        # 召回曲线 (蓝色 - 辅助参考)
        fig.add_trace(
            go.Scatter(x=thresholds, y=recalls, name="Recall (召回)",
                       mode='lines', line=dict(color='royalblue', width=2, dash='dot'),
                       hovertemplate="阈值: %{x:.2f}<br>召回: %{y:.2%}<extra></extra>"),
            secondary_y=False,
        )

        # 信号数量 (灰色柱状图 - 背景)
        fig.add_trace(
            go.Bar(x=thresholds, y=counts, name="Signal Count (开枪次数)",
                   marker=dict(color='lightgrey', opacity=0.5),
                   hovertemplate="阈值: %{x:.2f}<br>信号数: %{y}次<extra></extra>"),
            secondary_y=True,
        )

        # 布局美化
        fig.update_layout(
            title="<b>Ztalk Sniper Analysis</b>: 阈值 vs 胜率 (寻找 Sweet Spot)",
            xaxis_title="Confidence Threshold (置信度阈值)",
            hovermode="x unified", # 鼠标一放，三条线数据同时显示
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
        )

        # 坐标轴设置
        fig.update_yaxes(title_text="Score (0~1.0)", range=[0, 1.05], secondary_y=False)
        fig.update_yaxes(title_text="Signal Count (数量)", showgrid=False, secondary_y=True)

        fig.show()

    # ==========================================
    # 🎯 运行入口
    # ==========================================
    # 确保传入的是你之前转好的 pandas dataframe (df_features)
    # 和训练好的 model 以及 feature_cols
    try:
        analyze_thresholds_plotly(model, df_test, feature_cols)
    except NameError as e:
        print("❌ 错误：找不到变量。请确保你已经运行了上面的训练代码，并且 df_features 和 model 还在内存中。")
        print(f"详情: {e}")
    return


@app.cell
def _(pd):
    def audit_model_confidence(model, df_test, case_list, feature_cols):
        """
        审计函数：用“十大完美B1”去考考模型，看它打多少分。
        Args:
            model: 训练好的 CatBoost 模型
            df_test: 测试集
            case_list: 完美案例清单 (List[Dict])
            feature_cols: 模型训练时使用的特征列名列表
        """
        print(f"\n====== 🕵️‍♂️ AI 信心分“验尸”报告 (Target: 完美 B1) ======")

        # 准备特征矩阵 X
        # 第一步：从字典列表中提取所有的 code 组成一个列表
        df_cases = pd.DataFrame(case_list)
        df_cases['date'] = pd.to_datetime(df_cases['date'])
        df_perfect = pd.merge(df_test, df_cases[['code', 'date', 'name', 'result']], on=['code', 'date'], how='inner')
        X_audit = df_perfect[feature_cols]

        # 5. 让 AI 打分 (Predict Proba)
        # 取 class=1 (是 B1) 的概率
        confidence_scores = model.predict_proba(X_audit)[:, 1]

        # 6. 生成最终报告
        report = df_perfect[["code", "date", "name", "result"]]
        report["AI_Score"] = (confidence_scores * 100).round(1) # 转为 0-100 分

        # --- 增加 Ztalk 评级逻辑 ---
        def rate_it(score):
            if score >= 80: return "⭐⭐⭐ (神懂)"
            elif score >= 60: return "⭐⭐ (及格)"
            elif score >= 50: return "⭐ (犹豫)"
            else: return "❌ (瞎眼)"

        report["Rating"] = report["AI_Score"].apply(rate_it)

        # 按分数降序排列，看看谁是 AI 眼中的“最美”
        report = report.sort_values("AI_Score", ascending=False)

        print("\n📊 审计结果详情：")
        print(report.to_markdown(index=False)) # 需要 pip install tabulate，没有的话 print(report) 也可以

        # 7. 诊断建议
        avg_score = report["AI_Score"].mean()
        print(f"\n💡 [合伙人诊断]: 平均信心分 {avg_score:.1f}")
        if avg_score < 60:
            print("🔴 警报：模型对“标杆”的识别度过低！")
            print("可能原因：")
            print("1. 特征缺失：完美图的“视觉美感”（如均线粘合度、K线极小实体）未被量化。")
            print("2. 负样本太强：训练集中混入了太多类似形态但失败的案例（假 B1）。")
            print("建议：检查 feature_importance，剔除干扰噪音，或者增加“缩量极致度”的权重。")
        else:
            print("🟢 通过：模型基本掌握了 B1 的审美标准。可以进行实盘回测。")
    return (audit_model_confidence,)


@app.cell
def _():
    # ==============================================================================
    # 🎯 验证《十大B1完美图》经典案例
    # ==============================================================================

    # 1. 构造案例清单 (已根据 PDF 提取精准代码与日期)
    # 注意：后缀已适配你的 polars 数据格式 (_SH/_SZ)
    perfect_cases = [
        # --- 完美一：标准缩量与反包 ---
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂", "result": "完美一(标准)"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升", "result": "完美一(反包)"},

        # --- 完美二：双底与极缩 ---
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物", "result": "完美二(双底)"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技", "result": "完美二(蓄势)"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业", "result": "完美二(极缩)"},

        # --- 完美三：趋势与回踩 ---
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科", "result": "完美三(趋势)"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池", "result": "完美三(突破)"},
        {"code": "600184_SH", "date": "2025-07-10", "name": "光电股份", "result": "完美三(回踩)"},

        # --- 完美四：激进与压轴 ---
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材", "result": "完美四(激进)"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康", "result": "完美四(压轴)"},
    ]
    return (perfect_cases,)


@app.cell
def _(audit_model_confidence, df_test, feature_cols, model, perfect_cases):
    audit_model_confidence(model, df_test, perfect_cases, feature_cols)
    return


@app.cell
def _(df_features, pd, perfect_cases, pl):
    def diagnose_perfect_cases(df_full_features, case_list):
        """
        诊断完美案例为什么被过滤或分低
        """
        print("🕵️‍♂️ [Case Diagnosis] 正在对 10 大完美案例进行“体检”...")

        # 提取案例的 Code 和 Date
        target_codes = [c['code'] for c in case_list]
        target_dates = [c['date'] for c in case_list]

        # 从全量特征中提取这些数据（不过滤任何东西）
        # 注意：我们要看的是 B1 当天的数据，以及它过去 20 天的历史
        debug_df = (
            df_full_features
            .filter(pl.col("code").is_in(target_codes))
            .collect() # 先把相关票的所有历史拿出来，方便回溯
            .to_pandas()
        )

        # 逐个案例分析
        results = []
        for case in case_list:
            code = case['code']
            date = case['date'] # B1 发生的日期

            # 找到那一天的数据
            row = debug_df[(debug_df['code'] == code) & (debug_df['date'].astype(str) == date)]

            if len(row) == 0:
                results.append({"name": case['name'], "status": "数据缺失 (Data Missing)"})
                continue

            # 获取当天的特征值
            rec = row.iloc[0]

            # 回溯查找过去 20 天最大的一根阳线（也就是模型眼里的“暴力K”）
            # 我们需要找到 date 之前 20 天的数据片段
            idx = row.index[0]
            past_20_days = debug_df.loc[idx-20 : idx] # 简单切片，假设数据是按日期排序的

            # 检查这 20 天里，有没有满足我们定义的“暴力K”
            # 定义回顾：涨幅 > 5% (0.05) 且 量比 > 2.0
            # 注意：这里要手动重算一遍，看看最大值是多少

            # 这里的计算比较粗略，主要看逻辑
            # 假设我们能在 row 里看到 feat_has_violent_k

            has_violent = rec.get('has_violent_k', 0)
            # max_impulse = rec.get('feat_impulse_strength', 0)
            is_positive = rec.get('label', 0)

            # 诊断结论
            reason = []
            if has_violent == 0:
                reason.append("❌ 无暴力K支撑")
            else:
                reason.append("✅ 有暴力K")


            if is_positive == 0:
                reason.append("❌ 训练目标设定有误")
            else:
                reason.append("✅ 在训练正样本中")

            results.append({
                "name": case['name'], 
                "date": date,
                "has_violent_k": has_violent,
                "is_positive": is_positive,
                "diagnosis": " | ".join(reason)
            })

        # 打印报表
        print(pd.DataFrame(results).to_markdown())

    # 运行诊断
    # 假设 df_raw (或者 prepare_morphology_dataset 生成的 lazy frame) 还在
    # 这里传入的是还没被 filter 的全量特征 df
    diagnose_perfect_cases(df_features, perfect_cases)
    return


@app.cell
def _(df_features, feature_cols, model, np):
    def diagnose_bad_score(model, df_features, feature_cols, target_case_code, target_case_date):
        """
        照妖镜：对比“完美案例”与“模型眼里的高分股”的特征差异
        """
        print(f"🕵️‍♂️ [Micro Diagnosis] 正在解剖案例: {target_case_code} @ {target_case_date}")
        df_pool = df_features.collect().to_pandas()
        # 1. 找到目标案例的特征行
        # 注意：这里需要 df_pool 包含 code 和 date 列，且特征齐全
        case_row = df_pool[
            (df_pool["code"] == target_case_code) & 
            (df_pool["date"].astype(str) == target_case_date)
        ]

        if len(case_row) == 0:
            print("❌ 找不到该案例的数据行，请检查日期是否完全匹配！")
            return

        # 2. 找到模型眼里的“高分股” (Score > 0.8)
        # 获取所有样本的打分
        y_proba = model.predict_proba(df_pool[feature_cols])[:, 1]
        high_score_indices = np.where(y_proba > 0.8)[0]

        if len(high_score_indices) < 10:
            print("⚠️ 高分样本太少，降低标准到 0.7...")
            high_score_indices = np.where(y_proba > 0.7)[0]

        high_score_samples = df_pool.iloc[high_score_indices]

        # 3. 对比核心特征
        # 我们取出 Feature Importance 前 10 的特征进行对比
        top_features = [
            "feat_dist_ma20", "feat_body_size", "feat_trend_gap", 
            "feat_dist_to_yellow", "feat_upper_shadow", "feat_lower_shadow",
            "feat_vol_rel_ma5", "feat_position_in_range", "feat_retrace_ratio", "feat_J_val"
        ]

        print(f"\n📊 特征对比表 (为什么只给了 {case_row.iloc[0].get('AI_Score', 'N/A')} 分？)")
        print(f"{'Feature (特征)':<25} | {'Case Value (案例值)':<18} | {'Avg High Score (高分均值)':<22} | {'Diff (差距)'}")
        print("-" * 80)

        rec = case_row.iloc[0]
        avg_stats = high_score_samples[top_features].mean()

        for feat in top_features:
            val_case = rec.get(feat, 0)
            val_ideal = avg_stats[feat]
            diff = val_case - val_ideal

            # 简单的评价标记
            mark = ""
            # 比如 body_size 越小越好，如果 case 很大，那就是扣分项
            if feat == "feat_body_size" and val_case > val_ideal * 1.5: mark = "❌ 太大了"
            if feat == "feat_dist_ma20" and val_case < val_ideal - 0.05: mark = "❌ 跌太深"
            if feat == "feat_vol_rel_ma5" and val_case > val_ideal * 1.5: mark = "❌ 没缩量"

            print(f"{feat:<25} | {val_case:>10.4f}         | {val_ideal:>10.4f}             | {diff:>10.4f} {mark}")

    # ==========================================
    # 🎯 运行诊断：针对“华纳药厂”
    # ==========================================
    # 假设 df_train_pool 是你刚才训练用的 dataframe（转pandas后的）
    # 且里面包含 code, date 列 (如果没有，你需要从原始数据里把 code/date 拼回来)
    diagnose_bad_score(model, df_features, feature_cols, "688799_SH", "2025-05-12")
    return


@app.cell
def _(df_test, feature_cols, model):
    def show_top_model_picks_strict(model, df_test, feature_cols, top_n=15):
        """
        展示模型眼里最完美的 Top N 案例 (修正版：严格日期切分)
        """
        print(f"🚀 [Top Picks] 正在严格测试集中寻找 '绝世好股' (Top {top_n})...")

        # -------------------------------------------------------
        # 1. 锁定测试集 (Test Set Locking)
        # -------------------------------------------------------
        print(f"✅ [Test Set Loaded] 加载测试样本: {len(df_test)} 条")
        print(f"   时间范围: {df_test['date'].min()} -> {df_test['date'].max()}")

        # -------------------------------------------------------
        # 2. 模型打分 (Scoring)
        # -------------------------------------------------------
        X_test = df_test[feature_cols]

        # 获取预测概率 (Probability)
        y_proba = model.predict_proba(X_test)[:, 1]

        # -------------------------------------------------------
        # 3. 构造榜单
        # -------------------------------------------------------
        res_df = df_test[["code", "date", "label"]].copy()
        res_df["AI_Score"] = y_proba

        # 把核心特征也附带上，方便肉眼诊断
        key_feats = ["feat_dist_ma20", "feat_dist_to_yellow", "feat_retrace_ratio", "feat_body_size", "feat_vol_rel_ma5"]
        for f in key_feats:
            if f in df_test.columns:
                res_df[f] = df_test[f]

        # 排序取 Top N
        top_picks = res_df.sort_values("AI_Score", ascending=False).head(top_n)

        # -------------------------------------------------------
        # 4. 深度诊断打印
        # -------------------------------------------------------
        print("\n🏆 模型眼里的 '2023年后完美B1' 排行榜:")
        print(f"{'Code':<10} | {'Date':<10} | {'Score':<6} | {'Label':<5} | {'MA20 Dist':<10} | {'Ylw Dist':<10} | {'Retrace':<8} | {'点评'}")
        print("-" * 110)

        for _, row in top_picks.iterrows():
            # 自动生成一句点评
            comment = "✅ 趋势完美"

            # 1. 检查是否深跌 (Ztalk大忌)
            if row['feat_dist_ma20'] < -0.10: 
                comment = "❌ 深跌超跌 (垃圾)"
            elif row['feat_dist_ma20'] < -0.02: 
                comment = "⚠️ 破位回踩"

            # 2. 检查是否悬空 (没踩实)
            elif row['feat_dist_ma20'] > 0.05: 
                comment = "⚠️ 悬空未踩"

            # 3. 检查回吐幅度
            if row['feat_retrace_ratio'] > 0.9: 
                comment += " + 涨幅全吐"

            # 格式化日期
            d_str = str(row['date'])[:10]

            print(f"{row['code']:<10} | {d_str:<10} | {row['AI_Score']:.4f} | {row['label']:<5} | {row['feat_dist_ma20']:<10.2%} | {row['feat_dist_to_yellow']:<10.2%} | {row['feat_retrace_ratio']:<8.2f} | {comment}")

    # ==========================================
    # 🎯 运行选秀
    # ==========================================
    show_top_model_picks_strict(model, df_test, feature_cols)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
