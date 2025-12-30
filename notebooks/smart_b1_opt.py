import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime
    # 引入机器学习库
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    return CatBoostClassifier, os, pd, pl


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
def _(CatBoostClassifier, pd, pl):
    # ==============================================================================
    # 2. 特征工程：回归初心 (专注回踩结构)
    # ==============================================================================
    def prepare_morphology_dataset(df: pl.LazyFrame) -> pl.DataFrame:
        print("🛠️ [Feature] 正在构建特征 (收紧过滤，强制聚焦形态)...")

        return (
            df.sort(["code", "date"])
            .with_columns([
                pl.col("close_adj").shift(1).over("code").alias("prev_close"),
                pl.col("high_adj").rolling_max(9).over("code").alias("high_9d"),
                pl.col("low_adj").rolling_min(9).over("code").alias("low_9d"),
                # 黄线 YL
                ((pl.col("close_adj").rolling_mean(14).over("code") + 
                  pl.col("close_adj").rolling_mean(28).over("code") + 
                  pl.col("close_adj").rolling_mean(57).over("code") + 
                  pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),
            ])
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
            # --- 🔥 形态学特征 (只保留描述 K 线本身长相的特征) ---
            .with_columns([
                # 1. 下影线比例 (越高越好)
                ((pl.min_horizontal("open_adj", "close_adj") - pl.col("low_adj")) / (pl.col("high_adj") - pl.col("low_adj") + 0.0001)).alias("feat_shadow_ratio"),

                # 2. 急跌系数 (越高说明是洗盘)
                ((pl.col("close_adj") / pl.col("close_adj").shift(3).over("code") - 1) / (pl.col("close_adj") / pl.col("close_adj").shift(10).over("code") - 1 + 0.0001)).alias("feat_drop_violence"),

                # 3. 极速缩量 (越高越好)
                (pl.col("volume").rolling_max(5).over("code").shift(1) / pl.col("volume")).alias("feat_vol_shrink_fast"),

                # 4. 余温 (妖股记忆，这个可以留，因为它不是位置特征)
                ((pl.col("close_adj") / pl.col("prev_close") - 1) > 0.095).cast(pl.Int32).rolling_max(10).over("code").alias("feat_limit_up_recent"),

                # 5. 黄线粘合度 (越低越好，B1核心)
                ((pl.col("low_adj") - pl.col("YL")).abs() / pl.col("YL")).alias("feat_yl_stickiness"),
            
                # 6. J值 (越低越好)
                pl.col("J").alias("feat_J"),
            
                # 7. 实体位置 (Close相对于High/Low的位置)
                # 既然是回踩，收盘价最好不要是光头阳线(那是突破)，也不要是光脚阴线
                # 我们引入一个特征：收盘价在当日振幅的什么位置
                ((pl.col("close_adj") - pl.col("low_adj")) / (pl.col("high_adj") - pl.col("low_adj") + 0.0001)).alias("feat_close_pos_in_candle"),
            ])
        
            # --- 🎯 目标构建 (保持不变) ---
            .with_columns([
                ((pl.col("high_adj").shift(-3).rolling_max(3).over("code") / pl.col("close_adj") - 1) * 100).alias("fwd_max_rise"),
                ((pl.col("close_adj").shift(-3) / pl.col("close_adj") - 1) * 100).alias("fwd_close_ret"),
                ((pl.col("low_adj").shift(-3).rolling_min(3).over("code") / pl.col("close_adj") - 1) * 100).alias("fwd_max_drop"),
            ])
            .with_columns([
                pl.when(
                    (pl.col("fwd_max_rise") >= 6.0) & 
                    (pl.col("fwd_close_ret") >= 2.0) &
                    (pl.col("fwd_max_drop") > -7.0)
                ).then(2)
                .when(
                    (pl.col("fwd_max_rise") >= 3.0) & 
                    (pl.col("fwd_close_ret") >= -1.0) &
                    (pl.col("fwd_max_drop") > -5.0)
                ).then(1)
                .otherwise(0)
                .alias("label_tier")
            ])
        
            # --- 🚪 过滤器修正 (关键！) ---
            # 1. J <= 25: 稍微比 13 放宽一点点，包含华纳药厂那种15左右的，但绝不要 50 的垃圾
            # 2. Close > YL * 0.96: 允许跌破 4%，防止漏掉深洗盘，但不能太离谱
            .filter(
                (pl.col("J") <= 25) & 
                (pl.col("close_adj") > pl.col("YL") * 0.96)
            )
            .drop_nulls()
            .collect()
        )

    def train_morphology_model(df_ml: pl.DataFrame):
        data = df_ml.to_pandas()
        data['date'] = pd.to_datetime(data['date'])

        # 🚨 剔除 feat_left_height, feat_price_pos, feat_yl_slope
        # 强迫模型看“微观结构”
        features = [
            "feat_shadow_ratio", "feat_drop_violence", "feat_vol_shrink_fast", 
            "feat_limit_up_recent", "feat_yl_stickiness", 
            "feat_J", "feat_close_pos_in_candle"
        ]
        target = "label_tier"

        # 切分时间
        split_date = pd.Timestamp("2024-01-01")
        train_data = data[data['date'] < split_date]
        test_data = data[data['date'] >= split_date]

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
    
        metadata_test = test_data[['code', 'date', 'close_raw']].copy()
    
        print(f"📊 训练集分布 (过滤器 J<=25): \n{y_train.value_counts(normalize=True)}")

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03, # 降低学习率，让它慢慢学细节
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            classes_count=3,
            verbose=100,
            random_seed=42,
            allow_writing_files=False,
            auto_class_weights='Balanced'
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)

        # 预测
        probs = model.predict_proba(X_test)
        metadata_test['score_tier2'] = probs[:, 2]
    
        # 特征重要性
        feature_importances = model.get_feature_importance()
        fi_dict = dict(zip(features, feature_importances))
        sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

        print(f"\n====== 🧠 AI 眼中的‘B1微观结构’权重 (去除干扰项) ======")
        for name, score in sorted_fi:
            print(f"{name:<25} | {score:>8.2f}%")

        return metadata_test, model
    return prepare_morphology_dataset, train_morphology_model


@app.cell
def _(load_data, prepare_morphology_dataset, train_morphology_model):
    # ==============================================================================
    # 4. 执行
    # ==============================================================================
    q_full = load_data()
    df_morph = prepare_morphology_dataset(q_full)
    df_res, model = train_morphology_model(df_morph)
    return df_morph, df_res, model


@app.function
def audit_model_confidence(model, df_features, case_list):
    print(f"\n====== 🕵️‍♂️ AI 信心分“验尸”报告 (Target: Tier 2 妖股概率) ======")
    print(f"{'代码':<10} | {'名称':<8} | {'信号日期':<10} | {'妖股概率(P2)':<12} | {'普通概率(P1)':<12} | {'真实结果'}")
    print("-" * 100)

    # 转换数据格式方便查询
    df_pd = df_features.to_pandas()
    df_pd['date_str'] = df_pd['date'].astype(str)

    # 🚨 必须与训练时的特征列表完全一致，加入了 feat_price_pos
    features = [
        "feat_shadow_ratio", "feat_drop_violence", "feat_vol_shrink_fast", 
        "feat_limit_up_recent", "feat_yl_stickiness", 
        "feat_J", "feat_close_pos_in_candle"
    ]

    scores = []

    for item in case_list:
        # 定位该股该日的数据
        mask = (df_pd['code'] == item['code']) & (df_pd['date_str'] == item['date'])
        row = df_pd[mask]

        if len(row) > 0:
            # 提取特征并让模型打分
            X_input = row[features]
            
            # predict_proba 返回形状 [N, 3] -> [[P(0), P(1), P(2)]]
            all_probs = model.predict_proba(X_input)[0] 
            
            p1 = all_probs[1] # Tier 1 (普通涨) 的概率
            p2 = all_probs[2] # Tier 2 (大肉/妖股) 的概率
            
            # 我们主要看 p2，以此作为核心信心分
            print(f"{item['code']:<10} | {item['name']:<8} | {item['date']:<10} | {p2:.4f}       | {p1:.4f}       | {item['result']}")
            scores.append(p2)
        else:
            print(f"{item['code']:<10} | {item['name']:<8} | {item['date']:<10} | {'MISSING':<12} | {'-':<12} | 数据缺失")

    print("-" * 100)
    if scores:
        print(f"💡 平均妖股概率: {sum(scores)/len(scores):.4f}")
        print(f"📈 最高信心: {max(scores):.4f}")
        print(f"📉 最低信心: {min(scores):.4f}")


@app.cell
def _(df_morph, model):
    # 1. 录入我们刚才验过的 Top 10 清单
    # 注意：日期必须是“信号触发日”（即买入前一天）
    top10_audit = [
        {"code": "688456_SH", "date": "2024-07-23", "name": "有研硅", "result": "潜伏/平"},
        {"code": "603990_SH", "date": "2025-06-20", "name": "麦迪科技", "result": "赢 +14%"},
        {"code": "688092_SH", "date": "2024-03-27", "name": "爱科赛博", "result": "赢 +14%"},
        {"code": "000831_SZ", "date": "2025-08-05", "name": "中国稀土", "result": "赢 +48%"},
        {"code": "300301_SZ", "date": "2024-11-21", "name": "长方集团", "result": "避坑 (量大)"},
        {"code": "300125_SZ", "date": "2025-02-20", "name": "聆达股份", "result": "止损 -5%"},
        {"code": "000792_SZ", "date": "2024-11-15", "name": "盐湖股份", "result": "避坑 (压力)"},
        {"code": "603006_SH", "date": "2025-07-31", "name": "联明股份", "result": "时间止损"},
        {"code": "600615_SH", "date": "2024-11-15", "name": "丰华股份", "result": "赢 +30%"},
        {"code": "688369_SH", "date": "2025-08-15", "name": "致远互联", "result": "时间止损"},
    ]

    # 运行验尸
    audit_model_confidence(model, df_morph, top10_audit)
    return


@app.cell
def _(df_morph, model):
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

    # 2. 执行验尸 (复用你已有的 audit_model_confidence 函数)
    print(f"\n====== 🏆 《十大B1完美图》AI 评分验证 ======")
    audit_model_confidence(model, df_morph, perfect_cases)
    return


@app.cell
def _(df_res):
    # ==============================================================================
    # 🕵️‍♂️ 揭秘：AI 眼中的 Top 10 “妖股候选”
    # ==============================================================================
    def show_model_top_picks(df_result, top_n=10):
        # 1. 按 Tier 2 (妖股概率) 倒序排列
        top_picks = df_result.sort_values(by="score_tier2", ascending=False).head(top_n).copy()
    
        print(f"\n====== 🌟 AI 模型自选 Top {top_n} (基于 Tier 2 概率) ======")
        print(f"{'代码':<10} | {'日期':<10} | {'妖股概率(P2)':<12} | {'位置(Pos)':<10} | {'缩量(Shrink)':<10}")
        print("-" * 80)
    
        # 辅助显示一些关键特征，方便我们人工判断它为什么选这些
        # Pos: 0=山底, 1=山顶
        # Shrink: 数值越大，缩量越狠
    
        for index, row in top_picks.iterrows():
            # 格式化输出
            code = row['code']
            date = row['date'].strftime('%Y-%m-%d')
            p2 = row['score_tier2']
            pos = row.get('feat_price_pos', -1) # 防报错
            shrink = row.get('feat_vol_shrink_fast', -1)
        
            print(f"{code:<10} | {date:<10} | {p2:.4f}       | {pos:.2f}       | {shrink:.2f}")

        print("-" * 80)
        print("💡 观察重点：\n1. Pos 是接近 0 (超跌反弹) 还是接近 1 (高位逼空)？\n2. Shrink 是否都很大 (极致缩量)？")

    # 运行揭秘
    show_model_top_picks(df_res)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
