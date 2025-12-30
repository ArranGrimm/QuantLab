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
    from loguru import logger
    # 引入机器学习库
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    return CatBoostClassifier, logger, os, pd, pl


@app.cell
def _(logger):
    # ==============================================================================
    # 1. 配置与数据加载
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    logger.info("🚀 [Smart B1] 启动! 正在加载全量数据...")
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
def _(CatBoostClassifier, logger, pd, pl):
    # 2. 特征工程：纯形态学 (Pure Morphology)
    # ==============================================================================
    def prepare_morphology_dataset(df: pl.LazyFrame) -> pl.DataFrame:
        logger.info("🛠️ [Feature] 正在构建‘形态学’特征 (寻找娜娜图)...")

        return (
            df.sort(["code", "date"])
            .with_columns([
                pl.col("close_adj").shift(1).over("code").alias("prev_close"),
                pl.col("high_adj").rolling_max(9).over("code").alias("high_9d"),
                pl.col("low_adj").rolling_min(9).over("code").alias("low_9d"),
                # 黄线 YL
                ((pl.col("close_adj").rolling_mean(14).over("code") + pl.col("close_adj").rolling_mean(28).over("code") + pl.col("close_adj").rolling_mean(57).over("code") + pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),
            ])
            .with_columns([
                # 基础 KDJ
                (pl.col("high_9d") - pl.col("low_9d")).alias("kdj_den"),
            ])
            .with_columns([
                pl.when(pl.col("kdj_den") == 0).then(50.0).otherwise((pl.col("close_adj") - pl.col("low_9d")) / pl.col("kdj_den") * 100).alias("rsv"),
            ])
            .with_columns([
                pl.col("rsv").ewm_mean(com=2, adjust=False).over("code").alias("K")
            ])
            .with_columns([
                pl.col("K").ewm_mean(com=2, adjust=False).over("code").alias("D")
            ])
            .with_columns([
                (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
                # B1 信号
                ((3 * pl.col("K") - 2 * pl.col("D") <= 13) & (pl.col("close_adj") > pl.col("YL"))).alias("is_b1")
            ])
            # --- 🔥 形态学特征 (Morphology Features) ---
            .with_columns([
                # 1. 下影线比例 (feat_shadow_ratio)
                # 逻辑：(min(Open, Close) - Low) / (High - Low)
                # 越大说明针越长，回踩支撑越强
                ((pl.min_horizontal("open_adj", "close_adj") - pl.col("low_adj")) / (pl.col("high_adj") - pl.col("low_adj") + 0.0001)).alias("feat_shadow_ratio"),

                # 2. 急跌系数 (feat_drop_violence)
                # 逻辑：(10日最高价 / 现价 - 1) / (10日最高价距今天数 + 1)
                # 这个很难精确矢量化，简化为：最近3天的跌幅 / 最近10天的跌幅
                # 如果最近3天跌幅占据了大部分跌幅，说明是急跌
                ((pl.col("close_adj") / pl.col("close_adj").shift(3).over("code") - 1) / (pl.col("close_adj") / pl.col("close_adj").shift(10).over("code") - 1 + 0.0001)).alias("feat_drop_violence"),

                # 3. 极速缩量 (feat_vol_shrink_fast)
                # 逻辑：前5天最大成交量 / 今日成交量
                # 越大说明缩量越剧烈
                (pl.col("volume").rolling_max(5).over("code").shift(1) / pl.col("volume")).alias("feat_vol_shrink_fast"),

                # 4. 余温 (feat_limit_up_recent)
                # 逻辑：过去10天内是否有涨停板 (1=有, 0=无)
                ((pl.col("close_adj") / pl.col("prev_close") - 1) > 0.095).cast(pl.Int32).rolling_max(10).over("code").alias("feat_limit_up_recent"),

                # 5. 黄线粘合度 (feat_yl_stickiness)
                # 逻辑：最低价距离黄线的百分比 (绝对值)
                # 越小说明踩得越准
                ((pl.col("low_adj") - pl.col("YL")).abs() / pl.col("YL")).alias("feat_yl_stickiness"),

                # 6. 左侧高度 (feat_left_height)
                # 过去20天最大涨幅，衡量是不是妖股下来的
                (pl.col("high_adj").rolling_max(20).over("code") / pl.col("close_adj") - 1).alias("feat_left_height"),

                # 7. 趋势斜率 (feat_yl_slope) - 保留
                (pl.col("YL") / pl.col("YL").shift(5).over("code") - 1).alias("feat_yl_slope"),

                # 8. J值 (feat_J) - 保留
                pl.col("J").alias("feat_J"),
            ])

            # --- 🎯 暴利目标 (Target: Explosive Rebound) ---
            # 逻辑：未来5天内，最高价曾摸到 +5% (爆发)，且最低价没跌破 -3.5% (安全)
            .with_columns([
                pl.col("high_adj").shift(-5).rolling_max(5).over("code").alias("t1t5_max_high"),
                pl.col("low_adj").shift(-5).rolling_min(5).over("code").alias("t1t5_min_low"),
            ])
            .with_columns([
                # 相对于 T日收盘价 的涨幅
                (pl.col("t1t5_max_high") / pl.col("close_adj") - 1).alias("fwd_max_ret"),
                (pl.col("t1t5_min_low") / pl.col("close_adj") - 1).alias("fwd_min_ret"),
            ])
            .with_columns([
                # Label = 1 if Max > 5% AND Min > -3.5%
                ((pl.col("fwd_max_ret") >= 0.05) & (pl.col("fwd_min_ret") >= -0.035)).cast(pl.Int32).alias("label_explosive")
            ])
            .filter(pl.col("is_b1")) # 只保留 B1
            .drop_nulls()
            .collect()
        )

    # ==============================================================================
    # 3. 训练：寻找图形规律
    # ==============================================================================
    def train_morphology_model(df_ml: pl.DataFrame):
        data = df_ml.to_pandas()
        data['date'] = pd.to_datetime(data['date'])

        # 新的特征列表 (不含市值)
        features = [
            "feat_shadow_ratio", "feat_drop_violence", "feat_vol_shrink_fast", 
            "feat_limit_up_recent", "feat_yl_stickiness", "feat_left_height",
            "feat_yl_slope", "feat_J"
        ]
        target = "label_explosive"

        # 切分时间
        split_date = pd.Timestamp("2024-01-01")
        train_data = data[data['date'] < split_date]
        test_data = data[data['date'] >= split_date]

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        metadata_test = test_data[['code', 'date', 'close_raw']].copy()

        logger.info(f"🐱 CatBoost 启动 (目标：寻找5天内暴涨5%且不破位的图形)...")

        model = CatBoostClassifier(
            iterations=800,        # 稍微增加迭代
            learning_rate=0.03,    # 降低学习率，更细腻
            depth=6,
            loss_function='Logloss',
            verbose=100,
            random_seed=42,
            allow_writing_files=False
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        # 预测
        probs = model.predict_proba(X_test)[:, 1]
        metadata_test['prob_win'] = probs

        # 合并特征用于展示
        for f in features:
            metadata_test[f] = X_test[f]

        # --- 特征重要性 ---
        feature_importances = model.get_feature_importance()
        fi_dict = dict(zip(features, feature_importances))
        sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

        print(f"\n====== 🧠 AI 眼中的‘完美图形’权重 (Morphology) ======")
        print(f"{'特征名':<25} | {'权重(%)':<10} | {'物理含义'}")
        print("-" * 70)

        interpretations = {
            "feat_shadow_ratio": "金针探底 (下影线)",
            "feat_drop_violence": "急跌系数 (洗盘凶狠度)",
            "feat_vol_shrink_fast": "极速缩量 (量能断崖)",
            "feat_limit_up_recent": "余温 (近期有板)",
            "feat_yl_stickiness": "精准回踩 (黄线距离)",
            "feat_left_height": "左侧高度 (妖股基因)",
            "feat_yl_slope": "趋势 (黄线方向)",
            "feat_J": "超跌 (J值)"
        }

        for name, score in sorted_fi:
            print(f"{name:<25} | {score:>8.2f}%   | {interpretations.get(name, '')}")

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

    return df_morph, model, q_full


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

    def audit_model_confidence(model, df_features, case_list):
        print(f"\n====== 🕵️‍♂️ AI 信心分“验尸”报告 ======")
        print(f"{'代码':<10} | {'名称':<8} | {'信号日期':<10} | {'AI评分':<8} | {'真实结果'}")
        print("-" * 80)
    
        # 转换数据格式方便查询
        df_pd = df_features.to_pandas()
        df_pd['date_str'] = df_pd['date'].astype(str)
    
        # 定义必须与训练时一致的特征列
        features = [
            "feat_shadow_ratio", "feat_drop_violence", "feat_vol_shrink_fast", 
            "feat_limit_up_recent", "feat_yl_stickiness", "feat_left_height",
            "feat_yl_slope", "feat_J"
        ]
    
        scores = []
    
        for item in case_list:
            # 定位该股该日的数据
            mask = (df_pd['code'] == item['code']) & (df_pd['date_str'] == item['date'])
            row = df_pd[mask]
        
            if len(row) > 0:
                # 提取特征并让模型打分
                X_input = row[features]
                # predict_proba 返回 [失败概率, 成功概率]，我们取 [1]
                prob = model.predict_proba(X_input)[0][1] 
            
                print(f"{item['code']:<10} | {item['name']:<8} | {item['date']:<10} | {prob:.4f}   | {item['result']}")
                scores.append(prob)
            else:
                print(f"{item['code']:<10} | {item['name']:<8} | {item['date']:<10} | {'MISSING':<8} | 数据缺失")

        print("-" * 80)
        if scores:
            print(f"💡 平均信心分: {sum(scores)/len(scores):.4f}")
            print(f"📈 最高分: {max(scores):.4f}")
            print(f"📉 最低分: {min(scores):.4f}")

    # 运行验尸
    audit_model_confidence(model, df_morph, top10_audit)
    return (audit_model_confidence,)


@app.cell
def _(audit_model_confidence, df_morph, model):
    # ==============================================================================
    # 🎯 验证《十大B1完美图》经典案例
    # ==============================================================================

    # 1. 构造案例清单 (已根据 PDF 提取精准代码与日期)
    # 注意：后缀已适配你的 polars 数据格式 (_SH/_SZ)
    perfect_cases = [
        # --- 完美一：标准缩量与反包 ---
        {"code": "688799_SH", "date": "2025-05-11", "name": "华纳药厂", "result": "完美一(标准)"},
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
def _(pl, q_full):
    q_full.filter(pl.col("code")=="688799_SH")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
