import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import os
    import gc
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import plotly.graph_objects as go
    import plotly.express as px
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from plotly.subplots import make_subplots
    from datetime import datetime
    # 引入深度学习库
    return DataLoader, go, nn, np, optim, os, pd, pl, torch, tqdm


@app.cell
def _(os, pl):
    # ==============================================================================
    # 1. 配置与天道 (Config & Regime)
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"

    # Ztalk 体系核心“天道” (活跃市值多头区域)
    # 只在这些好日子里训练，培养模型的“牛市思维”
    MANUAL_LOOSE_PERIODS = [
        ("2019-02-11", "2019-04-10"),
        ("2019-12-16", "2020-03-02"),
        ("2020-06-19", "2020-07-15"),
        ("2020-12-24", "2021-01-25"),
        ("2021-04-16", "2021-09-14"),
        ("2022-04-27", "2022-07-05"),
        ("2023-01-15", "2023-04-15"),
        ("2024-02-06", "2024-03-20"),
        ("2024-09-24", "2024-10-15"),
        ("2025-04-08", "2025-09-04"), 
    ]

    # ==============================================================================
    # 2. 数据加载 (Data Loading)
    # ==============================================================================
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
    return MANUAL_LOOSE_PERIODS, load_data


@app.cell
def _(MANUAL_LOOSE_PERIODS, np, pl):
    # ==============================================================================
    # 3. 数据预处理与切片 (Preprocessing & Slicing)
    # ==============================================================================
    def prepare_tensor_data(df_lazy: pl.LazyFrame, seq_len=40):
        print("🛠️ [Preprocessing V5.2] 修正数据泄漏，构建真正的精英数据集...")

        # --- A. 天道 (不变) ---
        regime_expr = pl.lit(False)
        for start, end in MANUAL_LOOSE_PERIODS:
            regime_expr = regime_expr | (
                (pl.col("date") >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d")) & 
                (pl.col("date") <= pl.lit(end).str.strptime(pl.Date, "%Y-%m-%d"))
            )

        # --- B. 清洗与 Label 定义 (🔥 核心修正) ---
        q = df_lazy.filter(
            (pl.col("market_cap_100m") >= 40) 
        ).sort(["code", "date"]).with_columns([
            # 1. 均线
            pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").alias("WL"),
            ((pl.col("close_adj").rolling_mean(14).over("code") + 
              pl.col("close_adj").rolling_mean(28).over("code") + 
              pl.col("close_adj").rolling_mean(57).over("code") + 
              pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),

            # 2. 天道
            regime_expr.cast(pl.Int32).alias("market_regime"),

            # 3. 🔥 V5.2 修正未来窗口: [T+1 ... T+10]
            (pl.col("high_adj").shift(-10).rolling_max(10).over("code") / pl.col("close_adj") - 1).alias("future_max_ret"),
        ]).with_columns([
            # 4. 标签定义 (含扩大负样本范围)
            pl.when(pl.col("future_max_ret") > 0.10).then(pl.lit(1))   # 赢: > 10%
              .when(pl.col("future_max_ret") <= 0.05).then(pl.lit(0))  # 输: <= 5% (包含平庸)
              .otherwise(pl.lit(-1))                                   # 灰度: 5%-10% (扔掉)
              .alias("label")
        ])

        print("⚡ [Tensor Building] 正在切片...")
        required_cols = ["code", "date", "close_adj", "volume", "WL", "YL", "market_regime", "label"]
        df = q.select(required_cols).drop_nulls().collect().to_pandas()

        X_list = []
        y_list = []
        meta_list = [] 

        groups = df.groupby('code')
        total_codes = len(groups)

        for i, (code, group) in enumerate(groups):
            if len(group) < seq_len + 15: continue # 多留点余量给 future shift

            closes = group['close_adj'].values
            vols = group['volume'].values
            wls = group['WL'].values
            yls = group['YL'].values
            regimes = group['market_regime'].values
            labels = group['label'].values
            dates = group['date'].values

            valid_indices = np.where(regimes[seq_len-1:] == 1)[0] + (seq_len - 1)

            for idx in valid_indices:
                if labels[idx] == -1: continue

                s_start = idx - seq_len + 1
                s_end = idx + 1

                # V4 相对归一化
                c_seg = closes[s_start:s_end]
                v_seg = vols[s_start:s_end]
                w_seg = wls[s_start:s_end]
                y_seg = yls[s_start:s_end]

                base_price = c_seg[0] + 1e-6 
                c_norm = (c_seg / base_price) - 1.0
                w_norm = (w_seg / base_price) - 1.0
                y_norm = (y_seg / base_price) - 1.0
                v_norm = v_seg / (v_seg.max() + 1e-6)

                sample = np.stack([c_norm, v_norm, w_norm, y_norm], axis=0)

                X_list.append(sample)
                y_list.append(labels[idx])
                meta_list.append((code, dates[idx]))

            if i % 1000 == 0: print(f"   已处理 {i}/{total_codes} 只股票...")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"✅ 数据准备完毕! 样本形状: {X.shape}")
        print(f"   正样本占比: {np.mean(y == 1):.2%}") 

        return X, y, meta_list

    # ==============================================================================
    # 4. 数据集类 (PyTorch Dataset)
    # ==============================================================================
    # class B1Dataset(Dataset):
    #     def __init__(self, X, y):
    #         self.X = torch.from_numpy(X)
    #         self.y = torch.from_numpy(y)

    #     def __len__(self):
    #         return len(self.y)

    #     def __getitem__(self, idx):
    #         return self.X[idx], self.y[idx]

    from data_utils import B1Dataset
    return B1Dataset, prepare_tensor_data


@app.cell
def _(nn, torch):
    # ==============================================================================
    # 5. 模型定义: 1D ResNet (Morphology Extractor)
    # ==============================================================================
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)

            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class ZtalkResNet(nn.Module):
        def __init__(self, num_channels=4, seq_len=40):
            super(ZtalkResNet, self).__init__()

            # 初始层：大感受野，捕捉大轮廓
            self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            # 残差层：捕捉微观形态
            self.layer1 = ResidualBlock(64, 64)
            self.layer2 = ResidualBlock(64, 128, stride=2)
            self.layer3 = ResidualBlock(128, 256, stride=2)

            # 输出层
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(p=0.5) # 🔥 50% 的概率丢弃神经元，防止死记硬背
            self.fc = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (Batch, 4, 40)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return self.sigmoid(x)
    return (ZtalkResNet,)


@app.cell
def _(B1Dataset, DataLoader, ZtalkResNet, nn, optim, pd, torch, tqdm):
    # ==============================================================================
    # 6. 训练流程 (Training Execution)
    # ==============================================================================
    def train_deep_model(X, y, meta_list):
        print("🚀 [Training] 启动 PyTorch 深度学习训练...")

        # --- A. 严格时间切分 ---
        # meta_list 包含 [(code, date), ...]
        # 我们取出所有日期，按 75% 切分
        all_dates = sorted(list(set([m[1] for m in meta_list])))
        split_idx = int(len(all_dates) * 0.75)
        split_date = all_dates[split_idx]

        print(f"✂️ 切分日期: {split_date}")

        # 转换为 pandas series 方便 mask
        dates_series = pd.to_datetime([m[1] for m in meta_list])
        train_mask = dates_series < pd.Timestamp(split_date)
        test_mask = dates_series >= pd.Timestamp(split_date)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # --- B. DataLoader ---
        batch_size = 512
        train_ds = B1Dataset(X_train, y_train)
        test_ds = B1Dataset(X_test, y_test)

        train_loader = DataLoader(train_ds, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=2,
                                  pin_memory=True,
                                 persistent_workers=True)
        test_loader = DataLoader(test_ds, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=2,
                                 pin_memory=True,
                                persistent_workers=True)

        print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")

        # --- C. 初始化模型 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 计算设备: {device}")

        model = ZtalkResNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # --- D. 训练循环 ---
        best_loss = float('inf')
        epochs = 5 # 深度学习需要多跑几轮

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1) # (Batch, 1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            # 验证
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets.unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            acc = correct / total

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2%}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "best_resnet_model.pth")

        print("✅ 训练完成。")
        return model, X_test, y_test, meta_list, test_mask
    return (train_deep_model,)


@app.cell
def _(load_data, prepare_tensor_data, train_deep_model):
    # ==============================================================================
    # 🎯 主执行逻辑
    # ==============================================================================
    # 1. 加载
    df_lazy = load_data()
    # 2. 预处理 (含市值过滤 < 40亿)
    X, y, meta_list = prepare_tensor_data(df_lazy)
    # 3. 训练
    model, X_test, y_test, all_meta, test_mask = train_deep_model(X, y, meta_list)
    return X_test, all_meta, df_lazy, test_mask, y_test


@app.cell
def _(X_test, ZtalkResNet, all_meta, np, pd, test_mask, torch, y_test):
    # ==============================================================================
    # 7. 加载最佳模型进行预测 (Load Best Model)
    # ==============================================================================
    print("📂 正在加载最佳模型权重 (best_resnet_model.pth)...")

    # 1. 重新实例化一个空模型结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = ZtalkResNet().to(device)

    # 2. 加载权重
    try:
        best_model.load_state_dict(torch.load("best_resnet_model.pth"))
        print("✅ 权重加载成功！")
    except FileNotFoundError:
        print("❌ 找不到模型文件，请检查是否训练成功并保存了 best_resnet_model.pth")

    # 3. 开启评估模式 (关闭 Dropout)
    best_model.eval()

    # 4. 预测测试集
    # 假设 X_test 还保留在内存中
    if 'X_test' in locals():
        test_tensor = torch.from_numpy(X_test).to(device)

        with torch.no_grad():
            # 预测分数
            scores = best_model(test_tensor).cpu().numpy().flatten()

        # 5. 组合结果
        meta_array = np.array(all_meta, dtype=object)
        test_meta = meta_array[test_mask]

        df_best = pd.DataFrame({
            "code": test_meta[:, 0],
            "date": test_meta[:, 1],
            "label": y_test,
            "score": scores
        })

        # 6. 展示 Top Picks
        print("\n🏆 [Best Model (Epoch 3) Top Picks]")
        top_picks = df_best.sort_values("score", ascending=False).head(10)
        print(top_picks)
    else:
        print("⚠️ 内存中找不到 X_test，请重新运行数据准备步骤。")
    return (best_model,)


@app.cell
def _(best_model, df_lazy, np, pd, pl, torch):
    # ==============================================================================
    # 9. 终极图灵测试：给 10 大完美案例打分 (修正版)
    # ==============================================================================
    PERFECT_CASES_CONFIG = [
        {"code": "688799_SH", "date": "2025-05-12", "name": "华纳药厂"},
        {"code": "300689_SZ", "date": "2025-07-18", "name": "澄天伟业"},
        {"code": "600601_SH", "date": "2025-07-23", "name": "方正科技"},
        {"code": "688321_SH", "date": "2025-06-20", "name": "微芯生物"},
        {"code": "002940_SZ", "date": "2025-07-11", "name": "昂利康"},
        {"code": "301076_SZ", "date": "2025-08-01", "name": "新瀚新材"},
        {"code": "600184_SH", "date": "2025-07-10", "name": "光电股份"},
        {"code": "002074_SZ", "date": "2025-08-01", "name": "国轩高科"},
        {"code": "605378_SH", "date": "2025-07-31", "name": "野马电池"},
        {"code": "600366_SH", "date": "2025-08-06", "name": "宁波韵升"}
    ]

    def check_perfect_cases_score(df_lazy, model, seq_len=40):
        print("👨‍🏫 [Teacher Check] 正在计算特征并回测 10 大完美案例...")

        target_codes = [c['code'] for c in PERFECT_CASES_CONFIG]

        # 1. 补全均线系统 (必须与训练时逻辑一致)
        df_pool = (
            df_lazy
            .filter(pl.col("code").is_in(target_codes)) # 只取这几只票
            .sort(["code", "date"])
            .with_columns([
                pl.col("close_adj").ewm_mean(span=10, adjust=False).over("code").alias("WL"),

                ((pl.col("close_adj").rolling_mean(14).over("code") + 
                  pl.col("close_adj").rolling_mean(28).over("code") + 
                  pl.col("close_adj").rolling_mean(57).over("code") + 
                  pl.col("close_adj").rolling_mean(114).over("code")) / 4).alias("YL"),
            ])
            .collect()
            .to_pandas()
        )

        # 2. 逐个提取并打分
        results = []

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"📊 提取完成，开始 AI 阅卷 (Device: {device})...")

        for case in PERFECT_CASES_CONFIG:
            code = case['code']
            target_date = case['date']

            # 定位索引
            idx_list = df_pool.index[
                (df_pool["code"] == code) & 
                (df_pool["date"].astype(str) == target_date)
            ].tolist()

            if not idx_list:
                print(f"⚠️ 缺失数据: {case['name']} {code} {target_date}")
                continue

            idx = idx_list[0]

            # 确保有足够历史数据
            start_idx = idx - seq_len + 1
            if start_idx < 0 or df_pool.iloc[start_idx]['code'] != code:
                print(f"⚠️ 历史数据不足: {case['name']}")
                continue

            # 提取切片
            segment = df_pool.iloc[start_idx : idx + 1]

            c_seg = segment['close_adj'].values
            v_seg = segment['volume'].values
            w_seg = segment['WL'].values 
            y_seg = segment['YL'].values

            # -------------------------------------------------------
            # 🔥 核心修正：放弃 Z-Score，改用 相对基准归一化 (V4)
            # -------------------------------------------------------

            # 1. 价格归一化 (相对于切片起点的涨跌幅)
            # 以该窗口第1天(c_seg[0])的收盘价为基准
            base_price = c_seg[0] + 1e-6 

            # 这样，如果不动，值就是 0；如果涨 10%，值就是 0.1
            # 这种归一化方式完美保留了"波动率"的真实大小
            c_norm = (c_seg / base_price) - 1.0
            w_norm = (w_seg / base_price) - 1.0
            y_norm = (y_seg / base_price) - 1.0

            # 2. 量能归一化 (保持原样，除以窗口最大值)
            v_norm = v_seg / (v_seg.max() + 1e-6)

            # (V4版不需要 clip 到 -3, 3，因为相对涨跌幅通常在 -0.5 到 0.5 之间，是自然的数值)

            # 3. 堆叠
            sample = np.stack([c_norm, v_norm, w_norm, y_norm], axis=0)
            tensor_in = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                score = model(tensor_in).item()

            results.append({
                "name": case['name'],
                "code": code,
                "date": target_date,
                "AI_Score": score
            })

        # 3. 展示结果
        res_df = pd.DataFrame(results).sort_values("AI_Score", ascending=False)
        print("\n📝 10大完美案例 AI 评分表:")
        print(res_df)

        if len(res_df) > 0:
            avg_score = res_df["AI_Score"].mean()
            print(f"\n📊 平均得分: {avg_score:.4f}")

            if avg_score > 0.85:
                print("✅ 结论: 模型精准捕捉到了祖师爷的神韵！")
            elif avg_score > 0.7:
                print("⚠️ 结论: 模型学了个大概，但有些案例它没看懂。")
            else:
                print("❌ 结论: 模型根本不知道什么是完美案例，训练失败。")

        return res_df

    # ==========================================
    # 🎯 运行检测
    # ==========================================
    res_df = check_perfect_cases_score(df_lazy, best_model)
    return (res_df,)


@app.cell
def _(X_test, best_model, go, np, res_df, torch):
    import plotly.figure_factory as ff
    def analyze_score_distribution_plotly(model, X_test, perfect_scores):
        print("⚖️ [Calibration Check] 正在扫描全市场（测试集）的评分分布...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        # --- 1. 批量预测全市场分数 ---
        batch_size = 4096 # 4050 显卡可以开大点，加快速度
        test_scores = []

        # 转为 Tensor
        X_tensor = torch.from_numpy(X_test).float()

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i : i+batch_size].to(device)
                # 预测并转回 CPU numpy
                batch_scores = model(batch_X).cpu().numpy().flatten()
                test_scores.extend(batch_scores)

        test_scores = np.array(test_scores)

        # --- 2. 统计核心指标 ---
        market_median = np.median(test_scores)
        market_mean = np.mean(test_scores)
        market_top10 = np.percentile(test_scores, 90) # 前 10% 的门槛
        market_top1 = np.percentile(test_scores, 99)  # 前 1% 的门槛

        perfect_avg = np.mean(perfect_scores)

        print("\n📊 [市场 vs 完美案例 统计数据]:")
        print(f"🔹 全市场中位数: {market_median:.4f}")
        print(f"🔹 全市场平均分: {market_mean:.4f}")
        print(f"🔹 市场前 10%% 门槛: > {market_top10:.4f}")
        print(f"🔹 市场前 1%% 门槛:  > {market_top1:.4f}")
        print("-" * 30)
        print(f"🔸 完美案例平均分: {perfect_avg:.4f}")

        # 计算完美案例平均分在市场中的排位
        percentile_rank = (test_scores < perfect_avg).mean() * 100
        print(f"🏆 完美案例的平均分击败了 {percentile_rank:.2f}% 的市场样本")

        # --- 3. Plotly 可视化 ---
        fig = go.Figure()

        # A. 全市场直方图 (背景)
        fig.add_trace(go.Histogram(
            x=test_scores,
            nbinsx=200, # 200个柱子，看精细度
            name='全市场样本分布',
            marker_color='rgba(100, 149, 237, 0.6)', # 浅蓝色半透明
            histnorm='probability', # 显示概率密度
        ))

        # B. 完美案例散点 (红色钻石)
        # y轴设为 0 或者稍微高一点点，为了显示在底部
        fig.add_trace(go.Scatter(
            x=perfect_scores,
            y=[0] * len(perfect_scores), # 放在底部
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='10大完美案例 (你的 B1)',
            text=[f"{s:.4f}" for s in perfect_scores],
            hoverinfo='x+name'
        ))

        # C. 关键线 (中位数 vs 完美均值)
        # 市场中位数
        fig.add_vline(x=market_median, line_width=2, line_dash="dash", line_color="blue", annotation_text=f"市场中位数:{market_median:.2f}")
        # 完美案例均值
        fig.add_vline(x=perfect_avg, line_width=3, line_color="red", annotation_text=f"完美均值:{perfect_avg:.2f}")

        fig.update_layout(
            title="⚔️ 巅峰对决: 全市场评分分布 vs 完美案例位置",
            xaxis_title="AI 评分 (0-1)",
            yaxis_title="样本占比 (概率)",
            template="plotly_white",
            bargap=0.1,
            height=500
        )

        fig.show()

    # ==========================================
    # 🎯 运行
    # ==========================================
    perfect_scores_list = res_df["AI_Score"].values
    analyze_score_distribution_plotly(best_model, X_test, perfect_scores_list)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
