import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    from sklearn.metrics import precision_score, recall_score, f1_score

    # ==============================================================================
    # 1. 配置与天道 (Config)
    # ==============================================================================
    DATA_ROOT = r"../QuantData/Ashare"
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


    # ==============================================================================
    # 2. 数据处理：Polars Turbo 管道 (含 NaN 清洗)
    # ==============================================================================
    def prepare_tensor_data_v8(df_lazy: pl.LazyFrame, seq_len=40):
        print("🚀 [System V8] 启动全量数据构建 (含 NaN 熔断清洗)...")

        # --- A. 指标计算 (Polars Expression) ---
        # 均线系统
        expr_yl = (
            pl.col("close_adj").rolling_mean(14) +
            pl.col("close_adj").rolling_mean(28) +
            pl.col("close_adj").rolling_mean(57) +
            pl.col("close_adj").rolling_mean(114)
        ) / 4

        # KDJ 系统
        expr_low9 = pl.col("low_adj").rolling_min(9)
        expr_high9 = pl.col("high_adj").rolling_max(9)
        expr_rsv_den = expr_high9 - expr_low9
    
        PRED_WINDOW = 10
        STOP_LOSS = -0.03
        TAKE_PROFIT = 0.10

        q = (
            df_lazy.sort(["code", "date"])
            .with_columns([
                pl.col("close_adj").ewm_mean(span=10, adjust=False).alias("WL"),
                expr_yl.alias("YL"),
                # 相对量能 (Vol Ratio)
                (pl.col("volume") / (pl.col("volume").rolling_max(20) + 1e-8)).alias("vol_ratio"),
                # RSV 安全计算
                pl.when(expr_rsv_den == 0)
                  .then(50.0)
                  .otherwise((pl.col("close_adj") - expr_low9)/expr_rsv_den*100)
                  .fill_null(50.0)
                  .alias("RSV_safe")
            ])
            .with_columns([
                pl.col("RSV_safe").ewm_mean(alpha=1/3, adjust=False).over("code").alias("K")
            ])
            .with_columns([
                pl.col("K").ewm_mean(alpha=1/3, adjust=False).over("code").alias("D")
            ])
            .with_columns([
                (3 * pl.col("K") - 2 * pl.col("D")).alias("J"),
                ((pl.col("close_adj") - pl.col("YL")) / pl.col("YL")).alias("dist_yl")
            ])
        )
    
        print("⚡ [1/4] 计算全量指标...")
        df_all = q.collect()
    
        # --- B. 标签与筛选掩码 ---
        print("⚡ [2/4] 预计算标签与掩码...")
        df_labeled = df_all.with_columns([
            pl.col("high_adj").rolling_max(PRED_WINDOW).shift(-PRED_WINDOW).over("code").alias("fut_max"),
            pl.col("low_adj").rolling_min(PRED_WINDOW).shift(-PRED_WINDOW).over("code").alias("fut_min"),
        ])
    
        # 硬筛选条件：J<20 & 缩量 & 回踩不破位
        # .fill_null(False) 是防止 Null 导致位运算报错的关键
        df_labeled = df_labeled.with_columns([
            (
                (pl.col("J") < 20) & 
                (pl.col("vol_ratio") < 0.7) & 
                (pl.col("dist_yl") > -0.05)
            ).fill_null(False).alias("is_candidate") 
        ])

        # --- C. 向量化切片 (Vectorized Unfold) ---
        feature_cols = ["open_adj", "high_adj", "low_adj", "close_adj", "vol_ratio", "J", "dist_yl"]
        X_list = []
        y_list = []
        meta_list = [] # (code, date)

        print("⚡ [3/4] 启动 Turbo 切片引擎...")
    
        for group in tqdm(df_labeled.partition_by("code", include_key=True), desc="Slicing"):
            if len(group) < seq_len + PRED_WINDOW: continue
        
            # 1. 提取原始数据 (N, 7)
            data_np = group.select(feature_cols).to_numpy()
        
            # 🔥🔥🔥 [关键修复 1] 彻底清洗 NaN/Inf，防止模型暴毙 🔥🔥🔥
            # 即使当天数据有效，回溯40天也可能遇到上市前停牌等导致的 NaN
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)
        
            metrics_tensor = torch.from_numpy(data_np).float()
        
            # 2. 标签生成
            curr_price = data_np[:, 3] # Close
            fut_max = group["fut_max"].to_numpy() # 含 NaN (最后几天)
            fut_min = group["fut_min"].to_numpy() # 含 NaN
        
            # 避免除以0
            curr_price = np.where(curr_price == 0, 1e-8, curr_price)
        
            # 忽略 runtime warning (因为最后几天是 NaN)
            with np.errstate(invalid='ignore'):
                r_high = (fut_max - curr_price) / curr_price
                r_low = (fut_min - curr_price) / curr_price
            
                labels = np.full(len(group), -1, dtype=np.int8)
                mask_win = r_high >= TAKE_PROFIT
                mask_loss = r_low <= STOP_LOSS
            
                # 简化逻辑：只要摸到涨停板就算赢，先跌破止损算输
                labels[mask_win] = 1
                labels[mask_loss] = 0
            
            # 3. 核心：Unfold 滑动窗口
            # (N, 7) -> (N-seq_len+1, 7, seq_len)
            windows = metrics_tensor.unfold(0, seq_len, 1).permute(0, 1, 2)
        
            # 4. 归一化 (Normalization)
            # 取每个窗口第0天的 Close 作为基准
            base_prices = windows[:, 3, 0].unsqueeze(1).unsqueeze(2)
        
            # 🔥🔥🔥 [关键修复 2] 防止基准价为 0 导致除以 0 🔥🔥🔥
            base_prices = torch.where(base_prices == 0, torch.tensor(1.0), base_prices)
        
            # OHLC (前4列) 归一化
            norm_ohlc = (windows[:, :4, :] / base_prices) - 1.0
        
            # 其他指标
            feat_others = windows[:, 4:, :].clone()
            feat_others[:, 1, :] /= 100.0 # J 值归一化
        
            final_X = torch.cat([norm_ohlc, feat_others], dim=1) # (Num_Windows, 7, 40)
        
            # 5. 筛选有效样本
            full_mask = group["is_candidate"].to_numpy()
        
            # 对齐：取窗口最后一天作为判定点
            valid_mask_indices = full_mask[seq_len-1:]
            valid_label_indices = labels[seq_len-1:]
        
            # 最终选择器
            final_selector = (valid_mask_indices) & (valid_label_indices != -1)
        
            if final_selector.sum() > 0:
                selected_X = final_X[torch.tensor(final_selector)]
                selected_y = torch.tensor(valid_label_indices[final_selector]).float()
            
                X_list.append(selected_X)
                y_list.append(selected_y)
            
                # 记录元数据
                c = group[0, "code"]
                raw_dates = group["date"].to_numpy()
                target_dates = raw_dates[seq_len-1:][final_selector]
                for d in target_dates:
                    meta_list.append((c, d))

        # --- D. 合并 ---
        print("⚡ [4/4] 合并张量...")
        if len(X_list) > 0:
            X = torch.cat(X_list, dim=0).numpy()
            y = torch.cat(y_list, dim=0).numpy()
        else:
            X, y = np.array([]), np.array([])
        
        print(f"✅ 数据准备完毕: {X.shape}")
        print(f"   正样本比例: {np.mean(y==1):.2%}")
    
        return X, y, meta_list

    # ==============================================================================
    # 3. 模型定义 (无 Sigmoid)
    # ==============================================================================
    class ZtalkResNet_V2(nn.Module):
        def __init__(self, in_channels=7):
            super(ZtalkResNet_V2, self).__init__()
            # 1. 视觉感知
            self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            # 2. 残差提取
            self.layer1 = self._make_layer(64, 64, stride=1)
            self.layer2 = self._make_layer(64, 128, stride=2)
            self.layer3 = self._make_layer(128, 256, stride=2)

            # 3. 决策 (输出 Logits)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(256, 1)
            # ❌ 注意：这里删除了 Sigmoid，配合 BCEWithLogitsLoss 使用

        def _make_layer(self, in_c, out_c, stride):
            downsample = None
            if stride != 1 or in_c != out_c:
                downsample = nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_c),
                )
            return nn.Sequential(ResidualBlock(in_c, out_c, stride, downsample))

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x) # 输出 Logits
            return x

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.downsample = downsample

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

    # ==============================================================================
    # 4. 数据集与训练 (Dataset & Training)
    # ==============================================================================
    from data_utils import B1Dataset

    def train_deep_model_v8(X, y, meta_list):
        print("🚀 [Training V8] 启动深度学习训练 (含自动加权与梯度裁剪)...")

        # --- 1. 数据切分 ---
        all_dates = sorted(list(set([m[1] for m in meta_list])))
        split_idx = int(len(all_dates) * 0.8) # 80% 训练
        split_date = all_dates[split_idx]
        print(f"✂️ 切分日期: {split_date}")

        dates_series = pd.to_datetime([m[1] for m in meta_list])
        train_mask = dates_series < pd.Timestamp(split_date)
        test_mask = dates_series >= pd.Timestamp(split_date)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # --- 2. 自动计算正样本权重 (关键！) ---
        # 既然正样本只有 17%，我们需要给它 4-5 倍的关注度
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight_val = num_neg / (num_pos + 1e-5)
        print(f"⚖️ [Class Imbalance] 正样本权重: {pos_weight_val:.2f} (拒绝模型躺平)")

        # --- 3. DataLoader ---
        batch_size = 512
        train_ds = B1Dataset(X_train, y_train)
        test_ds = B1Dataset(X_test, y_test)
    
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # --- 4. 初始化 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ZtalkResNet_V2(in_channels=7).to(device)
    
        # 🔥🔥🔥 [关键修复 3] 带权重的 Loss 🔥🔥🔥
        pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        # --- 5. 训练循环 ---
        epochs = 10 # 增加轮数，让模型有时间学
        best_f1 = 0
    
        for epoch in range(epochs):
            model.train()
            train_loss = 0
        
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs) # Logits
                loss = criterion(outputs, targets)
                loss.backward()
            
                # 🔥🔥🔥 [关键修复 4] 梯度裁剪 (防止梯度爆炸) 🔥🔥🔥
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # --- 验证 (看 Precision/Recall) ---
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
        
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += loss.item()
                
                    # Logits -> Probs -> 0/1
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float().cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
        
            # 计算真实指标
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"   >>> 胜率 (Precision): {precision:.2%} (信号质量)")
            print(f"   >>> 召回 (Recall):    {recall:.2%}    (抓机会能力)")
            print(f"   >>> F1 Score:         {f1:.4f}")
        
            scheduler.step(avg_val_loss)
        
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "best_ztalk_resnet.pth")
                print("   🌟 New Best Model Saved!")

        print("✅ 训练完成。")
        return model, X_test, y_test


    return (
        ZtalkResNet_V2,
        load_data,
        np,
        pd,
        prepare_tensor_data_v8,
        torch,
        train_deep_model_v8,
    )


@app.cell
def _(load_data, prepare_tensor_data_v8, train_deep_model_v8):
    # ==============================================================================
    # 🎯 主执行逻辑
    # ==============================================================================
    # 1. 加载
    df_lazy = load_data()
    # 2. 预处理 (含市值过滤 < 40亿)
    X, y, meta_list = prepare_tensor_data_v8(df_lazy)
    # 3. 训练
    model, X_test, y_test, all_meta, test_mask = train_deep_model_v8(X, y, meta_list)
    return X_test, all_meta, test_mask, y_test


@app.cell
def _(X_test, ZtalkResNet_V2, all_meta, np, pd, test_mask, torch, y_test):
    # ==============================================================================
    # 7. 加载最佳模型进行预测 (Load Best Model)
    # ==============================================================================
    print("📂 正在加载最佳模型权重 (best_resnet_model.pth)...")

    # 1. 重新实例化一个空模型结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = ZtalkResNet_V2().to(device)

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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
