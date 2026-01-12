# QuantLab 量化研究实验室

基于 **B1 选股策略** 的量化交易研究平台，集成了 Python 因子计算和 Rust ECS 回测引擎。

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python 数据层                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ 数据获取    │ →  │ 因子计算    │ →  │ 信号导出    │        │
│  │ baostock   │    │ b1_factors  │    │ export_rust │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                              ↓                                  │
│                     market_data.parquet                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Rust 回测引擎 (Bevy ECS)                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ 买入系统    │ →  │ 卖出系统    │ →  │ 统计系统    │        │
│  │ [开盘]     │    │ [收盘]     │    │ [收盘后]   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│  特性: 分批止盈 | 动态止损 | 仓位管理 | 最大回撤跟踪            │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
QuantLab/
├── backtest-engine/       # 🦀 Rust 回测引擎 (Bevy ECS)
│   ├── src/
│   │   ├── main.rs        # 主程序入口
│   │   ├── components/    # ECS 组件 (Position, ClosedTrade)
│   │   ├── resources/     # ECS 资源 (Portfolio, MarketData, Config)
│   │   └── systems/       # ECS 系统 (买入/卖出/统计)
│   └── Cargo.toml
├── utils/                 # 🐍 Python 工具模块
│   ├── b1_factors_opt.py  # B1 选股因子计算 (V3.0)
│   ├── signal_export.py   # 信号导出 (Parquet)
│   ├── backtest.py        # Python 简易回测
│   └── data_utils.py      # 数据工具
├── notebooks/             # 📓 研究 Notebook (Marimo)
│   └── simple_b1_opt.py   # B1 策略主程序
├── data/                  # 📊 数据目录 (git ignored)
│   └── signals/           # 导出的信号文件
└── strategies/            # 策略脚本
```

## 🚀 快速开始

### 1. Python 环境

```bash
# 使用 uv (推荐)
uv sync

# 或 pip
pip install polars pandas numpy baostock
```

### 2. Rust 环境

```bash
cd backtest-engine
cargo build --release
```

### 3. 运行流程

**Step 1: 计算因子并导出**
```python
from utils import calc_b1_factors_opt, export_for_rust

# 计算 B1 因子
df_signals = calc_b1_factors_opt(df_raw)

# 导出供 Rust 使用
LOOSE_PERIODS = [
    ("2025-04-09", "2025-09-04"),
    ("2026-01-05", "2026-03-31"),
]

export_for_rust(
    df_signals,
    output_path="data/signals/market_data.parquet",
    loose_periods=LOOSE_PERIODS,
    start_date="2025-01-01",
)
```

**Step 2: 运行回测**
```bash
cd backtest-engine
cargo run --release -- --data ../data/signals/market_data.parquet
```

**可选参数:**
```
--capital         初始资金 (默认: 100000)
--max-positions   最大持仓数 (默认: 5)
--stop-loss       止损比例 (默认: 0.03)
--max-hold-days   最大持有天数 (默认: 30)
```

## 📈 回测策略

### 买入条件
- `pre_b1_signal = true` (昨日产生 B1 信号)
- `is_loose = true` (处于活跃期)
- 按 `vol_ratio` 排序 (缩量优先)

### 卖出条件
1. **止损**: 收盘价 ≤ 止损价
2. **分批止盈**:
   - 涨 15% → 卖 1/3
   - 涨 30% → 再卖 1/3
   - 剩余部分跌破 WL 时卖出
3. **超时**: 持有 ≥ 30 天

### 回测指标示例
```
Total Trades: 40
Win Rate: 57.5%
Total Return: +13.68%
Max Drawdown: 3.69%
```

## 🔧 开发说明

### Rust 编译优化

`Cargo.toml` 配置了三种模式:

| 命令 | 编译速度 | 运行速度 | 用途 |
|------|---------|---------|------|
| `cargo run` | ⚡ 最快 | 🐢 较慢 | 日常调试 |
| `cargo run --release` | ⚡ 快 | 🚀 快 | 开发回测 |
| `cargo run --profile production` | 🐢 慢 | 🚀🚀 极致 | 正式回测 |

## 📝 注意事项

- `data/` 目录已被 `.gitignore` 忽略
- Rust 回测引擎需要 `market_data.parquet` 文件
- 建议使用 Marimo 替代 Jupyter 进行交互式研究
