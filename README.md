# QuantLab 量化研究实验室

基于 **B1 选股策略** 的量化交易研究平台，集成了 Python 因子计算和 Rust ECS 回测引擎。

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python 数据层                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ 数据获取    │ →  │ 因子计算    │ →  │ 信号导出    │        │
│  │ qmt/baostock│    │ b1_factors  │    │ export_rust │        │
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
│  特性: 动态仓位(复利) | 分批止盈 | 交易成本 | 最大回撤跟踪     │
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
│   ├── config.toml        # 策略配置文件
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

## ⚙️ 配置文件

回测引擎通过 `config.toml` 进行配置，无需重新编译即可调参：

```toml
[backtest]
initial_capital = 100000.0  # 初始资金
max_positions = 5           # 最大持仓数量
max_daily_buys = 5          # 每天最多买入数量
position_size_pct = 0.2     # 单只股票目标仓位 = 总资产 × 20% (动态复利)
max_hold_days = 30          # 最大持有天数
start_date = "2025-01-05"   # 回测开始日期 (可选)
end_date = "2026-01-15"     # 回测结束日期 (可选)
sort_field = "vol_ratio"    # 买入排序字段 (缩量优先)
sort_ascending = true       # 升序排序

[stop_loss]
enabled = true
pct = 0.03                  # 止损比例 (3%)

[take_profit]
tp1_pct = 0.15              # 第一阶段止盈 (15%)
tp2_pct = 0.30              # 第二阶段止盈 (30%)
sell_ratio = 0.333          # 每阶段卖出比例 (1/3)
sell_on_break_wl = true     # Stage2 跌破 WL 清仓

[weak_performance]
enabled = true
days = 10                   # N 天后检查
min_gain_pct = 0.05         # 最低涨幅要求 (5%)

[trailing_stop]
enabled = false             # 移动止损 (可选)
activation_pct = 0.10
trailing_pct = 0.05

[costs]
commission_rate = 0.00025   # 佣金 (万2.5)
stamp_duty_rate = 0.001     # 印花税 (千1，仅卖出)
slippage_pct = 0.001        # 滑点 (0.1%)
```

## 📈 回测策略

### 买入条件
- `pre_b1_signal = true` (昨日产生 B1 信号)
- `is_loose = true` (处于活跃期)
- 按 `vol_ratio` 排序 (缩量优先)
- 每日最多买入 `max_daily_buys` 只

### 仓位计算 (动态复利)
```
目标仓位 = (现金 + 持仓市值) × position_size_pct
持仓市值 = Σ(持仓股数 × 昨日收盘价)
```
- 账户盈利 → 新仓位自动变大 (复利效应)
- 账户亏损 → 新仓位自动变小 (风险控制)
- 现金略不足 → 尽量买 (至少目标的 50%)

### 卖出条件 (按优先级)
1. **止损**: 收盘价跌幅 ≥ 3%
2. **移动止损**: 涨幅达激活点后，从最高点回撤超阈值 (可选)
3. **弱势清仓**: 10 天后涨幅不足 5%
4. **分批止盈**:
   - 涨 15% → 卖 1/3 (Stage 1)
   - 涨 30% → 再卖 1/3 (Stage 2)
   - Stage 2 后跌破 WL → 卖出剩余
5. **超时**: 持有 ≥ 30 天

### 回测指标 (2025.01 - 2026.01)

```
========================================
           Backtest Results
========================================
Total Trades: ~65
Win Rate: ~57%
Total Return: +15% ~ +18%  (动态仓位复利效应)
Max Drawdown: ~5%
----------------------------------------
Trading Costs: ~2500 (佣金+印花税+滑点)
========================================
```

> 注：由于动态仓位机制，盈利时仓位自动放大，收益比固定仓位更高。

## 💡 策略容量发现

回测显示该策略存在 **容量天花板**：

| 资金规模 | 持仓数 | 收益率 | 说明 |
|----------|--------|--------|------|
| 10万 | 5只 | **+15~18%** | ✅ 最优 (动态复利) |
| 50万 | 10只 | +6~8% | 信号稀释 |

**结论**: 策略最佳规模为 **10-30万**，资金过大会导致收益稀释。

## 🔧 开发说明

### Rust 编译优化

`Cargo.toml` 配置了三种模式:

| 命令 | 编译速度 | 运行速度 | 用途 |
|------|---------|---------|------|
| `cargo run` | ⚡ 最快 | 🐢 较慢 | 日常调试 |
| `cargo run --release` | ⚡ 快 | 🚀 快 | 开发回测 |
| `cargo run --profile production` | 🐢 慢 | 🚀🚀 极致 | 正式回测 |

### 为什么用 Bevy ECS？

- **模块化**: 买入/卖出/统计逻辑完全解耦
- **高性能**: Rust + ECS 架构，毫秒级回测
- **可扩展**: 新增卖出条件只需添加 System

## 📝 注意事项

- `data/` 目录已被 `.gitignore` 忽略
- Rust 回测引擎需要 `market_data.parquet` 文件
- 建议使用 Marimo 替代 Jupyter 进行交互式研究
- 配置文件修改后无需重新编译
