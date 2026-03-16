# QuantLab 量化研究实验室

基于 **B1 超跌反转策略** 的量化交易研究平台，集成了 Polars 因子计算、多周期 MACD 过滤、统计学回测、Rust ECS 回测引擎，以及 AI 多模态评审 Agent。

## 架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Python 数据层                                    │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────┐              │
│  │ 数据获取     │ → │ 因子计算       │ → │ 信号输出     │              │
│  │ QMT/Baostock │   │ B1 / Renko     │   │ Parquet/JSON │              │
│  └──────────────┘   └────────────────┘   └──────────────┘              │
│                              ↓                                           │
│               calc_b1_factors_wmacd (周线MACD+WL>YL)                    │
└──────────────────────────────────────────────────────────────────────────┘
            ↓                                    ↓
┌────────────────────────┐       ┌─────────────────────────────────────────┐
│  Rust 回测引擎 (ECS)    │       │  AI Agent 工作流                        │
│  动态仓位 | 分批止盈    │       │  Plotly图表 → Gemini多模态评审 → 推荐   │
│  交易成本 | 最大回撤     │       │  结构化指标 + 视觉分析双通道            │
└────────────────────────┘       └─────────────────────────────────────────┘
```

## 目录结构

```
QuantLab/
├── utils/                     # Python 核心工具模块
│   ├── b1_factors_opt.py      #   B1 选股因子 (V3.0 + 周月MACD + 周线WL>YL)
│   ├── renko_factors.py       #   砖型图反转因子
│   ├── backtest.py            #   统计学回测 (run_backtest / run_backtest_short)
│   ├── signal_export.py       #   信号导出 (Parquet, 供 Rust 使用)
│   ├── duckdb_utils.py        #   DuckDB 数据加载 (load_daily_data_full)
│   ├── get_data.py            #   QMT 数据同步 → DuckDB
│   ├── baostock_sync.py       #   Baostock 数据同步 → DuckDB
│   └── ...
│
├── agent/                     # AI Agent 每日选股工作流
│   ├── run.py                 #   主入口: python -m agent.run
│   ├── chart.py               #   Plotly K线图生成 + PNG导出
│   ├── context.py             #   结构化指标文本 (供LLM阅读)
│   ├── prompt.md              #   B1专用评审提示词
│   ├── config.yaml            #   配置 (数据库/模型/参数)
│   ├── report.py              #   终端输出 + JSON保存
│   └── reviewers/             #   AI评审模块 (可插拔)
│       ├── base.py            #     BaseReviewer 抽象基类
│       └── gemini.py          #     GeminiReviewer (Gemini多模态)
│
├── backtest-engine/           # Rust 回测引擎 (Bevy ECS)
│   ├── src/
│   │   ├── main.rs
│   │   ├── components/        #   ECS 组件 (Position, ClosedTrade)
│   │   ├── resources/         #   ECS 资源 (Portfolio, MarketData, Config)
│   │   └── systems/           #   ECS 系统 (买入/卖出/统计)
│   └── config.toml            #   策略配置 (仓位/止盈止损/成本)
│
├── notebooks/                 # Marimo 研究 Notebook
│   ├── simple_b1_opt.py       #   B1 策略主程序 (因子计算+回测)
│   ├── simple_renko.py        #   砖型图策略研究
│   ├── stock_viewer.py        #   单股K线 + WL/YL 看盘
│   ├── smart_b1_*.py          #   CatBoost 形态学模型
│   ├── sequence_b1_*.py       #   序列模型
│   └── ...
│
├── strategies/                # 通达信 TDX 策略脚本
│   └── tdx_scripts/
│       ├── B1代码3.0.0b.txt   #   B1 日线选股 (当前版本)
│       ├── B1周线过滤器.txt    #   B1 周线过滤器 (MACD+WL>YL)
│       ├── B1大周期择时.txt    #   周线MA多头排列择时
│       └── ...
│
└── data/                      # 数据目录 (git ignored)
    ├── signals/               #   导出的信号 Parquet
    ├── charts/                #   Agent 导出的K线图
    └── review/                #   Agent AI评审结果
```

## 快速开始

### 1. Python 环境

```bash
uv sync
# 或
pip install polars duckdb plotly kaleido pyyaml google-genai
```

### 2. 研究流程 (Notebook)

```python
from utils import load_daily_data_full, calc_b1_factors_wmacd, run_backtest, print_backtest_report

conn = duckdb.connect("path/to/qmt_data.duckdb", read_only=True)
df = load_daily_data_full(conn)

# B1 因子计算 (V3.0 + 周月MACD + 周线WL>YL)
df_signals = calc_b1_factors_wmacd(df, config={"WEEKLY_WL_YL_FILTER": True})

# 统计学回测
df_result = run_backtest(df_signals, return_days=[5, 10, 15, 20, 25, 30])
print_backtest_report(df_result, [5, 10, 15, 20, 25, 30])
```

### 3. AI Agent 每日选股

```bash
# 完整流程: 数据加载 → 因子计算 → 图表导出 → AI评审 → 推荐
python -m agent.run

# 指定日期
python -m agent.run --date 2026-02-24

# 只生成图表, 跳过AI评审
python -m agent.run --skip-review
```

### 4. Rust 回测

```bash
cd backtest-engine
cargo run --release -- --data ../data/signals/market_data.parquet
```

## B1 策略体系

### 核心信号条件

| 条件 | 说明 |
|------|------|
| J <= 13.8 | KDJ J值超卖 |
| 倍量柱/关键K | 28天内有资金异动 |
| WL > YL | 日线知行双线多头 |
| 形态收敛 + 量能窒息 | 视觉量化严选 |
| 均线基因指纹 | WL/YL 三重乖离率约束 |

### 多周期过滤 (wmacd 版本)

| 过滤器 | 作用 |
|--------|------|
| 周线 MACD: DIF > 0 且红柱 | 中期趋势确认 |
| 月线 MACD: 红柱 | 大趋势向上 |
| 周线 WL > YL (可选) | 周级别短期趋势健康 |

### 统计学回测表现 (wmacd + 周线WL>YL, 2020-2026)

```
信号总数: 5320
持仓5天:  53.2% 胜率, 1.49% 均值, 1.84x 盈亏比
持仓10天: 50.7% 胜率, 2.49% 均值, 2.33x 盈亏比
持仓20天: 47.5% 胜率, 4.44% 均值, 3.32x 盈亏比
持仓30天: 43.0% 胜率, 5.56% 均值, 4.29x 盈亏比
```

## Rust 回测引擎

通过 `config.toml` 配置，无需重新编译即可调参：

- 动态仓位 (复利): 盈利时自动加仓，亏损时自动减仓
- 分批止盈: 15% 卖 1/3 → 30% 卖 1/3 → 跌破 WL 清仓
- 3% 止损 + 10 天弱势清仓
- 交易成本: 佣金万 2.5 + 印花税千 1 + 滑点 0.1%

### 为什么用 Bevy ECS？

- **模块化**: 买入/卖出/统计逻辑完全解耦
- **高性能**: Rust + ECS 架构，毫秒级回测
- **可扩展**: 新增卖出条件只需添加 System

## 注意事项

- `data/` 目录已被 `.gitignore` 忽略
- 数据存储使用 DuckDB，支持 QMT 和 Baostock 两种数据源
- 建议使用 Marimo 替代 Jupyter 进行交互式研究
- AI Agent 需要 `GEMINI_API_KEY` 环境变量 (使用 Gemini 评审时)
