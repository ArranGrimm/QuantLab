# QuantLab 量化研究实验室

A 股多策略量化研究平台，集成 Polars 因子工程、LightGBM Walk-Forward 模型、Rust ECS 回测引擎。当前研究两条策略线：**截面轮动**（日频 ML 选股）和 **B1 超跌反转**（事件驱动）。

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Python 信号层                                    │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────────────┐  │
│  │ 数据加载      │ → │ 因子计算       │ → │ LightGBM Walk-Forward   │  │
│  │ DuckDB       │   │ 42因子 (T日)   │   │ 打分 → Parquet 导出     │  │
│  │ + 涨跌停标记  │   │ + 涨跌停过滤   │   │ (训练排除涨停样本)      │  │
│  └──────────────┘   └────────────────┘   └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓ Parquet
┌─────────────────────────────────────────────────────────────────────────┐
│                      Rust 回测引擎 (Bevy ECS)                            │
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────┐  │
│  │ bt-core (共享)    │  │ bt-rotation       │  │ bt-b1               │  │
│  │ Portfolio/Stats   │  │ 截面轮动策略      │  │ B1 超跌反转策略     │  │
│  │ 涨跌停判定       │  │ 涨停过滤/跌停锁仓 │  │ 止损/止盈/弱势出场  │  │
│  └──────────────────┘  └───────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
QuantLab/
├── utils/                     # Python 核心工具模块
│   ├── rotation_factors.py    #   截面轮动因子 (42个, T日数据版)
│   ├── b1_factors_opt.py      #   B1 选股因子 (V3.0 + 周月MACD + 周线WL>YL)
│   ├── renko_factors.py       #   砖型图反转因子
│   ├── ic_analysis.py         #   Polars 原生 IC 分析 (Spearman 秩相关)
│   ├── duckdb_utils.py        #   DuckDB 数据加载 + 涨跌停标记
│   ├── signal_export.py       #   信号导出 (Parquet, 供 Rust 使用)
│   ├── backtest.py            #   统计学回测 (run_backtest / run_backtest_short)
│   ├── get_data.py            #   QMT 数据同步 → DuckDB
│   ├── baostock_sync.py       #   Baostock 数据同步 → DuckDB
│   └── ...
│
├── notebooks/                 # Marimo 研究 Notebook
│   ├── cross_section_rotation.py  # 截面轮动策略 (因子→IC→LightGBM→导出)
│   ├── b1_ml_explore.py       #   B1 全市场 ML 排序探索
│   ├── b1_ml_dedicated.py     #   B1 专属模型 (结论: 不如全市场)
│   ├── renko_ml_explore.py    #   砖型图 ML 探索
│   ├── simple_b1_opt.py       #   B1 统计学回测
│   └── ...
│
├── backtest-engine/           # Rust 回测引擎 (Bevy ECS, Cargo Workspace)
│   └── crates/
│       ├── core/              #   bt-core: 共享类型 (Portfolio/Stats/涨跌停)
│       ├── rotation/          #   bt-rotation: 截面轮动 (涨停过滤/跌停锁仓)
│       └── b1/                #   bt-b1: B1 超跌反转 (止损/止盈/弱势出场)
│
├── agent/                     # AI Agent 每日选股工作流
│   ├── run.py                 #   主入口: python -m agent.run
│   └── reviewers/             #   AI 评审模块 (Gemini 多模态)
│
├── strategies/                # 通达信 TDX 策略脚本
│   └── tdx_scripts/
│
├── results/                   # 回测报告 (自动保存, 带时间戳)
├── experiments/               # 实验记录 (markdown)
└── data/                      # 数据目录 (git ignored)
    ├── signals/               #   导出的信号 Parquet
    └── sector_map_em.csv      #   东方财富行业分类
```

## 快速开始

### 1. Python 环境

```bash
uv sync
```

### 2. 截面轮动策略 (主要研究方向)

```bash
# 1. Marimo notebook: 因子计算 → IC 分析 → LightGBM 训练 → Parquet 导出
marimo edit notebooks/cross_section_rotation.py

# 2. Rust 回测
cd backtest-engine
cargo run -p bt-rotation --release
```

### 3. B1 超跌反转策略

```bash
# Marimo notebook: B1 信号 + ML 排序
marimo edit notebooks/b1_ml_explore.py

# Rust 回测
cd backtest-engine
cargo run -p bt-b1 --release
```

### 4. AI Agent 每日选股

```bash
python -m agent.run              # 完整流程
python -m agent.run --date 2026-02-24  # 指定日期
```

## Rust 回测引擎

Cargo Workspace 架构，三个 crate：

| Crate | 功能 |
|-------|------|
| `bt-core` | 共享: Portfolio, BacktestStats, CostModel, 涨跌停判定 |
| `bt-rotation` | 截面轮动: Top-N 选股, 涨停过滤, 跌停锁仓, 排名/止损退出 |
| `bt-b1` | B1 超跌反转: 信号买入, 分批止盈, 止损, 弱势出场 |

通过 `config.toml` 配置，无需重新编译即可调参。

## 注意事项

- `data/` 目录已被 `.gitignore` 忽略
- 数据存储使用 DuckDB，支持 QMT 和 Baostock 两种数据源
- 建议使用 Marimo 替代 Jupyter 进行交互式研究
- AI Agent 需要 `GEMINI_API_KEY` 环境变量
