# QuantLab 量化研究实验室

QuantLab 是面向 A 股的多策略量化研究仓库，核心目标是把“标签侧看起来有效”的因子探索，推进到可执行口径下的真实回测和组合归因。

当前主线已经收口到 **AMV Bull Pool TopN**：使用 Python/Polars 做因子与 executable-aware 评估，用 Rust `bt-amv-topn` 做 `T+1 open` 真实交易回测，并通过 Canvas/JSON 文档化年度表现、交易归因和 sleeve 互补性。

## 当前主线

- 主策略底座: `manual_p2_k0p5_r0_6td`
  - 修正 A 股手数后，Rust `6td static strict Top3` 净收益 `+170.80%`, MaxDD `15.30%`
- 主基线替换候选: `P3/K0.5/R0`
  - Rust `6td static strict Top3` 净收益 `+201.69%`, MaxDD `13.52%`
  - 优势主要来自少量边际换票，机制偏向更纯的高位突破延续
- 互补 sleeve 代表: `PB3/CP1/RV0 rolling21 refill`
  - rolling21 refill Top10 净收益 `+99.62%`, MaxDD `20.70%`
  - 与 P/K 主线低相关，用于后续 allocation/gating
- 已降级路线:
  - P/K/M 动量增强: 标签侧收益强，但真实 Rust 回测未兑现为更好主基线
  - Direct LTR Top3: executable label 与真实执行口径错配，暂不接交易
  - B3/TDX、B1、rotation: 作为归档或辅助策略线，不再是当前主线

## 研究框架

```
Python 研究与导出
  ├─ 因子计算 / 权重网格 / 全因子扫描
  ├─ executable-aware 评估: D+1 open -> horizon close
  ├─ 污染归因: close 涨停 / T+1 高开 / 补位表现
  └─ 信号导出: selected_signals.csv / signal.parquet

Rust 真实回测
  ├─ bt-core: 账户、费用、涨跌停、交易统计
  ├─ bt-amv-topn: AMV static / rolling TopN 回测
  └─ bt-amv-cohort-diagnostic: close-to-close cohort 诊断

归因与可视化
  ├─ reports/*.json: 结构化实验结果
  ├─ reports/canvases/*.canvas.tsx: 可视化分析
  ├─ progress.md: 实验流水账
  └─ project-status.md: 当前状态看板
```

## 目录结构

```
QuantLab/
├── scripts/
│   ├── amv_executable_weight_grid.py      # executable-aware 权重网格
│   ├── amv_executable_factor_scan.py      # 早期因子 executable 重扫
│   ├── amv_executable_pullback_grid.py    # pullback combo grid
│   ├── amv_static_sleeve_signal_export.py # AMV sleeve 信号导出
│   └── backtest_trade_attribution.py      # 通用 bt-amv-topn 交易归因
│
├── backtest-engine/
│   └── crates/
│       ├── core/                  # 共享账户、费用、涨跌停逻辑
│       ├── amv-topn/              # AMV TopN static / rolling 回测
│       ├── amv-cohort-diagnostic/ # close-to-close cohort 诊断
│       ├── rotation/              # 旧截面轮动策略
│       ├── b1/                    # B1 超跌反转
│       ├── b3/                    # B3/TDX 事件策略
│       └── renko/                 # Renko 相关探索
│
├── reports/
│   └── canvases/                  # 可追踪 Canvas 分析
├── experiments/                   # 归档实验文档
├── utils/                         # Python 数据与因子工具
├── notebooks/                     # Marimo 研究入口
├── .agents/skills/                # 项目级 Agent Skills
├── progress.md                    # 实验流水账
└── project-status.md              # 当前状态看板
```

## 快速开始

### Python 环境

```bash
uv sync
```

后续 Python 命令统一使用 `uv run`：

```bash
uv run python scripts/amv_executable_weight_grid.py
uv run python scripts/backtest_trade_attribution.py --help
```

### Rust 回测

```bash
cd backtest-engine
cargo run -p bt-amv-topn --release -- \
  --data ../artifacts/amv_static_sleeve_signals/<signal_id>/signal.parquet \
  --config crates/amv-topn/config_6td_static_strict_top3_no_stop.toml \
  --output-dir ../artifacts/amv_static_sleeve_signals/<signal_id>/backtests/<run_id>
```

常用 AMV 配置位于：

```
backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml
backtest-engine/crates/amv-topn/config_6td_static_refill_top10_no_stop.toml
backtest-engine/crates/amv-topn/config_6td_rolling21_strict_top3_no_stop.toml
backtest-engine/crates/amv-topn/config_6td_rolling21_refill_top10_no_stop.toml
```

### 交易归因

```bash
uv run python scripts/backtest_trade_attribution.py \
  --left-backtest artifacts/.../backtests/<left_run> \
  --right-backtest artifacts/.../backtests/<right_run> \
  --left-label "Ref" \
  --right-label "Candidate" \
  --out reports/<name>.json
```

## 文档入口

- `project-status.md`: 当前结论、优先级、主线状态
- `progress.md`: 按日期倒序的实验流水账
- `reports/canvases/`: 可视化分析与关键 Canvas
- `.agents/skills/amv-trade-attribution/SKILL.md`: AMV 回测对比与交易归因工作流

## 注意事项

- `data/`、`artifacts/`、`results/` 等大文件目录默认被 `.gitignore` 忽略
- `reports/` 下的关键 JSON 和 Canvas 用于追踪稳定结论
- `.cursor/` 保持 ignore；需要跨设备追踪的 Agent Skill 放在 `.agents/skills/`
- 可执行评估优先级高于 close-to-close 标签收益，所有新因子探索都需要关注涨停/高开污染
