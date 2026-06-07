# QuantLab 量化研究实验室

QuantLab 是面向 A 股的多策略量化研究仓库，核心目标是把“标签侧看起来有效”的因子探索，推进到可执行口径下的真实回测和组合归因。

当前主线已经收口到 **AMV Bull Pool TopN**：使用 Python/Polars 做因子与 executable-aware 评估，用 Rust `bt-amv-topn` 做 `T+1 open` 真实交易回测，并通过 `scripts/qlab.py` 统一日常导出、回测、对比和归因入口。

## 当前主线

- 当前真实口径: Rust `bt-amv-topn` + raw OHLC / raw pre-close 执行；旧 adjusted-execution 指标只作历史参考。
- Reference baseline: `reference_p2_k0p5_b0_c0_r0`，Raw `6td static strict Top3` 总收益 `+145.10%`, MaxDD `18.97%`。
- 核心替换候选: `candidate_p3_k0p5_b0_c0_r0`，Raw `6td static strict Top3` 总收益 `+172.37%`, MaxDD `13.53%`。
- 当前最强静态 challenger: `p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0`，Raw `6td static strict Top3` 总收益 `+238.54%`, MaxDD `14.03%`。
- Pullback sleeve 代表: `pullback_p0_k0_pb3_cp1_rv0` + regime gate，Raw `6td rolling21 refill Top10` 总收益 `+80.55%`, MaxDD `11.75%`。
- 涨停生态 sleeve 仍是 research candidate，`limit-weakgate` 防守改善但还不是 allocation-ready。

## 研究框架

```
Python 研究与导出
  ├─ scripts/qlab.py: status / export / backtest / compare / attribution
  ├─ strategies/amv/: canonical sleeve id、score expression、规则模块
  ├─ executable-aware 评估: D+1 open -> horizon close
  └─ 信号 artifact: signal.parquet / signal.meta.json

Rust 真实回测
  ├─ bt-core: 账户、费用、涨跌停、交易统计
  ├─ bt-amv-topn: AMV static / rolling TopN 回测
  └─ bt-amv-cohort-diagnostic: close-to-close cohort 诊断

归因与可视化
  ├─ reports/canvases/*.canvas.tsx: 少量核心可视化分析
  ├─ strategies/amv/status.py: 当前 raw ground truth 摘要
  ├─ CURRENT_STATE.md: 第一阅读入口 / 当前状态看板
  └─ strategies/archive-index.md: 已归档路线索引
```

## 目录结构

```
QuantLab/
├── scripts/
│   └── qlab.py                    # 日常 CLI 入口
│
├── strategies/
│   ├── target-strategy-evolution.md  # 外部对标：博主策略演化
│   ├── archive-index.md              # 已归档路线索引
│   └── amv/
│       ├── attribution.py         # qlab attribution 归因实现
│       ├── registry.py            # export target / backtest preset / report alias
│       ├── workflows.py           # qlab native workflow 编排
│       └── rules/                 # context、PB3 gate、limit weakgate 规则
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
├── utils/                         # Python 数据与因子工具
├── notebooks/                     # Marimo 研究入口
├── CURRENT_STATE.md               # 第一阅读入口 / 当前状态看板
└── AGENTS.md                      # agent 工作规则
```

## 快速开始

### Python 环境

```bash
uv sync
```

后续 Python 命令统一使用 `uv run`：

```bash
uv run python scripts/qlab.py status
uv run python scripts/qlab.py export p3
uv run python scripts/qlab.py export context
uv run python scripts/qlab.py backtest artifacts/amv_static_sleeve_signals/<signal_id> --preset 6td-static
uv run python scripts/qlab.py compare p3 context
```

### Rust 回测

日常优先使用 `qlab backtest`，不再要求手写 Rust config 路径：

```bash
uv run python scripts/qlab.py backtest artifacts/amv_static_sleeve_signals/<signal_id> --preset 6td-static
```

底层 Rust 命令仍可用于排障：

```bash
cd backtest-engine
cargo run -p bt-amv-topn --release -- \
  --data ../artifacts/amv_static_sleeve_signals/<signal_id>/signal.parquet \
  --config crates/amv-topn/config_6td_static_strict_top3_no_stop.toml \
  --output-dir ../artifacts/amv_static_sleeve_signals/<signal_id>/backtests/<run_id>
```

常用 preset 位于 `strategies/amv/registry.py`：

```
6td-static
5td-static
3td-static
6td-rolling
```

### 交易归因

```bash
uv run python scripts/qlab.py attribution trade \
  --left-backtest artifacts/.../backtests/<left_run> \
  --right-backtest artifacts/.../backtests/<right_run> \
  --left-label "Ref" \
  --right-label "Candidate" \
  --out artifacts/attribution/<name>.json
```

## 文档入口

- `CURRENT_STATE.md`: 第一阅读入口和当前状态看板，记录真实口径、baseline、关键判断、活跃风险和日常命令
- `AGENTS.md`: agent 工作规则、代码分层、跨设备注意事项
- `strategies/target-strategy-evolution.md`: 外部对标（博主策略演化）
- `strategies/archive-index.md`: 已归档探索路线索引
- `reports/canvases/`: 可视化分析与关键 Canvas

## 注意事项

- `data/`、`artifacts/`、`results/` 等大文件目录默认被 `.gitignore` 忽略
- `reports/` 只保留少量核心 Canvas；结构化状态摘要迁入 `strategies/amv/status.py`
- `.cursor/` 保持 ignore
- 产品化清理已告一段落，当前重新回到 AMV 探索阶段；新探索仍应复用 `qlab.py` 和 `strategies/amv/`
- 可交易结论以 Rust raw execution 为准；Python close-to-close 标签收益只作参考
