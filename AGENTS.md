# QuantLab - Agent 指令

A 股多策略量化研究仓库。主线是 AMV Bull Pool TopN：Python/Polars 做因子与 executable-aware 评估，Rust `bt-amv-topn` 做 T+1 open 真实回测。

## 第一阅读顺序

1. `CURRENT_STATE.md`: 当前真实口径、baseline、challenger、sleeve 状态和日常命令。
2. `AGENTS.md`: agent 工作规则、代码分层、整理期约束。
3. `project-status.md`: 当前详细结论、关键指标、活跃风险和优先级；整理期不作为频繁更新入口。
4. `progress.md`: 倒序实验流水账；整理期只记录阶段性里程碑。
5. `docs/script-inventory.md`: 脚本状态标签，删除或迁移前先看。
6. `docs/cleanup-plan.md`: 第二阶段删除 / 迁移门禁。

主线策略结论仍以 `project-status.md` 为准；但产品化整理期间，不要每次小改都更新 `project-status.md` 或 `progress.md`。

## 当前工作模式

- 当前阶段优先做项目产品化整理，不继续新增策略探索。
- 日常入口收敛到 `scripts/qlab.py`，不要再随手新增一堆一次性入口脚本。
- 第一阶段只建立新秩序，不删除脚本、不大规模移动文件、不改 Rust 引擎结构。
- 第二阶段再按 `docs/script-inventory.md` 和 `docs/cleanup-plan.md` 删除或迁移。

常用命令：

```bash
uv run python scripts/qlab.py status
uv run python scripts/qlab.py export p3
uv run python scripts/qlab.py backtest <signal_dir> --preset 6td-static
uv run python scripts/qlab.py compare p3 context
uv run python scripts/qlab.py attribution p3-raw-vs-adjusted
```

## 设备分支

仓库会在 Windows 和 Mac 上交替使用，差异必须先确认再动手。

- **判定**: 用 `uname` / `$env:OS` / `sys.platform` 判断；不要假设当前 shell。
- **Shell**:
  - Windows 是 PowerShell（不是 bash）。`ls -la`、heredoc、`&&` 链式语义都会失败；用 `Get-ChildItem`、`;` 分隔、或拆成多次调用。终端读取大文件中文易乱码，优先用 Read 工具直接看文件。
  - Mac 是 zsh/bash，标准 POSIX 即可。
- **路径**:
  - Windows: `D:\WorkSpace\Tinkering\QuantLab`，反斜杠或正斜杠都行，但传给 PowerShell 时优先正斜杠或加引号。
  - Mac: 一般是 `~/Workspace/QuantLab` 一类的 POSIX 路径。引用项目内文件用相对路径，避免硬编码绝对路径。
- **Python 后端**:
  - Torch/Torchvision: Windows 走 `pytorch-cu130` 显式源（已在 `pyproject.toml` 配置），Mac 走默认源（CPU/MPS）。任何动 torch 依赖的改动都要保留 `marker = "sys_platform == 'win32'"` 路由规则。
  - `pyobjc-framework-*` 只在 `sys_platform == 'darwin'` 安装；Windows 上不要 import 这类包。
- **性能预算**: full grid (`scripts/amv_executable_pullback_grid.py` 等) 在 Windows 上可能跑不动，能跑 focused grid 就别上 full grid；过夜任务在 Mac 上跑。
- **uv 镜像**: 默认源锁死阿里云 `aliyun`（见 `pyproject.toml`），不要为了拉包临时改源；要新增源用 `[[tool.uv.index]]` 并加 explicit 路由，避免污染 `uv.lock`。

## Python 工作流

- **环境**: Python 3.13，统一用 uv 管理。
- **执行命令一律 `uv run`**，不要直接 `python`、不要 `pip install`：
  - `uv run python scripts/xxx.py`
  - `uv add <pkg>` 添加依赖（不要手改 `pyproject.toml` 的 `dependencies` 再 sync）
  - `uv sync` 拉齐环境
- **数据处理用 Polars，不用 Pandas**。仓库里 `pandas` 仍在依赖中只是历史兼容，新代码全部用 Polars（lazy frame 优先），需要互转时显式 `.to_pandas()` 并注明原因。
- **Notebook**: 用 Marimo（`uv run marimo edit notebooks/xxx.py`），不要新增 Jupyter `.ipynb`；`__marimo__/` 是临时目录，已 gitignore。
- **日志**: 用 `loguru`；不要 print 调试信息（除非临时）。

## Python 代码分层

### `strategies/`

放可复用策略定义和稳定元数据：

- canonical sleeve id 与别名
- score expression registry
- preset 名称与稳定参数
- CLI 或重复 runner 会用到的策略族元信息

不要把 DuckDB 访问、报告写入、一次性诊断放在这里。

当前轻量 registry：

- `strategies/amv/registry.py`: 维护 `qlab.py` 的 export target、backtest preset、report alias。

### `utils/`

放跨策略通用的底层工具：

- DuckDB / QMT 数据读取
- ST 黑名单、上市状态、基础证券元数据
- raw / adjusted price 加载和处理
- 行业映射底层能力
- 通用 Polars helper
- 文件系统 / 报告工具函数

不要把策略特定排序公式放进 `utils/`。如果 helper 名字直接包含 P3、PB3、medium128、sector tailwind、limit ecology，第二阶段应迁到 `strategies/amv/`。

### `scripts/`

只放命令入口：

- `scripts/qlab.py`
- 少数 canonical export runner
- 少数 canonical backtest / compare / attribution runner
- 仍需复现稳定报告的 focused diagnostic runner

新脚本必须尽量薄。重的因子和 score 逻辑在第二阶段逐步迁入 `strategies/amv/`。

### `reports/`

只放稳定产物：

- JSON 结论
- Canvas 可视化
- 长期保留的诊断输出

不要放大体积 parquet、CSV 回测明细或临时数据。

### `experiments/`

放历史路线说明和归档研究笔记，不作为日常入口。

## Rust 回测引擎

位于 `backtest-engine/`，工作区根 `Cargo.toml` 管理所有 crates：

- `core`: 账户、费用、A 股涨跌停、交易统计
- `amv-topn`: AMV TopN static / rolling 主回测，**当前主线**
- `amv-cohort-diagnostic`: close-to-close cohort 诊断
- `b1`, `b3`, `renko`, `rotation`: 旧策略线，归档维护

执行：

```bash
cd backtest-engine
cargo run -p bt-amv-topn --release -- \
  --data ../artifacts/amv_static_sleeve_signals/<signal_id>/signal.parquet \
  --config crates/amv-topn/config_6td_static_strict_top3_no_stop.toml \
  --output-dir ../artifacts/.../backtests/<run_id>
```

- 优先 `--release`，debug 编译跑大数据慢一个数量级。
- `Cargo.lock` 已 gitignore；不要提交。
- `bt-amv-topn` Rust 输出是 **可交易结果的 source of truth**；Python label 收益只是参考。

## A 股可执行口径（硬约束）

- **执行模型**: D+1 open 开仓 → horizon close 平仓；不要用 D close 直接当 entry。
- **成交价格口径**: 因子、排序、趋势收益可以用前复权；真实成交、资金占用、手数、费用、涨跌停 / 高开过滤必须用 raw OHLC / raw pre-close。
- **手数**: 普通 A 股 100 股/手；`sh.688*` 科创板 200 股、最小 1 股。
- **涨跌停**: close 涨停默认禁开仓 (`max_open_gap_pct = 0.098`)；任何新因子探索都必须报告 close 涨停污染率和 T+1 高开污染。
- **重复持仓**: `bt-amv-topn` 的 `allow_duplicate_positions` 默认 `false`；rolling/refill 结果对这个开关敏感，对比时注明。
- **AMV 命名**:
  - `Ref` = `reference_p2_k0p5_b0_c0_r0`
  - `P3` = `candidate_p3_k0p5_b0_c0_r0`
  - Pullback 用 `PB/CP/RV`：`PB = ma_bias_20 + disp_bias_20`，`CP = KSFT + intraday_pos`，`RV = atr_14_pct + panic_vol_ratio_20d`

## Preset 名称

稳定 preset registry 位于 `strategies/amv/registry.py`：

- `6td-static` -> `config_6td_static_strict_top3_no_stop.toml`
- `5td-static` -> `config_5td_static_strict_top3_no_stop.toml`
- `3td-static` -> `config_3td_static_strict_top3_no_stop.toml`
- `6td-rolling` -> `config_6td_rolling21_refill_top10_no_stop.toml`

日常操作不要再要求用户记 Rust config 文件名。

## 脚本状态标签

每个脚本只标一个 operational status，记录在 `docs/script-inventory.md`：

- `canonical`: 明确推荐的直接入口。
- `implementation`: CLI 或 canonical script 内部调用，用户一般不直接运行。
- `diagnostic`: 可复现实验诊断，不是日常入口。
- `historical`: 历史结论已沉淀，保留用于追溯。
- `deprecated`: 第二阶段删除或迁移候选。

这些标签只是操作状态，不是价值判断。`historical` 脚本仍可能有价值，只是不应该作为新用户第一入口。

## 目录与产出

| 目录 | 用途 | 是否入 git |
| --- | --- | --- |
| `scripts/` | 命令入口、少数 runner | 是 |
| `strategies/` | 策略定义、score registry、canonical 参数 | 是 |
| `utils/` | 数据读取、价格处理、ST、行业映射等通用工具 | 是 |
| `backtest-engine/` | Rust 工作区 | 是（不含 `target/`） |
| `notebooks/` | Marimo 入口 | 是 |
| `reports/` | 稳定 JSON 结论、`canvases/*.canvas.tsx` 可视化 | 是 |
| `experiments/` | 归档实验文档 | 是 |
| `.agents/skills/` | 跨设备追踪的 Agent Skills | 是 |
| `artifacts/`, `results/`, `data/`, `logs/` | 大文件、回测产物、中间数据 | **否** |
| `.cursor/` | Cursor 本地配置 | **否** |

- 大文件 / parquet / csv / pkl / html 全部已被 `.gitignore`，不要尝试 `git add`。
- 想跨设备同步的 skill / 配置必须放 `.agents/skills/`，不要放 `.cursor/`。

## 文档与状态更新规则

- **`CURRENT_STATE.md`**: 第一阅读入口，只保留当前真实口径、baseline、challenger、sleeve 状态、禁用旧结论和日常命令。
- **`progress.md`**: 实验流水账，不是整理期操作日志。新增策略实验、关键调试结论、阶段性整理完成时再更新；普通文档改名、引用清理、脚本分类微调不要更新。
- **`project-status.md`**: 当前详细状态看板。只有稳定策略决策或项目级里程碑变化时才更新；整理期的小步重构、删除候选调整、文档措辞变化不要更新。
- **`reports/canvases/*.canvas.tsx`**: 长期可视化结论。新增/编辑 canvas 时遵循 `canvas` skill。
- **`README.md`**: 项目入口，改动只在结构/主线变化时更新。
- **文档语言**: 面向项目使用者的文档默认中文；代码标识、命令、技术名词可保留英文。

## 常用分析入口

- AMV 回测对比、交易归因、sleeve 互补：先用 `.agents/skills/amv-trade-attribution/SKILL.md` 的流程。
- 日常入口: `scripts/qlab.py`
- 通用归因脚本: `scripts/backtest_trade_attribution.py`（支持任意两个 `bt-amv-topn` artifact）。
- 当前 signal export 由 `qlab export` 包装，不再优先手动记忆具体脚本名。
- Executable 评估旧入口已进入历史/诊断清单；使用前先看 `docs/script-inventory.md`。

## 编码与风格

- 遵循 SOLID（特别是开闭原则、接口隔离）。
- 错误处理显式，不要吞异常；脚本入口处用 `loguru` 记录。
- **不要删除已有注释**；注释只解释非显然的意图/权衡，不复述代码做什么。
- 遵守现有 prettier / formatter；新代码风格与同目录现有文件保持一致。
- 性能敏感路径（因子扫描、grid）优先 Polars lazy + `collect(streaming=True)`，避免 `for row in df`。

## 不要做的事

- 不要把回测输出、parquet、log 写进 git。
- 不要绕过 `uv run` 直接调系统 Python。
- 不要在新代码里用 Pandas 处理主流程数据。
- 不要假设 shell 是 bash，当前设备可能是 Windows PowerShell。
- 不要修改默认 uv 镜像或 torch 路由规则，除非明确要解决跨设备 lock 漂移问题。
- 不要把 close-to-close 标签收益当作可交易结论；以 Rust `bt-amv-topn` 输出为准。
- 不要把所有历史脚本都包装进 `qlab.py`。CLI 只包当前 canonical workflow。
