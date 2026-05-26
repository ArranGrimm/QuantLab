# QuantLab - Claude 指令

A 股多策略量化研究仓库。主线是 AMV Bull Pool TopN：Python/Polars 做因子与 executable-aware 评估，Rust `bt-amv-topn` 做 T+1 open 真实回测。

主线状态、归因结论、当前优先级一律以 `project-status.md` 为准；新实验流水写入 `progress.md`。

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
- **可复用代码放 `scripts/` 或 `utils/`**，不要堆在 notebook 里。一次性探索可以 notebook，但结论要落 script。
- **日志**: 用 `loguru`；不要 print 调试信息（除非临时）。

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
- **手数**: 普通 A 股 100 股/手；`sh.688*` 科创板 200 股、最小 1 股。
- **涨跌停**: close 涨停默认禁开仓 (`max_open_gap_pct = 0.098`)；任何新因子探索都必须报告 close 涨停污染率和 T+1 高开污染。
- **重复持仓**: `bt-amv-topn` 的 `allow_duplicate_positions` 默认 `false`；rolling/refill 结果对这个开关敏感，对比时注明。
- **AMV 命名**:
  - `Ref` = `reference_p2_k0p5_b0_c0_r0`
  - `P3` = `candidate_p3_k0p5_b0_c0_r0`
  - Pullback 用 `PB/CP/RV`：`PB = ma_bias_20 + disp_bias_20`，`CP = KSFT + intraday_pos`，`RV = atr_14_pct + panic_vol_ratio_20d`

## 目录与产出

| 目录 | 用途 | 是否入 git |
| --- | --- | --- |
| `scripts/` | 可复用 Python 分析/导出脚本 | 是 |
| `utils/`, `strategies/` | Python 包，工厂工具 | 是 |
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

- **`progress.md`**: 倒序流水账。每个新实验、每条调试结论都要写一行，包含日期、命令、关键数字、Canvas/JSON 路径。
- **`project-status.md`**: 当前状态看板，只反映**当前结论与优先级**。只有当稳定决策变化时才更新（比如主基线替换、归档某条路线）。
- **`reports/canvases/*.canvas.tsx`**: 长期可视化结论。新增/编辑 canvas 时遵循 `canvas` skill。
- **`README.md`**: 项目入口，改动只在结构/主线变化时更新。

## 常用分析入口

- AMV 回测对比、交易归因、sleeve 互补：先用 `.agents/skills/amv-trade-attribution/SKILL.md` 的流程。
- 通用归因脚本: `scripts/backtest_trade_attribution.py`（支持任意两个 `bt-amv-topn` artifact）。
- 信号导出: `scripts/amv_static_sleeve_signal_export.py`、`scripts/amv_bull_pool_export_signals.py`。
- Executable 评估: `scripts/amv_executable_weight_grid.py`、`scripts/amv_executable_factor_scan.py`、`scripts/amv_executable_pullback_grid.py`。

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
- 不要假设 shell 是 bash —— 当前设备可能是 Windows PowerShell。
- 不要修改默认 uv 镜像或 torch 路由规则，除非明确要解决跨设备 lock 漂移问题。
- 不要把 close-to-close 标签收益当作可交易结论；以 Rust `bt-amv-topn` 输出为准。
