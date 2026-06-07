# QuantLab - Agent 指令

A 股多策略量化研究仓库。主线 AMV Bull Pool TopN：Python/Polars 因子 + Rust `bt-amv-topn` T+1 open 回测。

## 第一阅读顺序

1. `CURRENT_STATE.md`: 当前真实口径、baseline、challenger、日常命令。
2. `AGENTS.md`: 本文件。
3. `strategies/archive-index.md`: 已归档路线索引。
4. `.claude/skills/`: 专项 skill（qlab 工作流、AMV 实现、清理维护、交易归因、Polars）。

主线结论以 `CURRENT_STATE.md` 为准。

## 设备分支

仓库在 Windows 和 Mac 交替使用。

- **判定**: 用 `uname` / `$env:OS` / `sys.platform`；不要假设 shell。
- **Shell**: Windows = PowerShell（非 bash）；Mac = zsh。
- **路径**: Windows `D:\WorkSpace\Tinkering\QuantLab`，Mac POSIX。引用项目文件用相对路径。
- **Python 依赖**: Torch 走 `sys_platform` marker 路由；`pyobjc-framework-*` 仅 darwin。
- **uv 镜像**: 锁死阿里云，不临时改源。

## Python 工作流

- Python 3.13 + `uv run`，不直接 `python` / `pip install`。
- **Polars only**（lazy frame 优先），不用 Pandas。
- Notebook: Marimo（`uv run marimo edit`），不新增 `.ipynb`。
- 日志: `loguru`，不 print。

## 代码分层

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `strategies/amv/` | 策略定义、因子、pipeline、configs、hook | DuckDB 访问、报告写入 |
| `utils/` | 通用数据读取、ST、行业映射、Polars helper | 策略特定排序公式 |
| `scripts/` | 只 `qlab.py` | 一次性研究脚本 |
| `reports/` | Canvas 可视化、少量诊断 | parquet / CSV / 临时数据 |

详细分层规则见 `quantlab-amv-implementation` skill。

## Rust 回测引擎

`backtest-engine/` → `cargo run -p bt-amv-topn --release`。Rust 输出是**可交易结果的 source of truth**，Python label 收益只是参考。旧 crate（b1/b3/renko/rotation）仅保留。

## 可执行口径（硬约束）

- **D+1 open** 开仓，不用 D close 当 entry。
- 成交用 **raw OHLC / raw pre-close**；因子/排序可用前复权。
- 科创板 200 股/手，普通 A 股 100 股/手。
- **close 涨停默认禁开仓**；新因子必须报告涨停污染率 + T+1 高开污染。
- **策略命名**: `家族-变体`（trend-p3, pullback-pb3, event-firstboard）。旧名 `ref`/`p3`/`context` 不再使用。

## 常用命令

```bash
uv run python scripts/qlab.py status
uv run python scripts/qlab.py run trend-p3
uv run python scripts/qlab.py results trend-p3 --diff
```

详细操作见 `quantlab-qlab-workflow` skill。

## 编码与风格

- 错误处理显式，不吞异常。
- **不要删除已有注释**。
- 性能敏感路径优先 Polars lazy + `collect(streaming=True)`。

## 不要做的事

- 回测输出 / parquet / log 写进 git。
- 绕过 `uv run` 直接调系统 Python。
- 新代码用 Pandas。
- 假设 shell 是 bash。
- close-to-close 标签收益当可交易结论。
- 新增一次性入口脚本。