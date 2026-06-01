# QuantLab 当前状态

这是项目第一阅读入口，也是唯一当前状态看板。它只描述 raw execution 修正后的当前可交易 AMV 状态、关键判断、活跃风险和日常命令。

## 策略命名规则

`家族-变体`：family 表达策略逻辑，variant 表达参数/规则差异。

| 家族 | 逻辑 | 代表策略 |
|---|---|---|
| `trend` | 趋势突破延续（贴近 20 日新高） | trend-p2, trend-p3, trend-p3-enhanced |
| `pullback` | 回调反弹（均线/离散回踩） | pullback-pb3 |
| `event` | 事件驱动（涨停生态） | event-firstboard |

## 真实口径

- 可交易结论以 Rust `bt-amv-topn` 为准。
- signal artifact 提供 raw 字段时，交易执行使用 `raw_ohlc_pre_close`。
- 旧 adjusted-execution 回测只作为历史参考，不能当当前真实指标。

## 当前 Baseline

`trend-p2`（趋势突破 P2/K0.5 静态 Top3）

- Raw execution `6td static strict Top3`
- 总收益: `+145.10%`，MaxDD: `18.97%`
- 摘要来源: `strategies/amv/status.py`，入口 `uv run python scripts/qlab.py status`

## 当前核心 Challenger

`trend-p3`（趋势突破 P3/K0.5 静态 Top3）

- Raw execution `6td static strict Top3`
- 总收益: `+172.37%`，MaxDD: `13.53%`
- 相对 trend-p2: 总收益 `+27.27pp`，且回撤更低
- P3 adjusted-vs-raw 归因: `uv run python scripts/qlab.py attribution p3-raw-vs-adjusted`

## 当前最强 Static Challenger

`trend-p3-enhanced`（trend-p3 + 板块顺风 + 中期结构质量增强）

- Raw execution `6td static strict Top3`
- 总收益: `+238.54%`，MaxDD: `14.03%`
- 规则: sector-tailwind (penalty) + medium-trend-quality (penalty)
- 状态: candidate / forward monitor，暂不直接替换默认 P3
- 风险: 2026-01 仍被牺牲（-110.4K trade delta）

## Pullback Sleeve

`pullback-pb3`（回调 PB3/CP1/RV0 rolling + AMV 风控）

- Raw execution `6td rolling21 refill Top10`
- 总收益: `+80.55%`，MaxDD: `11.75%`
- 规则: amv-regime-gate (gate)
- 与 trend 家族日收益相关性 ~0.26，是自然互补 sleeve
- 状态: 进入组合权重前，需要重新做 raw-execution allocation 分析

## 涨停生态 Sleeve

`event-firstboard`（首板后回踩 + 弱窗口过滤）

- Base `5td`: `+130.95%`, MaxDD `45.38%`
- Weakgate `5td`: `+155.04%`, MaxDD `34.12%`
- 规则: event-weakgate (gate)
- 状态: 仅保留为 research candidate，尚未 allocation-ready

## 关键判断

- 当前主线已经从 Rotation / B1 / B3 收敛到 AMV Bull Pool TopN；旧路线只作为归档和低相关策略候选回看。
- raw execution 没有推翻 P3 / context 方向，但显著压低旧 adjusted 口径收益。
- P3 adjusted-vs-raw 买入的是同一批 `274/274` 笔交易，收益下降主要来自 raw 价格下的资金占用、手数取整、路径复利和少数除权窗口。
- pullback-pb3 与 trend 家族低相关，是更自然的互补 sleeve 候选。
- 上下文增强方向保留为 challenger / forward monitor。
- 涨停生态首板后回踩有 alpha 线索，但弱窗口连续回撤仍重。

## 活跃风险与未决项

- 2026 路径脆弱性仍是真风险：上下文增强改善中位路径，但 worst offset 未修复。
- 行业顺风因子仍需历史行业映射版本复核，避免静态东方财富行业映射造成历史偏差。
- Pullback sleeve 需要重新做 raw-execution allocation。
- Event sleeve 需要更好的弱窗口定义或连续 downsize / position sizing。

## 不要再当当前真实结论使用

- 旧 adjusted-execution 指标。
- Python close-to-close label 收益。
- 任何没有通过 Rust raw execution 复核的策略结论。
- 旧名称 `ref` / `p3` / `context` / `pb3-gated` / `limit-weakgate`。
- `qlab status` 的数值来自 `artifacts/<strategy>/backtests/<latest>/result.json`，不是硬编码。

## 文档归属

- `CURRENT_STATE.md`: 当前真实口径、核心指标、关键判断、活跃风险和日常命令。
- `strategies/target-strategy-evolution.md`: 小红书博主外部对标与 QuantLab 长期方向。
- `strategies/archive-index.md`: B1 / B3 / Rotation / LTR / oracle 等已归档路线的回看索引。

## 日常命令

```bash
# 查看状态（自动从 result.json 读取）
uv run python scripts/qlab.py status

# 导出信号 → artifacts/<strategy>/signal.parquet
uv run python scripts/qlab.py export trend-p3

# 回测 → artifacts/<strategy>/backtests/<ts>/
uv run python scripts/qlab.py backtest trend-p3

# 查看历史结果
uv run python scripts/qlab.py results trend-p3
uv run python scripts/qlab.py results trend-p3 --diff

# 一键导出 + 回测
uv run python scripts/qlab.py run trend-p3

# 回测参数覆盖
uv run python scripts/qlab.py backtest trend-p3 --top-n 5 --max-hold 3
```
