# QuantLab 当前状态

这是项目第一阅读入口，也是唯一当前状态看板。它只描述 raw execution 修正后的当前可交易 AMV 状态、关键判断、活跃风险和日常命令。

## 真实口径

- 可交易结论以 Rust `bt-amv-topn` 为准。
- signal artifact 提供 raw 字段时，交易执行使用 `raw_ohlc_pre_close`。
- 因子、排序、趋势特征和诊断仍可使用前复权价。
- 旧 adjusted-execution 回测只作为历史参考，不能当当前真实指标。

## 当前 Baseline

`reference_p2_k0p5_b0_c0_r0` 仍是命名上的 reference baseline。

- Raw execution `6td static strict Top3`
- 总收益: `+145.10%`
- MaxDD: `18.97%`
- 摘要来源: `strategies/amv/status.py`，入口 `uv run python scripts/qlab.py status`

## 当前核心 Challenger

`candidate_p3_k0p5_b0_c0_r0` 仍是核心替换候选。

- Raw execution `6td static strict Top3`
- 总收益: `+172.37%`
- MaxDD: `13.53%`
- 相对 reference: 总收益 `+27.27pp`，且回撤更低
- P3 adjusted-vs-raw 归因摘要: `uv run python scripts/qlab.py attribution p3-raw-vs-adjusted`
- 机制判断: P3 的优势主要来自约 `30` 笔边际替换交易，不是整体交易池重写；换票更偏向 P-block 的高位突破延续，仍需要 forward 样本确认。

## 当前最强 Static Challenger

`p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0`
是当前最强静态 challenger。

- Raw execution `6td static strict Top3`
- 总收益: `+238.54%`
- MaxDD: `14.03%`
- 状态: candidate / forward monitor，暂不直接替换默认 P3
- 风险: context combo 改善 2026 annual restart 中位路径，但没有修复最差路径；不能只看全样本 total。

## Pullback Sleeve

带 regime gate 的 `pullback_p0_k0_pb3_cp1_rv0` 是当前 pullback 代表。

- Raw execution `6td rolling21 refill Top10`
- 总收益: `+80.55%`
- MaxDD: `11.75%`
- 状态: 进入组合权重前，需要重新做 raw-execution allocation 分析

## 涨停生态 Sleeve

首板后回踩家族有 alpha 线索，但还没有 allocation-ready。

- Base `5td`: `+130.95%`, MaxDD `45.38%`
- Weakgate `5td`: `+155.04%`, MaxDD `34.12%`
- Weaktop1 / weaktier: MaxDD 仍超过 `50%`，已否决
- 状态: 仅保留为 research candidate

## 关键判断

- 当前主线已经从 Rotation / B1 / B3 收敛到 AMV Bull Pool TopN；旧路线只作为归档和低相关策略候选回看。
- 可交易结果以 `bt-amv-topn` raw execution 为准；`qlab status` 读取 `strategies/amv/status.py` 中的稳定摘要。
- raw execution 没有推翻 P3 / context 方向，但显著压低旧 adjusted 口径收益。
- P3 adjusted-vs-raw 买入的是同一批 `274/274` 笔交易，收益下降主要来自 raw 价格下的资金占用、手数取整、路径复利和少数除权窗口，不是换票或信号缺失。
- PB3 gated 与 P3 静态相关性较低，是更自然的互补 sleeve 候选；进入组合前需要重新做 raw-execution allocation 分析。
- 涨停生态首板后回踩有 alpha 线索，但弱窗口连续回撤仍重；`weakgate` 是当前防守上限，不代表 allocation-ready。
- 上下文增强方向保留为 challenger / forward monitor；短期不把 context combo 直接替换成默认 P3。

## 活跃风险与未决项

- 2026 路径脆弱性仍是真风险：上下文增强改善中位路径，但 worst offset 未修复。
- 行业顺风因子仍需历史行业映射版本复核，避免静态东方财富行业映射造成历史偏差。
- Pullback sleeve 需要重新做 raw-execution allocation，确认 `P3 static + PB3 gated rolling` 的真实组合效果。
- Limit ecology sleeve 需要更好的弱窗口定义或连续 downsize / position sizing；不要用第一版 top1/downshift 作为当前结论。

## 不要再当当前真实结论使用

- 旧 adjusted-execution 指标，例如 old Ref `+170.80%`、old P3 `+201.69%`、old context combo `+272.06%`。
- Python close-to-close label 收益。
- 任何没有通过 Rust raw execution 复核的策略结论。

## 当前工作模式

产品化清理已告一段落，项目重新切回 AMV 探索阶段。探索仍沿用清理后的轻量入口：

1. 日常入口使用 `scripts/qlab.py`。
2. 当前状态只维护本文件与 `strategies/amv/status.py`，不再维护第二份当前看板。
3. 已归档路线集中到 `strategies/archive-index.md`。
4. 更早的实验流水只在 git history 中保留，不再放在当前工作区。
5. 新探索优先围绕 `活跃市值 regime -> AMV bull pool -> TopN / sleeve -> raw execution`，不要恢复旧 Rotation / B1 / B3 日常入口。

## 跨设备交接

当前核心 raw ground truth 目标已经能从 `qlab export` 原生导出：

- `ref`
- `p3`
- `context`
- `pb3-gated`
- `limit-weakgate`

这些目标已由 `strategies/amv/` native workflow 复现，不再依赖对应旧 export 脚本作为当前入口。继续探索时优先保持两条约束：

1. 已把 `strategies/amv/workflows.py` 里膨胀的 context penalty、PB3 gate、limit weakgate 规则拆到 `strategies/amv/rules/`；后续 `workflows.py` 只保留 native workflow 编排。
2. `scripts/` 已收敛到 `qlab.py`；不要恢复旧 one-off 脚本，必要的复现能力应进入 `qlab` 或 `strategies/amv/`。

## 文档归属

- `CURRENT_STATE.md`: 当前真实口径、核心指标、关键判断、活跃风险和日常命令。
- `strategies/amv/status.py`: `qlab status` 使用的稳定数值摘要。
- `strategies/target-strategy-evolution.md`: 小红书博主外部对标与 QuantLab 长期方向。
- `strategies/archive-index.md`: B1 / B3 / Rotation / LTR / oracle 等已归档路线的回看索引。

## 核心 Canvas

- `reports/canvases/amv-p3-vs-ref-trade-attribution.canvas.tsx`
- `reports/canvases/amv-p3-sector-tailwind-complete.canvas.tsx`
- `reports/canvases/amv-medium-trend-quality-diagnostic.canvas.tsx`
- `reports/canvases/amv-limit-ecology-diagnostic.canvas.tsx`
- `reports/canvases/amv-p3-pb3-gated-allocation.canvas.tsx`

## 日常命令

```bash
uv run python scripts/qlab.py status
uv run python scripts/qlab.py export p3
uv run python scripts/qlab.py export context
uv run python scripts/qlab.py export pb3-gated
uv run python scripts/qlab.py export limit-weakgate
uv run python scripts/qlab.py backtest artifacts/amv_static_sleeve_signals/<signal_id> --preset 6td-static
uv run python scripts/qlab.py compare p3 context
uv run python scripts/qlab.py attribution p3-raw-vs-adjusted
```
