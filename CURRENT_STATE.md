# QuantLab 当前状态

这是项目第一阅读入口。它只描述 raw execution 修正后的当前可交易 AMV 状态和日常命令。

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
- 报告: `reports/amv_raw_execution_ground_truth_summary.json`

## 当前核心 Challenger

`candidate_p3_k0p5_b0_c0_r0` 仍是核心替换候选。

- Raw execution `6td static strict Top3`
- 总收益: `+172.37%`
- MaxDD: `13.53%`
- 相对 reference: 总收益 `+27.27pp`，且回撤更低
- P3 adjusted-vs-raw 归因: `reports/amv_p3_raw_vs_adjusted_trade_attribution.json`

## 当前最强 Static Challenger

`p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0`
是当前最强静态 challenger。

- Raw execution `6td static strict Top3`
- 总收益: `+238.54%`
- MaxDD: `14.03%`
- 状态: candidate / forward monitor，暂不直接替换默认 P3

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

## 不要再当当前真实结论使用

- 旧 adjusted-execution 指标，例如 old Ref `+170.80%`、old P3 `+201.69%`、old context combo `+272.06%`。
- Python close-to-close label 收益。
- 任何没有通过 Rust raw execution 复核的策略结论。

## 当前工作模式

暂停新增策略探索。下一阶段先产品化整理研究工作流：

1. 日常入口使用 `scripts/qlab.py`。
2. `project-status.md` 保留为详细状态看板。
3. `progress.md` 保留为倒序实验流水。
4. 删除或移动脚本前，先看 `docs/script-inventory.md` 和 `docs/cleanup-plan.md`。

## 跨设备交接

当前核心 raw ground truth 目标已经能从 `qlab export` 原生导出：

- `ref`
- `p3`
- `context`
- `pb3-gated`
- `limit-weakgate`

这些目标已由 `strategies/amv/` native workflow 复现，不再依赖对应旧 export 脚本作为当前入口。回家后继续整理时，优先做两件事：

1. 把 `strategies/amv/workflows.py` 里已经膨胀的 context penalty、PB3 gate、limit weakgate 规则继续拆到更清晰的策略规则模块。
2. 按 `docs/script-inventory.md` 和 `docs/cleanup-plan.md` 继续处理历史脚本；不要直接删除仍承担诊断复现职责的脚本。

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
