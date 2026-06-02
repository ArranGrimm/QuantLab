# QuantLab 当前状态

这是项目第一阅读入口，也是唯一当前状态看板。它只描述 raw execution 修正后的当前可交易 AMV 状态、关键判断、活跃风险和日常命令。

## 策略命名规则

`家族-变体`：family 表达策略逻辑，variant 表达参数/规则差异。

| 家族 | 逻辑 | 代表策略 |
|---|---|---|
| `trend` | 趋势突破延续（贴近 20 日新高） | trend-p2, trend-p3, trend-p3-medium, trend-p3-enhanced |
| `pullback` | 回调反弹（均线/离散回踩） | pullback-pb3 |
| `event` | 事件驱动（涨停生态） | event-firstboard |

## 真实口径

- 可交易结论以 Rust `bt-amv-topn` 为准。
- 交易执行使用 `raw_ohlc_pre_close`（raw OHLC + raw pre-close）。
- 旧 adjusted-execution 回测只作为历史参考，不能当当前真实指标。

## 策略指标（raw execution，2026-06-02 重跑）

| 策略 | Return | MaxDD | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|--------|-------|------|------|------|------|------|------|
| **trend-p2** (基线) | +141.8% | 19.0% | +4.9 | +38.5 | +11.0 | +46.9 | +11.3 | -8.8 |
| **trend-p3** (挑战) | +172.4% | 13.5% | +4.0 | +39.2 | +17.6 | +43.6 | +12.5 | -0.8 |
| **trend-p3-medium** | +227.3% | 14.1% | +8.5 | +33.7 | +21.0 | +54.5 | +22.6 | -1.3 |
| **trend-p3-enhanced** | +227.3% | 14.1% | +8.5 | +33.7 | +21.0 | +54.5 | +22.6 | -1.3 |
| **pullback-pb3** (互补) | +79.3% | 11.8% | +0.3 | +5.5 | +9.7 | +17.3 | +15.4 | +14.0 |
| **event-firstboard** (研究) | +155.0% | 34.1% | +54.4 | -4.7 | +4.9 | +12.2 | +20.6 | +17.3 |

## 当前 Baseline

`trend-p3-medium`（趋势突破 P3 + 中期结构 / 趋势质量）

- Raw execution `6td static strict Top3`，274 笔交易
- 相对旧基线 trend-p2: 总收益 `+85.5pp`，回撤更低 `-4.9pp`
- 相对 trend-p3: 总收益 `+54.9pp`
- 规则: medium-trend-quality (linear penalty, p=0.03)
- trend-p2 已归档，不再研究

## 当前 Challenger

`trend-p3-enhanced`（trend-p3-medium + 行业顺风）

- 当前与 trend-p3-medium **完全等价**（sector-tailwind 在申万分类下 penalty 0.02 太小，不改变排名）
- 待行业分类参数重新调优后再评估

## 历史参考

`trend-p2`（已归档）— +141.8% / 19.0%
`trend-p3`（保留）— +172.4% / 13.5%，用于隔离 P-block 权重提升的边际效果

## Pullback Sleeve

`pullback-pb3`（回调 PB3/CP1/RV0 rolling + AMV 风控）

- Raw execution `6td rolling21 refill Top10`，1061 笔交易
- 规则: amv-regime-gate（aged + 非加速 OR 混沌期连续阴跌，开仓日 gate）
- 唯一 2026 年全为正的策略（+14.0%），与 trend 家族日收益相关性 ~0.26
- 状态: 互补 sleeve，待做 raw-execution allocation 分析

## 涨停生态 Sleeve

`event-firstboard`（首板后回踩 + 弱窗口过滤）

- Raw execution `5td static strict Top3`，256 笔交易
- 规则: event-weakgate (gate)
- 2021 年爆赚（+54.4%），但 2022 年唯一亏损策略（-4.7%）
- 年度分布与 trend/pullback 家族完全不同，是潜在独立第三 sleeve
- MaxDD 34.1% 仍是主要障碍，尚未 allocation-ready

## 关键判断

- 当前主线已从 Rotation / B1 / B3 收敛到 AMV Bull Pool TopN。
- raw execution 没有推翻趋势突破方向，但压低旧 adjusted 口径收益约 25-30pp。
- pullback-pb3 与 trend 家族低相关，且 2026 年唯一盈利，是自然互补 sleeve。
- 中期结构/趋势质量增强（medium penalty）是当前最强单因子提升，+55pp vs raw P3。
- 行业顺风（sector-tailwind）从东方财富切换到申万分类后，原有参数（p=0.02 linear）不再生效，需要重新调参。
- 涨停生态有独立 alpha 线索，但 MaxDD 34% 仍需改善。

## 活跃风险与未决项

- 2026 年：trend 家族全线微亏或微盈，仅 pullback 和 event 在涨。
- sector-tailwind 需要针对申万分类重新做参数扫描。
- pullback-pb3 需要 redo raw-execution allocation 分析。
- AKShare 已完全移除，行业分类从东方财富 → 申万（Baostock），历史回测中 sector-tailwind 相关数字不可直接对比。

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
