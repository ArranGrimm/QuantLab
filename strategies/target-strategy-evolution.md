# 对标博主策略全景

> **外部对标**: 汇总小红书博主公开信息，区分其早期截面多因子阶段与后期多策略 rule-based 阶段。
> **QuantLab 自有路线**: 不复刻博主完整体系，而是以 **活跃市值 (0AMV)** 为 regime 主轴，在 AMV bull pool 上构建可执行的 TopN / sleeve 体系。
> 当前可交易结论与日常命令见 `CURRENT_STATE.md`；归档旧路线见 `strategies/archive-index.md`。

---

## 一、结论先行

### 博主侧（外部对标不变）

- 博主**并不是一直在做截面多因子排序**；后期更接近 **12 个子策略 + 市场状态切换 + 日频底仓 + 分钟级 T+0**。
- 她公开表达的核心转折: 纯截面排序有缺陷，纯量价 rule-based 难跨环境，因此需要 **多策略组合**。
- 这些仍是我们理解的**长期系统目标**，但不等于要逐条复刻她的 12 策略。

### QuantLab 侧（我们已经找到的主线）

- 项目真正沉淀下来的主轴，是 **围绕活跃市值做文章**:
  - **Regime / timing**: 活跃市值多头区间决定是否值得进攻；alpha 本质是 timing，不是单纯 cross-section 排序。
  - **选股池**: AMV bull pool — 在活跃市值偏多的市场环境中，从 bull 候选里做 TopN 排序与 sleeve 分工。
  - **可执行验证**: Rust `bt-amv-topn` + raw execution；Python 只做因子、导出与诊断。
- 当前主线不是 `Rotation`、不是 B1、也不是复刻博主早期 128 日截面模型，而是:
  - **核心静态**: P3 / context combo（`6td static strict Top3`）
  - **互补 sleeve**: PB3 gated rolling、limit ecology（research / allocation 评估中）
  - **数据基础设施**: `rpa_capture/` + `rpa_parse/` 把指南针 0AMV 变成可查询、可日更的数据源
- 旧路线 (`Rotation`、B1、B3、LTR 等) 已归档；详见 `strategies/archive-index.md`。

### 一句话定位

**对标博主的多策略 + regime 切换方向；实现上走活跃市值驱动的 AMV Bull Pool TopN，而不是复刻她的截面排序或 12 策略清单。**

---

## 二、信息源整合

### 早期公开信息

- 5 年日频回测基线指标
- 128 日日 K 量价窗口
- 80~500 亿流通市值
- 尾盘交易, 次日卖出, 20 持仓
- 1 分钟高频 T+0 为日内增强

### 后续公开评论 / 私信信息

- 2025 年上半年仍在研究多因子, 实盘也以动量因子为主
- 后来发现“截面多因子排序这一套存在诸多缺陷”, 开始转向 `rule-based`
- 6 月底开始切到新策略, 从 4 个策略逐步扩展到 12 个
- 纯量价策略只有 3 个, 其他策略混合基本面或另类数据
- 另类数据示例: 公司 CEO 年龄、年报页数
- 她也明确说过: 不是不做动量, 而是不做**截面排序**; trigger 变了

---

## 三、策略演化时间线

### 阶段 A: 日频截面多因子研究期

| 维度 | 信息 |
|---|---|
| 主体逻辑 | 日频截面多因子排序 |
| 数据主轴 | 过去 `128` 个交易日日 K 量价 |
| 标的范围 | A 股正股, 80~500 亿, 尾盘交易 |
| 日内增强 | 1 分钟级 T+0 |

### 阶段 B: 认知转折

- 多因子优点 (容量、抗过拟合、团队协作) 对个人未必是核心矛盾
- 纯量价 rule-based 跨环境难稳定 → 必须 **多策略组合**

### 阶段 C: 当前公开可见的实盘体系

| 维度 | 信息 |
|---|---|
| 总体形态 | 多策略组合, 非单模型 |
| 子策略数 | 从 `4` 个扩展到 `12` 个 |
| 切换方式 | 按不同市场状态自动切换 |
| 数据 | 约 3 个纯量价 + 其余混合基本面 / 另类数据 |

---

## 四、早期日频基线指标

> 博主某一阶段公开展示的日频能力参考，**不等于她当前完整体系，也不等于 QuantLab 当前目标**。

| 指标 | 值 |
|---|---|
| 5 年年化 | **50.42%** |
| 毛年化 | **90.25%** |
| 最大回撤 | **9.13%** |
| Sharpe (adj.) | **0.76** |
| 胜率 | **54.01%** |
| 盈亏比 | **3.04 : 2.02** |
| 日内增强 | 1 分钟 T+0 后年化再 `+10% ~ +20%` |

---

## 五、2025 年阶段性实盘总结

| 指标 | 信息 |
|---|---|
| 2025 收益率 | **38.8%** |
| 2025 最大回撤 | **5.89%** |
| 2025 Sharpe | **2.6** |
| 阶段特征 | 5 月前多因子研究, 6 月底后切到新策略 |
| 组合扩张 | 从 `4` 个策略逐步补充到 `12` 个 |

---

## 六、博主观点对我们的启发

- 截面多因子排序不是终局；**跨环境稳定** 比单模型回测漂亮更重要。
- 对个人量化，alpha / beta / timing / execution 应一体考虑。
- 多策略 + 状态切换是合理长期形态 — 这与我们围绕 **活跃市值 regime** 做 sleeve 分工的方向一致。

---

## 七、QuantLab 自有路线：活跃市值主线

### 为什么是活跃市值

- 活跃市值 (`0AMV`) 是指南针专利指标，无公开 API；通过 RPA 截图 + OCR 自建数据源（见 `rpa_capture/`、`rpa_parse/`）。
- 早期 B1 / Rotation 探索已证明: **稳定 alpha 来自 regime timing**（只在多头环境进攻），而不是在固定池子里无限堆排序因子。
- 活跃市值因此成为 QuantLab 的 **regime 主轴**，而不是某个已归档策略的附属字段。

### 当前实现结构

```
活跃市值 (0AMV) ──→ regime / bull 环境判断
        │
        ▼
AMV bull pool 候选 ──→ TopN 排序 (P-block / K-block / context / PB / CP / RV …)
        │
        ▼
bt-amv-topn (raw execution) ──→ 可交易结论
        │
        ├── 核心静态: Ref / P3 / context combo
        ├── 互补 sleeve: PB3 gated rolling
        └── 事件 sleeve: limit ecology (research)
```

### 当前策略分层（raw execution 口径）

| 角色 | 代表 | 状态 |
|---|---|---|
| 命名 baseline | Ref `6td static` | 对照 |
| 核心替换候选 | P3 `6td static` | 主线候选 |
| 最强静态 challenger | context combo `6td static` | forward monitor |
| Pullback 互补 | PB3 gated `6td rolling` | allocation 待评估 |
| 涨停生态 | limit weakgate `5td static` | research, 非 allocation-ready |

指标摘要见 `strategies/amv/status.py`，日常入口 `uv run python scripts/qlab.py status`。

### 与博主目标的对应关系

| 博主长期形态 | QuantLab 当前对应 | 差距 / 暂不追求 |
|---|---|---|
| 多策略组合 | AMV 下多条 sleeve (P3 / context / PB3 / limit) | 尚未做统一 allocation 层 |
| 市场状态切换 | 活跃市值 regime + PB3 gate / context 惩罚 | 分钟级 T+0 未做 |
| 日频底仓 | `6td static Top3` 为主 | 非尾盘 14:30 执行模型 |
| 混合基本面 / 另类数据 | 行业顺风、128 日结构等上下文因子 | CEO 年龄类另类数据未做 |

**结论**: 方向对齐（多 sleeve + regime），实现路径不同（活跃市值 + AMV TopN，而非复刻 12 策略或 128 日截面）。

---

## 八、跨设备统一约定

1. 本文负责 **外部对标 + 系统级长期目标**；不替代 `CURRENT_STATE.md` 的当前指标。
2. 讨论 QuantLab 当前主线时，默认指 **AMV Bull Pool TopN + raw execution**，不是 Rotation / B1 / B3。
3. 讨论“对标博主”时，默认指 **多策略 + regime 切换** 的方向，不是要求指标对齐她的 5 年年化或 12 策略清单。
4. 活跃市值数据与 RPA 运维见 `rpa_capture/README.md`、`rpa_parse/README.md`。
5. 已归档路线见 `strategies/archive-index.md`；更早实验流水只从 git history 回看。

---

## 九、阶段性结论

- 博主文档的价值，是防止我们把 **某一阶段的公开基线** 误当成 **她的完整体系**。
- QuantLab 已经走出自己的一条路: **以活跃市值为 regime 主轴，在 AMV bull pool 上做可执行 TopN 与 sleeve 分工**。
- 短期不追求复刻博主 12 策略或分钟 T+0；中期在 raw execution 验证过的 sleeve 上补 **allocation / regime 规则**；长期再评估是否扩展更多低相关 sleeve 或另类数据。
- 旧 `rotation-benchmark.md` 式错误 — 把不同阶段混成一个静态目标 — 不应在 AMV 主线上重演；当前数值与决策以 `CURRENT_STATE.md` 为准。
