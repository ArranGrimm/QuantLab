# Project Status

> 本文件是“当前状态看板”，只保留当前结论、关键指标、归档路线和索引。实验流水账见 `progress.md`，长篇分析见 Canvas 与 `experiments/`。

---

## 当前决策摘要

- 当前主策略底座: `manual_p2_k0p5_r0_6td`
  - 修正涨跌停口径后仍为净收益 `+168.01%`
  - 最大回撤 `14.97%`
  - 当前仍是 AMV bull pool 下最强可交易基线
- 当前理论锚点: `amv-constrained-oracle.canvas.tsx`
  - 后续讨论 sleeve switching、降仓、进攻切换时优先从“AMV 受约束 Oracle”出发
  - 不再回到完整 hindsight oracle 或 8 类 sleeve selector 重新推导
- 当前不继续主线深挖:
  - 直接 LTR 选 Top3: 标签错配和执行口径塌缩，暂不接交易
  - `attack_ok` 第一版: 当前状态特征未证明可稳定学习进攻切换
  - B3 TDX 接力: 已归档为事件型补充候选，不作为主线继续优化
- 文档约定:
  - `project-status.md`: 当前状态、最终决策、关键指标
  - `progress.md`: 按日期倒序的实验日志
  - `experiments/` / Canvas: 长篇分析、历史归档、可视化结果

---

## 当前主线

### AMV Bull Pool TopN

- 当前策略底座: `manual_p2_k0p5_r0_6td`
- Rust 口径:
  - 引擎: `bt-amv-topn`
  - 执行: `T+1 open`
  - 持有: `6td`
  - TopN: `3`
  - 固定止损: 当前主基线为 no-stop
- 修正后核心指标:
  - `1td`: 净收益 `-51.03%`
  - `2td`: 净收益 `-37.33%`
  - `3td`: 净收益 `+38.11%`
  - `6td`: 净收益 `+168.01%`, 最大回撤 `14.97%`
- 当前判断:
  - `manual_p2_k0p5_r0_6td` 是当前最强底座
  - 主要问题不是继续找新单策略，而是识别什么时候该避开底座、什么时候可以进攻
  - 2024 大赢家来自 AMV bull 叠加高弹性行情窗口；2026 亏损段主要是入场后上冲空间不足

### 下一步候选

- 第一优先: `cash_ok` / 降仓标签
  - 目标是识别是否应该避开 `manual_p2_k0p5_r0_6td` 的亏损日
  - 理论依据来自 AMV 受约束 Oracle 中的 cash 上限
- 第二优先: 收紧 `attack_ok`
  - 第一版 `attack_ok` 太抽象，且 future-best attack sleeve 噪声较高
  - 后续如继续，应固定单一进攻袖子或提高标签门槛，追求更高 precision
- 第三优先: 回到主策略入场/环境确认
  - 关注执行日是否明显收弱、AMV bull 中短期市场宽度、赚钱效应、涨停阻塞率等可观测变量

---

## 理论锚点

### AMV 受约束 Oracle

- Canvas: `amv-constrained-oracle.canvas.tsx`
- 脚本: `scripts/amv_constrained_oracle_lab.py`
- 关键设定:
  - base: `manual_p2_k0p5_r0_6td`
  - attack: `ret_5d_6td / ret_20d_6td / ret_20d_2td / kmid2_6td`
  - cash: base 和 attack 都差时允许空仓
- 关键结论:
  - “主策略 + 例外切换”有足够理论空间
  - 按持有期总收益训练会变成高频 attack，不像少数例外
  - 按日化收益并设置 `3% margin` 后，attack 频率明显下降，更像可学习 gating 目标
  - 完整 oracle 继续只作为上限诊断，不作为直接训练目标

### Oracle Explainability

- Canvas:
  - `amv-horizon-aware-oracle.canvas.tsx`
  - `amv-horizon-oracle-explainability.canvas.tsx`
  - `amv-constrained-oracle.canvas.tsx`
- 关键结论:
  - 完整 8 类 sleeve selector 当前不可直接学习
  - 状态特征对完整 oracle class 的区分度偏弱
  - 更合理入口是低维 gating: `base_ok / cash_ok / attack_ok`

---

## 归档路线

### Direct LTR Top3

- 状态: 已归档，暂不接交易
- 核心原因:
  - 训练标签与真实执行口径错配
  - `close-to-close` 高收益无法兑现到 `T+1 open -> T+N close`
  - Rust 真实回测显著弱于 `manual_p2_k0p5_r0_6td`
- 关键结果:
  - `kbar_momentum_old_state` Rust `6td`: `-66.97%`
  - `no_risk_old_state` Rust `6td`: `-0.01%`
- 后续回看条件:
  - 只在重新定义 executable label、固定更窄问题、或作为状态特征辅助时回看

### `attack_ok` 第一版

- 状态: 已归档为第一版失败实验
- 脚本: `scripts/amv_attack_ok_lab.py`
- Canvas: `amv-attack-ok-lab.canvas.tsx`
- 关键结果:
  - 2023/2024 AUC 接近随机
  - 2025 仅略高
  - 2026 样本太少，不足以证明有效
  - 验证集 F1 阈值退化为接近 `always_attack`
- 当前判断:
  - 当前状态特征未证明可稳定学习“什么时候进攻”
  - 后续如果继续，应收紧标签或固定单一进攻袖子

### B3 TDX / AMV Bull 接力

- 状态: 已归档为事件型补充候选，不作为主线继续深挖
- 保留资产:
  - `scripts/b3_tdx_signal_export.py`
  - `scripts/b3_candidate_ranking_lab.py`
  - `backtest-engine/crates/b3`
  - `b3-tdx-signal-backtest.canvas.tsx`
  - `b3-candidate-ranking-lab.canvas.tsx`
  - `b3-all-signal-payoff.canvas.tsx`
- 收口依据:
  - 固定 `6td`: 净收益 `+22.61%`, 最大回撤 `42.20%`
  - 波段 v0: 净收益 `+18.49%`, 最大回撤 `43.39%`
  - B1 `rw_dif_pct` 排序迁移: 净收益 `-28.32%`, 最大回撤 `56.31%`
  - 全量候选 `6td`: 平均收益 `+1.02%`, 胜率 `49.87%`, 平均盈亏比 `1.50`
  - 单字段 Top3 排序均未跑赢全部候选平均
- 最终判断:
  - B3 有一定事件型 edge，但收益依赖少数大涨票
  - 不像稳定可反复交易的主策略 alpha
  - 后续只在组合/低重合度补充时回看

### Rotation 旧路线

- 状态: 候选子策略，不作为当前主线
- 当前主线候选仍是 `core_plus_alpha158(kbar_shape)`，但 AMV 方向已经优先级更高
- 旧路线收口:
  - `46-all / 36-pruned`: 不再主推
  - `core_plus_alpha158_top1`: 已验证失败
  - `rank_pct / rank_gauss`: 弱于 `zscore`
  - `alpha158(kbar_shape)` 单跑: 是交互增强器，不是独立主线
- 后续回看条件:
  - 当需要横截面多策略组合或非 AMV 子策略时再回看

### B1 专属模型

- 状态: 已归档
- 结论:
  - B1 专属 ML 不如全市场 ML / 手搓基线稳定
  - B1 的部分字段可作为其它策略的诊断字段，但不能直接迁移为排序字段
- 相关文档:
  - `experiments/b1-ml-fullmarket.md`
  - `experiments/b1-ml-dedicated.md`
  - `experiments/b1-next-phase.md`

---

## 关键基线

### AMV Static Sleeve

- 主要可保留候选:
  - `manual_p2_k0p5_r0_6td`: 当前主策略底座
  - `ret_5d_6td`: 有正收益但回撤大
  - `ret_20d_2td`: 资金效率角度可观察，但不是 6td 主基线
- 关键 Canvas:
  - `amv-static-sleeve-backtest.canvas.tsx`
  - `amv-bull-pool-factor-regime.canvas.tsx`
  - `amv-bull-pool-yearly-factor.canvas.tsx`

### Price Limit 口径

- `bt_core::price_limit_pct` 已修复，支持 `sz.300xxx / sh.688xxx` 等 QMT 前缀代码
- `cargo test -p bt-core` 已通过
- 影响:
  - B3 结果受影响较明显
  - `manual_p2_k0p5_r0_6td` 基线基本不受影响

---

## 文档索引

### 当前必读

- `progress.md`: 实验流水账，按日期倒序
- `experiments/archive-index.md`: 已归档路线索引
- `experiments/target-strategy-evolution.md`: 博主策略演化与多策略全景

### 关键 Canvas

- `amv-constrained-oracle.canvas.tsx`: 当前理论锚点
- `amv-static-sleeve-backtest.canvas.tsx`: 静态袖子 Rust 对比
- `amv-bull-pool-factor-regime.canvas.tsx`: 因子分年份 / 分 regime 标签分析
- `amv-topn-6td-trade-analysis.canvas.tsx`: 6td 交易归因
- `amv-topn-segment-attribution.canvas.tsx`: 分段归因
- `amv-attack-ok-lab.canvas.tsx`: attack_ok 第一版失败实验
- `b3-all-signal-payoff.canvas.tsx`: B3 收口依据

---

## 维护规则

- 新实验先写入 `progress.md`
- 只有形成稳定结论后才同步到本文件
- 已归档路线只保留最终判断、关键指标和回看条件
- 不在本文件保留完整实验过程、长表格或逐轮参数记录
