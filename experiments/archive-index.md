# Archived Strategy Routes

> 本文件记录已经收口或暂不继续主线深挖的路线。目标是避免重复分析，同时保留以后回看的入口。

## 阅读规则

- `project-status.md` 是当前状态看板。
- `progress.md` 是完整实验流水账。
- 本文件只回答三个问题:
  - 为什么探索过这条路线？
  - 为什么暂时归档？
  - 以后什么情况下值得回看？

---

## B3 TDX / AMV Bull 接力

### 探索原因

- 用户提供 `strategies/tdx_scripts/b3选股代码.txt`，希望评估另一个 AMV bull 下的明确事件型策略。
- 原策略视频强调买入后有多种应对情况，是波段策略，不适合只用固定持有期评估。
- 因此新增:
  - `scripts/b3_tdx_signal_export.py`
  - `scripts/b3_candidate_ranking_lab.py`
  - `backtest-engine/crates/b3`

### 关键结果

- 固定 `6td`: 净收益 `+22.61%`, 最大回撤 `42.20%`
- B3 波段 v0: 净收益 `+18.49%`, 最大回撤 `43.39%`
- B1 `rw_dif_pct` 排序迁移: 净收益 `-28.32%`, 最大回撤 `56.31%`
- 全量 B3 候选 `6td`: 平均收益 `+1.02%`, 胜率 `49.87%`, 平均盈亏比 `1.50`
- 单字段 Top3 排序均未跑赢全部候选平均。

### 归档原因

- 结果明显弱于 `manual_p2_k0p5_r0_6td` 的 `+168.01% / MaxDD 14.97%`。
- 收益依赖少数大涨事件票，不像稳定可反复利用的股票池 alpha。
- 继续优化排序字段或波段参数的边际价值不高。

### 回看条件

- 后续做多策略组合时，需要低重合度事件型补充。
- 需要研究某类强异动 K 的事件脉冲特征。
- 需要复用 `bt-b3` 的波段出场框架做别的事件策略。

### 主要入口

- Canvas:
  - `b3-tdx-signal-backtest.canvas.tsx`
  - `b3-candidate-ranking-lab.canvas.tsx`
  - `b3-all-signal-payoff.canvas.tsx`
- Artifacts:
  - `artifacts/b3_tdx_signals/20260517_180746/backtests_v0_summary/report.json`
  - `artifacts/b3_tdx_signals/20260517_184650/backtests_rw_dif_pct/report.json`
  - `artifacts/b3_candidate_ranking_lab/20260517_185254/summary.json`

---

## Direct LTR Top3

### 探索原因

- 希望让模型直接从 AMV bull pool 中学习 Top3 选股，替代或增强手工规则。
- 初期 listwise LTR 在标签侧看起来有较强 edge。

### 关键结果

- `kbar_momentum_old_state` Rust `6td`: `-66.97%`
- `no_risk_old_state` Rust `6td`: `-0.01%`
- 执行口径校准后，`T+1 open -> T+N close` 标签下当前 LTR 不如简单 `ret_5d` 基线。

### 归档原因

- 原始高收益主要来自 `close-to-close` 标签和真实执行口径错配。
- 真实 `T+1 open` 买入后 edge 明显塌缩。
- 直接让模型选 Top3 的问题太难，且容易学习到不可执行或兑现不了的条件。

### 回看条件

- 重新定义 executable label。
- 不再直接选 Top3，而是作为状态特征或低维 gating 输入。
- 固定更窄问题，例如只在一个 sleeve 内排序。

### 主要入口

- Canvas:
  - `amv-ltr-rust-backtest.canvas.tsx`
  - `amv-ltr-robustness-test.canvas.tsx`
  - `amv-ltr-selection-attribution.canvas.tsx`
  - `amv-ltr-variant-ablation.canvas.tsx`

---

## `attack_ok` 第一版

### 探索原因

- 受 AMV 受约束 Oracle 启发，希望从完整 sleeve selector 降维成“是否应该进攻”的二分类。
- 标签来自 `top_ret_dailyized + 3% margin + allow_cash=true`。

### 关键结果

- 2023/2024 AUC 接近随机。
- 2025 略高。
- 2026 样本太少，不足以证明有效。
- 验证集 F1 阈值退化为接近 `always_attack`。

### 归档原因

- 当前交易前状态特征没有证明能稳定学习“什么时候进攻”。
- 标签仍使用 future-best attack sleeve，经济指标只能作为标签侧诊断。
- 第一版问题过抽象，噪声太大。

### 回看条件

- 固定单一 attack sleeve 后重新建标签。
- 提高 attack 门槛，追求更高 precision。
- 先完成 `cash_ok`，减少必须在 base/attack 中选择的噪声。

### 主要入口

- `scripts/amv_attack_ok_lab.py`
- `artifacts/amv_attack_ok/20260517_150811/summary.json`
- `amv-attack-ok-lab.canvas.tsx`

---

## Full Oracle Sleeve Selector

### 探索原因

- 用 hindsight oracle 评估 sleeve switching 的理论上限。
- 理解不同 factor sleeve 和 holding horizon 的潜在切换空间。

### 关键结果

- 完整 oracle 上限极高，但未来函数极重。
- Horizon-aware oracle 显示不同 sleeve 的最优持有周期不同。
- Explainability 显示完整 8 类 class 的状态区分度偏弱。

### 归档原因

- 完整 oracle 只适合作为上限诊断。
- 不适合作为第一版直接训练目标。
- 当前应以 AMV 受约束 Oracle 为理论锚点。

### 回看条件

- 需要重新评估 theoretical upper bound。
- 新增稳定 sleeve 后需要重新比较上限。

### 主要入口

- `amv-oracle-sleeve-rust-backtest.canvas.tsx`
- `amv-horizon-aware-oracle.canvas.tsx`
- `amv-horizon-oracle-explainability.canvas.tsx`
- `amv-constrained-oracle.canvas.tsx`

---

## Rotation 旧路线

### 探索原因

- Rotation 是较早的横截面机器学习主线，用于复刻/接近博主早期公开的日频截面基线。
- 后续加入 Alpha158、kbar_shape、hold_buffer、entry_rank 等优化。

### 当前状态

- 当前定位: 候选子策略，不再是主线。
- 可保留候选: `core_plus_alpha158(kbar_shape)`
- 旧路线收口:
  - `46-all / 36-pruned`: 不再主推
  - `core_plus_alpha158_top1`: 失败
  - `rank_pct / rank_gauss`: 弱于 `zscore`
  - `alpha158(kbar_shape)` 单跑: 是交互增强器，不是独立主线

### 归档原因

- AMV bull pool 方向已经成为更高优先级主线。
- Rotation 的成本、换手和收益兑现仍不如当前 AMV TopN 底座。

### 回看条件

- 需要非 AMV 的横截面子策略。
- 需要多策略组合中的独立收益来源。

### 主要入口

- `experiments/rotation-factors.md`
- `experiments/rotation-next-phase.md`
- `notebooks/cross_section_rotation.py`
- `notebooks/rotation_factor_lab.py`

---

## B1 专属模型

### 探索原因

- B1 是早期事件型/形态策略，尝试通过专属 ML 或全市场 ML 排序增强。

### 关键结论

- B1 专属模型不如全市场 ML / 手搓基线稳定。
- B1 字段如 `rw_dif_pct` 可作为诊断字段，但不能直接迁移为其它策略排序主轴。

### 归档原因

- 当前主线已转向 AMV bull pool。
- B1 形态规则的边际 alpha 不足以继续作为主线。

### 回看条件

- 需要研究波段出场机制。
- 需要借用 B1 的字段或事件结构做辅助诊断。

### 主要入口

- `experiments/b1-ml-fullmarket.md`
- `experiments/b1-ml-dedicated.md`
- `experiments/b1-next-phase.md`
- `backtest-engine/crates/b1`
