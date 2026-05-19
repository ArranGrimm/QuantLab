# Project Status

> 本文件是“当前状态看板”，只保留当前结论、关键指标、归档路线和索引。实验流水账见 `progress.md`，长篇分析见 Canvas 与 `experiments/`。

---

## 当前决策摘要

- 当前主策略底座: `manual_p2_k0p5_r0_6td`
  - 修正买入手数口径后，净收益更新为 `+170.80%`
  - 最大回撤 `15.30%`
  - 当前仍是 AMV bull pool 下最强可交易基线
- 最新 P/K/M 验证: `amv-pkm-sleeve-rust-backtest.canvas.tsx`
  - 标签侧动量增强没有兑现为更好的 Rust 主基线
  - 新手数口径下 `P3/K1/M2` 最好，净收益 `+103.58%`，但仍弱于主基线 `+170.80%`
  - `P1/K0.5/M1 / P3/K1/M2` 明显改善 2025，但 MaxDD 仍约 `48%`
  - `1td` 到 `7td` 持仓周期扫描后，P/K/M 最佳仍是 `6td`；`4td / 5td / 7td` 也没有改善结论
  - 2026 YTD 亏损比当前主基线更深，因此 P/K/M 不作为默认替代
- 待验证 rolling cohort 口径:
  - 已完成第一版真实组合测试: `5td rolling18` 与 `6td rolling21`
  - manual 最好为 `6td rolling21`: net `+23.61%`, MaxDD `9.33%`
  - P/K/M rolling 未兑现标签侧优势，最好仅 `P2/K0.5/M0.5 6td rolling21`: net `+0.21%`
  - 当前 rolling 实现会跳过已持有代码，因此是“不重复加仓”的真实账户口径
  - 买入手数已修正为普通 A 股 `100` 股整数倍、科创板 `sh.688*` 最少 `200` 股后按 1 股递增
- close-to-close 诊断:
  - 新增 `bt-amv-cohort-diagnostic`，用于定位 Python rolling NAV 与真实 Rust rolling cohort 的损耗断点
  - 第一版 B 口径: close-to-close、no repeat、with costs、资金/手数约束、close 涨停不可买、默认 `max_open_gap_pct = 0.098`
  - 配套新增 unshifted 诊断信号导出脚本，避免和 `T+1 open` 真实交易信号混用
  - B 口径 `6td rolling21`: manual `+20.88%`; P/K/M 最好 `P2/K0.5/M0.5 +15.31%`
  - P/K/M close 涨停不可买过滤很重: `P1 812 / P2 602 / P3 856` 条，说明 Python rolling NAV 高收益大量来自不可成交信号
- Python refill rolling NAV 诊断:
  - 新增 `scripts/amv_limit_refill_rolling_nav.py`，用于验证“Top3 涨停则顺位补位”的标签侧上限
  - `6d` 原始 Top3 NAV -> 跳过 close 涨停并补满 Top3: manual `+248.54% -> +90.33%`; P1 `+1054.55% -> +55.71%`; P2 `+768.74% -> +53.72%`; P3 `+1045.97% -> +31.32%`
  - 结论: P/K/M 的千百分比级 rolling NAV 主要来自 close 涨停不可买样本；补位后仍有正 alpha，但不再超过 manual，且仍不是可交易口径
- Executable-aware 权重网格 v2:
  - 新增 `scripts/amv_executable_weight_grid.py`，重开早期 AMV 因子/组合/权重探索
  - Canvas: `reports/canvases/amv-executable-weight-grid-v2.canvas.tsx`
  - 主评估改为 `D+1 open -> D+7 close`，辅助保留 `D close -> D+6 close`
  - 首轮 `6d` 全量 `90` 个 ranker: `P3/K0.5/R0` exec NAV `+160.14%`, 当前 reference `P2/K0.5/R0` exec NAV `+152.21%`
  - 带动量 P/K/M 的 close-to-close NAV 仍为 `+768% ~ +1055%`，但 executable NAV 仅 `+29% ~ +33%`，且 close 涨停覆盖 `59.7% ~ 76.4%` 天
  - 当前判断: 早期“高位 + K 线确认、低/无动量”结构在可执行口径下仍最可信；`P3/K0.5/R0` 可列入 Rust 真实回测候选，但主基线暂不变
- Executable-aware 早期全因子扫描:
  - 新增 `scripts/amv_executable_factor_scan.py`，重跑早期 `RANKERS + COMBO_RANKERS` 共 `47` 个 ranker
  - Canvas: `reports/canvases/amv-executable-factor-scan.canvas.tsx`
  - 低污染强候选: `ma_bias_20_asc` exec NAV `+231.03%`, `disp_bias_20_asc` `+187.54%`, `KSFT_asc` `+178.36%`
  - 这些候选 close 涨停覆盖仅 `0.0% ~ 0.7%` 天，且跳过 close 涨停补位后几乎不衰减
  - `ret_5d_desc` exec NAV `+264.07%` 最高，但 MaxDD `47.11%`、close 涨停覆盖 `69.0%` 天，仍归类为 attack/event archetype
  - 当前判断: 下一轮组合探索应优先围绕“回调/成本线下回归/收盘偏低”与现有 P/K 结构组合，而不是继续加动量权重
- Executable-aware pullback combo grid v1:
  - 新增 `scripts/amv_executable_pullback_grid.py`
  - Canvas: `reports/canvases/amv-executable-pullback-grid.canvas.tsx`
  - focused grid `164` 个 ranker 已跑通；full grid `618` 个 ranker 已在当前 Mac 复跑完成
  - full grid 产物: `artifacts/amv_executable_pullback_grid/20260519_213813/summary.json`
  - 最强低污染组合 `pullback_p0_k0_b1_c0_r0`: exec NAV `+245.37%`, MaxDD `23.29%`, close 涨停覆盖 `0.5%`
  - `pullback_p0_k0_b3_c1_r0`: exec NAV `+215.37%`, MaxDD `20.29%`, close 涨停覆盖 `0.0%`
  - full grid 新增折中候选 `pullback_p0_k0_b2_c0p5_r0`: exec NAV `+210.37%`, MaxDD `21.89%`, close 涨停覆盖 `0.0%`
  - refill 场景低回撤混合候选 `pullback_p2_k0_b0_c0p5_r0p5`: exec NAV `+167.31%`, MaxDD `5.73%`, rank q95 `4`
  - `P/K + C` 候选如 `pullback_p2_k0_b0_c1_r0` refill exec NAV `+155.87%`, MaxDD `10.14%`, rank q95 `5`
  - 当前判断: pullback sleeve 成立，但回撤深于主基线；应作为独立 sleeve / 互补 sleeve 候选接 Rust 真实回测，不直接替换 `manual_p2_k0p5_r0_6td`
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
  - 持有: `6td`，当前已明确为 `max_hold_trading_days = 6` 的交易日口径
  - TopN: `3`
  - 固定止损: 当前主基线为 no-stop
- 修正后核心指标:
  - `1td`: 净收益 `-51.03%`
  - `2td`: 净收益 `-37.33%`
  - `3td`: 净收益 `+38.11%`
  - `6td`: 净收益 `+170.80%`, 最大回撤 `15.30%`
- 当前判断:
  - `manual_p2_k0p5_r0_6td` 是当前最强底座
  - 主要问题不是继续找新单策略，而是识别什么时候该避开底座、什么时候可以进攻
  - 2024 大赢家来自 AMV bull 叠加高弹性行情窗口；2026 亏损段主要是入场后上冲空间不足

### 下一步候选

- 第一优先: `cash_ok` / 降仓标签
  - 目标是识别是否应该避开 `manual_p2_k0p5_r0_6td` 的亏损日
  - 理论依据来自 AMV 受约束 Oracle 中的 cash 上限
- 第二优先: executable-aware 权重网格 v2 后续验证
  - 先把 `P3/K0.5/R0` 等可执行口径高分候选导出为静态 sleeve，接 Rust `T+1 open / 6td / Top3 / no-stop`
  - 后续所有因子/权重探索必须同时输出 executable 主指标、close-to-close 辅助指标和涨停/高开污染归因
- 第三优先: executable-aware 回调因子组合
  - 基于全因子扫描，把 `ma_bias_20_asc / disp_bias_20_asc / KSFT_asc` 与现有 P/K 结构组合
  - 目标是检验“高位强势”和“回调后反弹”能否形成互补 sleeve，而非继续强化涨停动量
  - full pullback grid 已证明低污染 pullback sleeve 成立；下一步导出 3-6 个候选做 Rust 静态 sleeve 回测
- 第四优先: rolling cohort 真实组合口径
  - 第一版结果显示 rolling cohort 没有帮助 P/K/M 追上 manual
  - 后续若继续，应先分析为什么 rolling 后收益被摊薄: 重复代码跳过、现金利用率、买入失败、分散化稀释
  - 在该诊断前，不把 rolling cohort 作为默认替代
- 第五优先: close-to-close cohort diagnostic
  - B 口径已证明 close 涨停不可买会大幅压低 P/K/M 的 close-to-close 上限
  - Python refill 诊断进一步证明，即使顺位补满 Top3，P/K/M 标签侧收益也从千百分比级降至几十个百分点
  - 若继续，需要做过滤归因: close 涨停过滤、重复代码跳过、账户分散、T+1 open 损耗分别贡献多少
  - 暂不急着做 A 口径，除非要精确复刻 Python rolling NAV 并拆解其不可成交成分
- 第六优先: 回到主策略入场/环境确认
  - 2026 亏损段主要是入场后上冲空间不足，P/K/M 排序权重没有解决
  - 关注执行日是否明显收弱、AMV bull 中短期市场宽度、赚钱效应、涨停阻塞率等可观测变量
- 第七优先: 收紧 `attack_ok`
  - 第一版 `attack_ok` 太抽象，且 future-best attack sleeve 噪声较高
  - 后续如继续，应固定单一进攻袖子或提高标签门槛，追求更高 precision

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

- `amv-pkm-sleeve-rust-backtest.canvas.tsx`: P/K/M 动量增强袖子 Rust 真实回测
- `amv-yearly-weight-grid.canvas.tsx`: 年度权重网格与 P/K/M 动量增强诊断
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
