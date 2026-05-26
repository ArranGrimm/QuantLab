# Project Status

> 本文件是“当前状态看板”，只保留当前结论、关键指标、归档路线和索引。实验流水账见 `progress.md`，长篇分析见 Canvas 与 `experiments/`。

---

## 当前决策摘要

- 当前主策略底座: `manual_p2_k0p5_r0_6td`
  - 修正买入手数口径后，净收益更新为 `+170.80%`
  - 最大回撤 `15.30%`
  - 当前仍作为 reference baseline；`candidate_p3_k0p5_b0_c0_r0` 已成为强替换候选，净收益 `+201.69%`, MaxDD `13.52%`
  - P3 归因 Canvas: `reports/canvases/amv-p3-vs-ref-trade-attribution.canvas.tsx`
  - P3 的优势主要来自 `30` 笔边际替换交易，而非整体交易池重写；换票机制已解释为更偏向 P-block 的高位突破延续，但仍需后续样本确认稳定性
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
  - 最强低污染组合 `pullback_p0_k0_pb1_cp0_rv0`: exec NAV `+245.37%`, MaxDD `23.29%`, close 涨停覆盖 `0.5%`
  - `pullback_p0_k0_pb3_cp1_rv0`: exec NAV `+215.37%`, MaxDD `20.29%`, close 涨停覆盖 `0.0%`
  - full grid 新增折中候选 `pullback_p0_k0_pb2_cp0p5_rv0`: exec NAV `+210.37%`, MaxDD `21.89%`, close 涨停覆盖 `0.0%`
  - refill 场景低回撤混合候选 `pullback_p2_k0_pb0_cp0p5_rv0p5`: exec NAV `+167.31%`, MaxDD `5.73%`, rank q95 `4`
  - `P/K + CP` 候选如 `pullback_p2_k0_pb0_cp1_rv0` refill exec NAV `+155.87%`, MaxDD `10.14%`, rank q95 `5`
  - 当前判断: pullback sleeve 成立，但回撤深于主基线；应作为独立 sleeve / 互补 sleeve 候选接 Rust 真实回测，不直接替换 `manual_p2_k0p5_r0_6td`
- Executable/Pullback sleeves Rust TopN 验证:
  - 新增 `scripts/amv_static_sleeve_signal_export.py` 的 6 个 executable/pullback sleeve 导出，并新增 `bt-amv-topn` 静态/rolling strict/refill 6td 配置
  - Canvas: `reports/canvases/amv-executable-sleeve-rust-complement.canvas.tsx`
  - pullback 命名已收口为 `PB/CP/RV`: `PB = ma_bias_20 + disp_bias_20`, `CP = KSFT + intraday_pos`, `RV = atr_14_pct + panic_vol_ratio_20d`
  - 静态 strict Top3: `candidate_p3_k0p5_b0_c0_r0` net `+201.69%`, MaxDD `13.52%`，优于 reference `+170.80%` / `15.30%`
  - 新命名 artifact 已复跑: `20260520_105222_pullback_p0_k0_pb1_cp0_rv0` 到 `20260520_105228_pullback_p2_k0_pb0_cp0p5_rv0p5`，结果与旧命名一致
  - 静态 strict pure pullback: `PB1/CP0/RV0` net `+190.28%` 但 MaxDD `43.43%`; `PB3/CP1/RV0` net `+152.84%`, MaxDD `41.85%`
  - rolling21 refill Top10: `PB3/CP1/RV0` net `+99.62%`, MaxDD `20.70%`; `PB2/CP0.5/RV0` net `+96.06%`, MaxDD `22.74%`; `PB1/CP0/RV0` net `+89.41%`, MaxDD `21.28%`
  - rolling21 reference/P3 仍低: reference refill `+21.44%`, P3 refill `+22.19%`; P3 strict `+30.65%`
  - 当前判断: `P3/K0.5/R0` 是主基线替换候选；pure pullback 是 rolling/互补 sleeve 候选，不是静态主策略替代
- P3 vs Ref 主基线替换归因:
  - Canvas: `reports/canvases/amv-p3-vs-ref-trade-attribution.canvas.tsx`
  - 两者各 `274` 笔交易，exact overlap (`entry_date + code`) 为 `244` 笔，重合率 `89.05%`
  - P3-only `30` 笔合计 PnL `+170,899`; Ref-only `30` 笔合计 PnL `+29,737`; 边际换票贡献约 `+141,161`
  - 年度上 P3 明显修复 2026: Ref `-8.80%` -> P3 `-0.77%`; 但 P3 弱于 Ref 的年份包括 2021 `-1.76pp` 和 2024 `-3.23pp`
  - 关键正贡献月份: `2026-01` trade PnL delta `+129,226`; `2023-04` trade PnL delta `+73,186`
  - 关键换票: P3 买入 `sz.003035` `+92,771` 且避开 Ref 的 `sh.688789` `-35,846`; 但也错过 Ref 的 `sz.000559` `+83,373` 和 `sz.002271` `+51,608`
  - 换票特征分解: `reports/amv_p3_ref_swap_feature_explain.json`
  - 当前判断: P3 是强替换候选，但不是完全定版；机制上偏向“更贴近 20 日高点/新高但 K 线项较弱”的突破延续票，仍需 forward 监控验证可重复性
- Executable sleeves 年度互补性:
  - P/K 主线内部高度相关: reference static vs P3 static daily return corr `0.916`，P3 更像替换而非互补
  - P/K 主线 vs rolling pullback 低相关: P3 static vs `PB3/CP1 rolling` corr `0.255`，vs `PB2/CP0.5 rolling` corr `0.243`，vs `PB1/CP0 rolling` corr `0.214`
  - rolling pullback 家族内部高度重叠: `PB3/CP1` vs `PB2/CP0.5` corr `0.988`，所以不应堆多个 pullback sleeve
  - 2026 YTD: reference static `-8.80%`, P3 static `-0.77%`, `PB3/CP1 rolling +15.15%`, `PB2/CP0.5 rolling +15.25%`, `PB1/CP0 rolling +12.32%`
  - 简单 daily rebalance 诊断: `P3 static + PB3/CP1 rolling` 80/20 total `+183.05%`, MaxDD `11.60%`, 2026 `+2.35%`; 70/30 total `+173.03%`, MaxDD `10.81%`, 2026 `+3.93%`
- Rolling pullback 代表选择:
  - 当前代表: `PB3/CP1/RV0 rolling21 refill`
  - 决策报告: `reports/amv_pullback_representative_choice.json`
  - `PB3/CP1` vs `PB2/CP0.5`: total return `+99.62%` vs `+96.06%`, MaxDD `20.70%` vs `22.74%`, exact overlap `1455`, daily corr `0.988`
  - `PB2/CP0.5` 保留为 forward challenger，不与 PB3 同时堆叠；`PB1/CP0` 暂不作为代表
- 原始 B1 executable-aware lab:
  - 新增脚本: `scripts/b1_executable_base_lab.py`
  - 产物: `artifacts/b1_executable_base_lab/20260520_142833/summary.json`
  - 原始三条件 `close > YL / WL > YL / J <= 13` 在 AMV bull + liquidity 下候选平均 exec NAV `+53.28%`, MaxDD `20.66%`
  - `J` 越低 Top3 仅 `+36.71%`，说明 `J <= 13` 更适合作为候选过滤而非排序主轴
  - B1 池内 pullback 排序更强: `PB2/CP0.5` exec NAV `+89.21%`, MaxDD `26.07%`; `PB3/CP1` exec NAV `+80.87%`, MaxDD `27.19%`
  - 当前判断: 原始 B1 是 pullback 机制旁证，不是独立新主线；现版复杂 B1 失败更可能来自额外约束和排序设计，而不是原始三条件完全无效
- B1 trend-only 对照:
  - `scripts/b1_executable_base_lab.py` 已支持 `--base-mode trend_only`
  - 产物: `artifacts/b1_executable_base_lab/20260520_143434/summary.json`
  - 只保留 `close > YL / WL > YL` 后，候选池扩到 `260,164` 行、`516` 个信号日、平均每天约 `504.2` 个候选
  - `trend_only + P3/K0.5` 跳过 close 涨停补位后 exec NAV `+118.74%`, MaxDD `7.27%`
  - `trend_only + PB2/CP0.5` refill exec NAV `+89.22%`, MaxDD `34.10%`; `PB3/CP1` refill exec NAV `+89.34%`, MaxDD `31.91%`
  - 当前判断: `J <= 13` 会压制突破排序，trend-only 更像一个趋势候选池；`P3/K0.5` trend-only 值得后续接 Rust strict/refill 验证，但不能用未过滤 close-to-close 结果下结论
- Trend-only executable focused grid:
  - 新增脚本: `scripts/amv_executable_trend_filter_grid.py`
  - focused 产物: `artifacts/amv_executable_trend_filter_grid/20260520_145545/summary.json`
  - 扫描 `factor + pullback focused + yearly` 共 `301` 个 ranker，候选池 `260,164` 行、`516` 个信号日、`2,282` 只股票
  - 全部 trend-only 候选平均 exec NAV `+70.40%`, MaxDD `21.20%`; close limit-up day share `94.0%`，所以优先看 refill
  - Top refill: `P1/K0/PB1/CP0/RV0.5` exec NAV `+213.74%`, MaxDD `4.26%`, ctc NAV `+161.58%`, rank q95 `3`
  - 其他强候选: `P1/K0.5/PB1/CP0/RV0.5` `+204.40%` / MaxDD `4.87%`; `P3/K0.5/PB2/CP1/RV0.5` `+195.26%` / MaxDD `4.78%`
  - Mac full grid 已跑通: `755` 个 ranker，产物 `artifacts/amv_executable_trend_filter_grid/20260520_212625/summary.json`
  - Full top refill: `P1/K0.5/PB1/CP0/RV1` exec NAV `+256.94%`, MaxDD `4.51%`, ctc NAV `+207.10%`; `P1/K1/PB1/CP0/RV1` `+243.77%` / MaxDD `4.38%`
  - Rust 验证报告: `reports/amv_trend_filter_rust_backtest_summary.json`
  - Rust 最好结果: `trend P1/K0/PB1/CP0/RV0.5` static refill net `+64.46%`, MaxDD `41.06%`; rolling21 refill net `+35.02%`, MaxDD `14.10%`
  - 已定位并修正一个导出口径差异: Python grid 在 trend 候选池内计算组件 rank，初版导出在全市场内计算组件 rank
  - 修正后报告: `reports/amv_trend_filter_corrected_export_rust_summary.json`
  - 修正后 `trend P1/K0/PB1/CP0/RV0.5`: static refill net `+112.54%`, MaxDD `38.27%`; rolling21 refill net `+41.47%`, MaxDD `11.66%`
  - Full top 新候选 Rust 报告: `reports/amv_trend_filter_full_top_rust_summary.json`
  - Full top 新候选 Canvas: `reports/canvases/amv-trend-full-rust-conversion.canvas.tsx`
  - Full top Rust 最好 static: `trend P2/K0.5/PB2/CP0/RV1` net `+123.58%`, MaxDD `38.27%`
  - Full top Rust 最好 rolling: `trend P1/K1/PB1/CP0/RV1` net `+60.17%`, MaxDD `11.58%`
  - 当前判断: full grid 抬高了 trend-only label 上限，且涨停污染不是主因；但 Rust gross edge 自身掉到 `+90%~95%` rolling，低于 `PB3/CP1/RV0` rolling gross `+130.78%`，暂不进入主线候选
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
  - `.agents/skills/amv-trade-attribution/SKILL.md`: AMV 回测对比、交易归因、换票解释、成本损耗和互补性分析的项目级工作流
  - `scripts/backtest_trade_attribution.py`: `bt-amv-topn` artifact 通用交易归因入口

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

- 第一优先: `P3/K0.5/R0` 主基线替换验证
  - Rust 静态 strict 已从 reference `+170.80%` / MaxDD `15.30%` 提升到 `+201.69%` / MaxDD `13.52%`
  - 年度/交易归因显示优势主要来自 `30` 笔边际换票；2026-01 与 2023-04 的换票机制已解释，下一步看更多样本和 forward 监控后再正式切换
  - P3 static cadence 敏感性报告: `reports/amv_p3_static_cadence_sensitivity.json`
  - 7 个起始 offset 的 no-cost Python-like static 路径最差 `+260.74%`、中位 `+285.60%`、最好 `+297.79%`；粗扣 `0.35%` 单轮往返成本后最差约 `+162.19%`、中位 `+181.25%`，说明 static 不是单一起点侥幸，但最终仍以 Rust net 为准
- 第二优先: rolling pure pullback sleeve 验证 + gating
  - 已选 `PB3/CP1/RV0 rolling21 refill` 作为代表袖子，rolling21 refill Top10 为 `+99.62%`, MaxDD `20.70%`
  - **新增 PB3 rolling gating 规则**: `(aged + 非加速) OR (neg_streak>=3 & amp>2.5%)`，跳过 18.5% 交易、净赚 `+52K`（+10.5%）、零误杀
  - 下一步: 在 `bt-amv-topn` 配置中接入该 ruleset，Rust 验证 what-if 结论；`PB2/CP0.5/RV0` 仅作为 challenger 监控
- P3 static gating 备注:
  - 纯 AMV 特征无法为 P3 构建正向 gating（最优规则仅 +13K），原因在于 P3 6d 静态周期与混沌期时间窗口不匹配
  - 若继续 P3 gating，需要 AMV 之外的维度：板块宽度或个股相对 AMV 的表现
- B1 备注:
  - 原始 B1 三条件与 pullback 方向同源，但 executable-aware 初筛弱于当前 pure pullback sleeve
  - 暂不接 Rust，不进入 allocation/gating 候选；如继续，只做条件 ablation，确认现版复杂 B1 哪些过滤压制了 alpha
- Trend-only 备注:
  - Mac full grid 显示 `P1/K0.5/PB1/CP0/RV1` 等新候选 label 侧更强，但 full top 新候选 Rust 验证后仍未超过 P3/PB3
  - 逐日/逐票归因报告: `reports/amv_trend_vs_pb3_signal_trade_overlap.json`
  - 主要问题不是涨停/高开污染，而是 Python Top3 在真实 rolling 账户中大量变成已持仓不可重复买入；`trend P1/K1/PB1/CP0/RV1` 实买只有 `49.7%` 仍在 Python Top3，而 `PB3/CP1/RV0` 为 `70.0%`
  - 已新增 `bt-amv-topn` duplicate lot 诊断开关 `allow_duplicate_positions`，默认 `false`；诊断配置为 `config_6td_rolling21_refill_top10_duplicate_no_stop.toml`
  - duplicate 诊断报告: `reports/amv_duplicate_position_diagnostic_summary.json`
  - duplicate 后 `trend P1/K1/PB1/CP0/RV1` rolling refill 从 `+60.17%` 升至 `+106.50%`，PB3 从 `+99.62%` 小升至 `+102.93%`
  - 当前结论: duplicate 坐实 trend-only 被 no-repeat 语义明显压制，但它会提高单票集中度，暂作为诊断/可选进攻口径，不替代默认分散持仓口径；如继续 trend-only，应同时看 no-repeat 和 duplicate 两套真实回测
- 第三优先: `cash_ok` / 降仓标签
  - PB3 rolling 已有简化版 gating（UNION 规则，+52K），不再是纯 cash_ok 问题
  - P3 static gating 仍需突破：AMV 内部分析显示事后 late 亏 -221K 但纯 AMV 特征无法前向分离，需要新维度
  - 理论依据来自 AMV 受约束 Oracle 中的 cash 上限；但当前判断 gating 路径从 ML（attack_ok）→ AMV 内部阶段（age/maturity）→ 混沌期（neg_streak+amplitude）逐步收敛到更简单的规则形式
- 第四优先: executable-aware 评估规范继续固化
  - 后续所有因子/权重探索必须同时输出 executable 主指标、close-to-close 辅助指标和涨停/高开污染归因
  - refill Top10 需要明确区分“涨停补位”和“已有持仓重复代码补位”，不能默认视为质量提升
  - 新增 rolling cohort 多统计脚本: `scripts/amv_signal_cohort_stats.py`
  - 最新报告: `reports/amv_signal_cohort_stats_main_pullback_trend.json`
  - Canvas: `amv-signal-cohort-stats.canvas.tsx`
  - `refill_top10` event-time cohort: reference `+98.55%`, P3 `+102.93%`, PB3 `+186.48%`, trend label top `+229.26%`, trend Rust top `+215.39%`
  - 该报告用于信号质量诊断，不替代 Rust account NAV；trend-only 统计分布最强，但真实组合承接仍受 no-repeat、资金暴露和成本影响
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

### AMV Regime 内部阶段诊断

- 新增脚本: `scripts/amv_regime_phase_diagnostic.py`
- 产物: `reports/amv_regime_phase_diagnostic.json`
- 目标: 在 AMV 牛市内部识别可交易的 gating 条件，替代 `attack_ok` 的 ML 路径
- 双阶段体系:
  - (A) hindsight：事后标签 early/mid/late（基于 regime_progress），用于诊断改善空间
  - (B) forward：仅用 entry 当天可观测特征（duration、momentum、acceleration、dd_from_high、regime_maturity、neg_streak、amplitude），用于制定可执行规则
- 关键发现:
  - P3 static 在事后 late 阶段亏 `-221K`（76t, WR=39.5%），mid 阶段赚 `+715K`（114t, WR=64%），改善空间真实存在
  - 事后 late 亏损中 AMV 平均走了 30d、成熟度 68%、加速度 -12%、距高点回撤 -4.4%
  - **纯 AMV 层面特征无法为 P3 static 构建正向 gating 规则**：最佳规则（aged+非加速）仅净赚 `+13K`，且误杀 4 笔大赢家；其他 duration/momentum/acceleration/dd/maturity 组合均为净亏损
  - 根本原因: P3 6d 静态 cadence 天然频率低（274t），且 late 阶段的"末期暴力反弹"和"真的见顶"在 AMV 特征上过于相似
  - 但 AMV 在 bear trigger（-2.3% 单日跌幅）前存在可检测的"混沌期"：触发前 5-10 天 AMV 日均收益持续为负、振幅从 mid-bull 的 2.5% 升至 2.9%
- **PB3 rolling 接入 gating 成立**:
  - 规则: `(regime_duration 16-30d & slope_5d < 1.5%) OR (neg_streak >= 3 & amplitude > 2.5%)`
  - 跳过 305t（18.5%），跳过 PnL `-52K`（61% 亏损），净赚 `+52K`（+10.5%），**零误杀大赢家**
  - 2022 年省下 `+33K`（全年仅赚 `+41K`），2023 年省下 `+50K`（全年总亏 `-18K` 被掰回）
  - PB3 高频 rolling 与混沌期的 3-5 天时间窗口匹配，而 P3 6d 周期天然跳过
  - 当前判断: PB3 rolling 可以接入这个 UNION gating；P3 static 的 gating 需要引入 AMV 之外的维度（板块宽度或个股状态）
- regime_maturity（经验生存 CDF）:
  - `d10: 35%`, `d17: 49%`（中位）, `d20: 60%`, `d30: 81%`, `d50: 93%`
  - 比硬 duration 分桶更平滑，但单独使用仍无法解决分离问题

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
