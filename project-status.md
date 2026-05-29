# Project Status

> 本文件是“当前状态看板”，只保留当前结论、关键指标、归档路线和索引。实验流水账见 `progress.md`，长篇分析见 Canvas 与 `experiments/`。

---

## 当前决策摘要

- 项目产品化整理:
  - 第一阅读入口改为 `CURRENT_STATE.md`；统一 agent 指令入口为 `AGENTS.md`，`CLAUDE.md` 仅保留指向 `AGENTS.md` 的壳
  - 面向使用者的整理文档默认中文；`project-status.md` 保留详细看板，`progress.md` 保留倒序流水
  - 日常入口新增 `scripts/qlab.py`: `status`, `export`, `backtest`, `compare`, `attribution`
  - `qlab export ref/p3/context/pb3-gated/limit-weakgate` 已接入 `strategies/amv` native workflow，并复现 raw execution ground truth
  - Python 边界已合并进 `AGENTS.md`: `strategies/` 放策略定义与 registry，`utils/` 放通用底层工具，`scripts/` 收敛为命令入口
  - 脚本状态清单: `docs/script-inventory.md`; 第二阶段删除/迁移门禁: `docs/cleanup-plan.md`
  - 当前优先级: 暂停新增策略探索，先验证新入口稳定，再进入第二阶段迁移/删除
- 当前主策略底座: `manual_p2_k0p5_r0_6td`
  - raw execution 复跑后，reference `+145.10%` / MaxDD `18.97%`
  - `candidate_p3_k0p5_b0_c0_r0` raw execution 为 `+172.37%` / MaxDD `13.53%`，相对 reference 仍有 `+27.27pp` 收益优势且回撤更低
  - 当前仍以 reference 作为命名上的底座，但策略研究判断应切到 raw execution ground truth；旧 adjusted-execution 的 `+170.80%` / MaxDD `15.30%` 不再作为最终指标
  - P3 归因 Canvas: `reports/canvases/amv-p3-vs-ref-trade-attribution.canvas.tsx`
  - P3 的优势主要来自 `30` 笔边际替换交易，而非整体交易池重写；换票机制已解释为更偏向 P-block 的高位突破延续，但仍需后续样本确认稳定性
- Rust 真实成交价格口径:
  - 已完成代码切换: Python 主线 AMV signal export 补出 `open_raw/high_raw/low_raw/close_raw/pre_close_raw`；Rust `bt-amv-topn` 读取时优先 raw OHLC / raw pre-close，旧 artifact 缺 raw 字段时回退 adjusted
  - `report.json` 已写入 `execution_price_basis`: 新 artifact 为 `raw_ohlc_pre_close`，旧 artifact fallback 为 `adjusted_ohlc_fallback`
  - smoke 验证: `artifacts/amv_topn_raw_execution_smoke/20260529_142649` 短窗口导出 + 6td Rust 成功；旧 `limit_first_board_pullback` artifact fallback Rust 成功
  - raw ground truth 复跑报告: `reports/amv_raw_execution_ground_truth_summary.json`
  - 核心静态: Ref `+145.10%` / MaxDD `18.97%`; P3 `+172.37%` / MaxDD `13.53%`; context combo `+238.54%` / MaxDD `14.03%`
  - P3 raw vs adjusted 逐笔归因: `reports/amv_p3_raw_vs_adjusted_trade_attribution.json`
    - P3 adjusted vs raw 买入的是同一批 `274/274` 笔交易，entry_date/code、exit_date、exit_reason 全部一致；不是换票、涨停过滤或信号缺失导致收益下降
    - raw 总 PnL 比 adjusted 少 `146.6K`；其中 notional/path sizing effect `-75.0K`，return effect `-71.6K`
    - adjusted shares > raw shares 的交易 `271/274`，平均 shares ratio `1.187`，但 raw/adjusted cost ratio 中位数 `0.943`，说明前复权低价确实放大了股数，并通过手数取整 / 复利路径让 adjusted 口径平均投入资金更高
    - `263/274` 笔单笔净收益率完全一致；只有 `11` 笔收益率不同，但贡献了约一半差异，集中在除权 / 复权比例在持仓期变化的窗口
  - PB3 gated rolling: `+80.55%` / MaxDD `11.75%`，低于旧 adjusted standalone `+109.73%`，但回撤也从 `16.20%` 降到 `11.75%`
  - 涨停首板: base `+130.95%` / MaxDD `45.38%`; weakgate `+155.04%` / MaxDD `34.12%`; weaktop1 `+197.70%` / MaxDD `50.76%`; weaktier `+219.58%` / MaxDD `51.38%`; weakscorepen `+152.17%` / MaxDD `41.73%`
  - 当前判断: raw execution 没有推翻 P3/context 方向，但显著压低收益；context combo 仍是最强核心静态 challenger。涨停首板 base 被明显降级，weakgate 是防守最好但收益一般；weaktop1/weaktier 继续因回撤过大否决
- P3 行业顺风增强:
  - 离散 `10d/bottom40/0.02` bucket penalty 已被 cadence 检查降级，不能直接升级为默认 P3
  - 完整表达候选 `mix10/20 + linear penalty + rel20_under0 + p0.03` 通过本轮 focused grid 与 cadence：Rust total `+219.55%`, Sharpe `1.279`, MaxDD `14.07%`，相对 raw P3 `+17.86pp`
  - Python no-cost cadence `7/7` 个 offset 优于 raw，median delta `+20.94pp`，worst delta `+19.59pp`
  - 当前判断: 可列为 P3 主线 challenger，但不直接替换默认 P3；本阶段先收口，不立即启动历史行业分类数据工程
  - 最终验收项: 完整走完下一轮策略路线后，再做历史行业分类/行业映射版本复核与 forward 监控，重点排除静态东方财富行业映射的历史分类偏差
  - 报告: `reports/amv_p3_sector_tailwind_complete_grid.json`, `reports/amv_p3_sector_tailwind_complete_cadence_p0p03.json`
  - Canvas: `reports/canvases/amv-p3-sector-tailwind-complete.canvas.tsx`
- P3 中期结构 / 趋势质量增强:
  - 第二阶段主干已完成 diagnostic -> soft rerank -> Rust -> 参数邻域 -> sector 交互 -> cadence
  - 说明: 下文 artifact 名里的 `medium128` 指“128 日中期结构 + 128 日趋势质量”，不是单独一个模糊指标
  - 单独 128 日中期结构 / 趋势质量最佳为 `p3_medium128_quality_linear_t0p5_p0p03`: Rust static strict total `+264.90%`, MaxDD `14.05%`
  - 参数邻域确认 `p0.03` 是局部峰值: `p0.025` total `+251.32%`, `p0.035` total `+240.97%`, `p0.04` total `+225.43%`
  - 组合最佳为 `sector mix10/20 linear rel20_under0 p0.02 + 128 日中期结构 / 趋势质量 p0.03`: Rust total `+272.06%`, MaxDD `14.05%`
  - 相对 raw P3: total `+70.37pp`, MaxDD `+0.53pp`, win rate `+2.19pp`
  - no-cost cadence `7/7` 个 offset 优于 raw，median delta `+31.69pp`
  - 归因: exact overlap `226/274`; raw-only `48` 笔 `-83.8K`; combo-only `48` 笔 `+163.1K`; unique trade delta `+246.9K`
  - 相对单独 128 日中期结构 / 趋势质量: total `+7.16pp`, MaxDD 基本不变，exact overlap `272/274`
  - 风险: 2026 年度仍小幅弱于 raw `-0.57pp`，2026-01 trade delta `-110.4K`
  - annual restart 复核: context combo 改善 2026 median `-2.16% -> +1.85%` 和 positive offsets `2/7 -> 5/7`，但 worst offset `-8.55% -> -9.72%` 未修复
  - 第二阶段剩余细分收口:
    - 趋势质量细分已补齐: 128 日回撤深度、显式趋势线性度、`MA20 > MA60 > MA120` 均线排列稳定性
    - 流动性 / 成交额异动已补齐: 成交额 `1/20`, `5/20`, `20/60` 相对扩张、放量不涨、缩量回调、突破日量能确认、流动性枯竭后恢复
    - P3 诊断: 成交额扩张、突破放量确认、流动性恢复都是正向事件，不适合 hard skip；唯一正向 skip 是 128 日深回撤过滤，跳过 `18` 笔、trade-level delta `+28.7K`，样本小且 2025 反向 `-42.9K`
    - PB3 诊断: 低均线排列稳定性过滤跳过 `386` 笔、trade-level delta `+35.5K`，但 2025 反向 `-50.3K`，弱于既有 PB3 regime gating
    - 当前判断: 剩余细分完成 diagnostic 收口，但没有产生足够强、足够稳的新 signal export / Rust 候选；不叠加到当前最佳 P3 context combo
  - 当前判断: 第二阶段验收复核通过并完成剩余细分收口；最佳组合上下文 rerank 是当前最强 P3 challenger，但 annual restart 暴露 2026 最差路径风险，仍不直接替换默认 P3
  - 报告: `reports/amv_p3_context_combo_validation.json`, `reports/amv_p3_context_combo_attribution.json`, `reports/amv_p3_context_combo_cadence.json`, `reports/amv_p3_annual_restart_cadence_context_combo.json`
  - 剩余细分报告: `reports/amv_liquidity_trend_refinement_diagnostic.json`
  - Canvas: `reports/canvases/amv-medium-trend-quality-diagnostic.canvas.tsx`, `reports/canvases/amv-liquidity-trend-refinement-diagnostic.canvas.tsx`
- P3 annual restart cadence 风险复核:
  - 快速报告: `reports/amv_p3_annual_restart_cadence_quick.json`
  - 完整报告: `reports/amv_p3_annual_restart_cadence_context_combo.json`
  - raw P3 在 2021-2025 每年独立重启 offset 均为 `7/7` 正收益；2026 只有 `2/7`，median `-2.16%`
  - `medium128 p0.03` 将 2026 改善到 `5/7`，median `+1.85%`，但 2026 worst offset 仍为 `-9.72%`
  - `sector complete` 在 2026 为 `2/7`，median `-4.67%`，不能独立修复 2026 annual restart 风险
  - `context combo` 在 2026 为 `5/7`，median `+1.85%`，但 worst 仍 `-9.72%`
  - 当前判断: raw P3 未被 annual restart 直接推翻，但 2026 路径脆弱性是真风险；上下文增强改善中位路径，不修复最差路径，因此不能只看全样本 total
- Rust 真实成交价格口径待办:
  - 当前 `bt-amv-topn` 从信号 parquet 读取 `open_adj/high_adj/close_adj/pre_close_adj`，买入、卖出、手数、费用、涨跌停判断都基于前复权价
  - 正确目标口径: 因子、排序、趋势收益继续使用前复权价；真实成交、资金占用、手数取整、涨跌停 / 高开过滤应使用 raw OHLC 与 raw pre-close
  - 影响判断: 短持仓且不跨除权除息时收益率通常接近，但手数、资金占用、涨跌停识别和跨除权交易会有误差；这属于 Rust ground truth 的基础口径修正项
  - 待办: Python signal export 统一补出 `open_raw/high_raw/low_raw/close_raw/pre_close_raw`，Rust `PriceBar` 同时保留 raw/adj，成交与交易制度事件切 raw，因子与 score 保持 adj；修正后需复跑 Ref/P3、最佳 context combo、PB3 gated 作为新口径基线
- 下一轮上下文因子路线:
  - 当前约定: 不马上补组合层 allocation，先继续完成 A 股上下文因子路线
  - 第一阶段: 板块顺风 + 市场赚钱效应，直接对应 P3 的假突破与 2026 弱段
  - 第二阶段: 64/128 日中期结构 + 趋势质量，补当前 20 日形态体系缺少的中期结构和“新高质量”
  - 第三阶段: 涨停生态 / 强势股事件 sleeve，可能形成低相关新策略族，不一定混进 P3/PB3
  - 暂缓: 基本面/另类数据因子放中期，因数据治理成本更高
  - 核心判断: 量价仍有空间，但不是继续堆 RSI/RSRS/MACD；优先补“板块、情绪、趋势质量、涨停生态”四类 A 股上下文因子
- 第三阶段涨停生态初筛:
  - 新增 `scripts/amv_limit_ecology_diagnostic.py`，使用当前可用的日线 raw OHLCV 构造涨停生态近似特征，并按 `signal_date + code` 拼到 P3/PB3 已成交交易
  - 当前可做: 近 N 日是否收盘涨停、距上次涨停天数、连板高度、首板上下文、炸板近似、近 N 日一字板近似、首板后回踩、涨停后收复、断板后承接
  - 当前不能做: 精确封板时间、封单金额、开板次数、竞价强度、Level2 委托强度；`stock_daily` 也没有官方 `pre_close`，当前涨跌停判断用“前一交易日 raw close”近似，除权日仍可能误判
  - P3 诊断: 近 20 日有涨停的 `24` 笔贡献 `+649.7K`，avg `+8.19%`, win rate `62.5%`; 其中昨日涨停 `14` 笔贡献 `+598.1K`, 10 日内再板/反包 `16` 笔贡献 `+644.9K`
  - PB3 诊断: 近 20 日有涨停的 `229` 笔贡献 `+164.5K`, avg `+1.98%`，有解释力但弱于 P3；近 5 日炸板样本 `22` 笔为 `-2.3K`
  - 首轮 event sleeve Rust:
    - 新增 `scripts/amv_limit_ecology_signal_export.py`，导出 `limit_reboard_reclaim`, `limit_recent_lu_ranked`, `limit_first_board_pullback`
    - `limit_reboard_reclaim`: Rust static strict total `-21.54%`, MaxDD `68.57%`; T+1 open 涨停阻塞 `144` 条 / `118` 天，强 trade-level 事件未兑现
    - `limit_recent_lu_ranked`: total `+46.25%`, MaxDD `59.87%`, 2026 `-7.31%`，正收益但回撤过深
    - `limit_first_board_pullback`: total `+85.32%`, MaxDD `56.50%`, 2026 `+20.16%`，当前最好但仍不够 sleeve-ready
  - first-board focused scan:
    - 新增 `config_3td_static_strict_top3_no_stop.toml` / `config_5td_static_strict_top3_no_stop.toml`
    - `limit_first_board_pullback 5td`: total `+183.57%`, MaxDD `45.35%`, 2021-2026 年年为正，收益最强但回撤仍过深
    - `limit_first_board_pullback_quality 6td`: total `+53.36%`, MaxDD `21.71%`, 无 T+1 涨停阻塞，是最防守版本但收益偏低且 2023 `-6.02%`
    - `lowvol 5td`: total `+16.94%`, MaxDD `21.74%`，回撤降低但 alpha 基本被过滤掉
  - first-board drawdown attribution:
    - 新增 `scripts/amv_limit_ecology_drawdown_attribution.py` 和报告 `reports/amv_limit_first_board_pullback_drawdown_attribution.json`
    - `base 5td` 最大回撤为 `2023-08-03 -> 2024-02-05`，到 `2025-02-19` 才收复；回撤区间已实现交易 `43` 笔，PnL `-431.9K`，win rate `25.6%`
    - 大回撤不是 T+1 涨停阻塞，也不是单笔黑天鹅，而是弱段内连续买入高 ATR 首板后回踩: 回撤区间平均 ATR rank `0.870`，`74.4%` 交易 ATR rank `>0.8`；最差 `12` 笔里 `91.7%` ATR rank `>0.8`
    - `quality 6td` 将 MaxDD 压到 `21.71%`，对应平均 ATR rank `0.537` 且无 ATR rank `>0.8` 样本，但只保留 `107` 笔、收益偏低，说明“硬 lowvol/quality”会过度削 alpha
  - medium128 / 风险 rerank 复核:
    - 新增 `scripts/amv_limit_first_board_medium128_diagnostic.py` 和 `scripts/amv_limit_first_board_variant_summary.py`
    - trade-level: `medium128 weak` 在全样本中反而贡献 `+446.3K`，直接过滤会误杀；在最大回撤窗口内有解释力但不是可泛化 hard gate
    - Rust `5td`: `medium128pen` total `+151.50%`, MaxDD `46.99%`，低于 `base 5td` 且不降回撤，否决为单独优化项
    - Rust `5td`: `atrpen2` total `+218.89%`, MaxDD `45.55%`，收益更高但回撤未改善；说明 ATR 静态惩罚能改善排序 alpha，但不能解决弱段连续回撤
  - 弱窗口诊断:
    - 新增 `scripts/amv_limit_first_board_weak_window_diagnostic.py` 和报告 `reports/amv_limit_first_board_weak_window_diagnostic.json`
    - 最大回撤期在开仓前有可观测画像: AMV 5 日斜率 `3.75% -> 0.24%`，AMV 距高点回撤 `-2.30% -> -4.29%`，候选数 `15.8 -> 6.8`，Top3 均分 `8.46 -> 6.03`，Top3 ATR rank `0.781 -> 0.872`，Top3 老涨停占比 `39.4% -> 62.8%`，Top3 reclaim 占比 `43.3% -> 25.6%`
    - 单项 AMV phase、昨日涨停溢价、炸板率、普通强势生态 hard gate 都不成立，trade-level 会误杀大量盈利；更有价值的是候选池健康度复合条件
    - 诊断候选: `limit_up_count_rank_pct < 0.45 & top3_score <= 6.5 & top3_atr > 0.80` 标记 `32` 笔，flagged PnL `-162.4K`，命中回撤窗口 `21` 笔；`candidate_count <= 8 & top3_score <= 6.5 & top3_atr >= 0.85` 标记 `35` 笔，flagged PnL `-132.1K`，命中回撤窗口 `21` 笔
  - weak-window Rust 对照:
    - `scripts/amv_limit_ecology_signal_export.py` 新增 `limit_first_board_pullback_weakgate`, `weaktop1`, `weaktier`, `weakscorepen`
    - `weakgate 5td`: weak_score `>=3.0` 的信号日完全不开仓，signal rows `2,282 -> 1,550`，trades `306 -> 256`，total `+158.02%`，MaxDD `34.14%`；防守有效但收益少 `-25.55pp`
    - `weaktop1 5td`: 弱日只买 rank1，total `+202.64%`，但 MaxDD `50.54%`；`weaktier 5td`: total `+226.38%`，MaxDD `51.30%`；两者说明“弱日保留 rank1”不稳定，暂时否决
    - `weakscorepen 5td`: 弱日惩罚高 ATR / 老涨停 / 缺少 reclaim，total `+157.07%`，MaxDD `41.70%`；温和降回撤但收益牺牲也较大，不足以替代 base
  - 当前判断: 第三阶段方向有 alpha 线索，但还没有 allocation-ready sleeve；不要进入 rolling/refill。`medium128` 对该 sleeve 暂无独立帮助；`top1/downshift` 第一版被否决，`weakgate` 是当前防守上限但收益代价明显，后续若继续应优化弱窗口定义或尝试连续 downsize / position sizing，而不是简单 Top1。raw execution 修正后也需要复核
  - 报告: `reports/amv_limit_ecology_diagnostic.json`
  - Rust 报告: `reports/amv_limit_ecology_event_sleeve_rust_summary.json`
  - Focused scan: `reports/amv_limit_first_board_pullback_hold_risk_scan.json`
  - Drawdown attribution: `reports/amv_limit_first_board_pullback_drawdown_attribution.json`
  - Risk rerank scan: `reports/amv_limit_first_board_medium128_diagnostic.json`, `reports/amv_limit_first_board_risk_variant_scan.json`
  - Weak window diagnostic: `reports/amv_limit_first_board_weak_window_diagnostic.json`
  - Canvas: `reports/canvases/amv-limit-ecology-diagnostic.canvas.tsx`
- P3 市场情绪 / 赚钱效应初筛:
  - 报告: `reports/amv_market_sentiment_diagnostic.json`
  - Canvas: `reports/canvases/amv-market-sentiment-diagnostic.canvas.tsx`
  - P3 简单冷市场过滤不成立：低涨停数、低 20 日新高占比、低市场上涨占比都会误杀大量大赢家
  - 过热/拥挤过滤 `hot_yday_limit_up_premium + high_new_high_20` 在 trade-level 看似有效: 跳过 `54` 笔、跳过 PnL `-132.5K`、delta `+132.5K`; 2026 delta `+62.9K`
  - Rust static strict 硬 gate 已否决: raw P3 `+201.69%` / MaxDD `13.52%` -> gated `+182.02%` / MaxDD `13.53%`
  - 当前判断: 市场情绪不能作为 P3 独立 hard gate；若继续，只能尝试 soft penalty / rerank 或与 sector-tailwind 交互
  - PB3 对低涨停数有小幅正 delta `+24.0K`，但弱于既有 regime gating，暂不优先升级
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
- PB3 rolling regime gating:
  - `scripts/amv_static_sleeve_signal_export.py` 已支持 `--pb3-regime-gate aged_non_accel_or_chaos`
  - 规则按 `signal_date` 收盘计算，避免 T+1 open 买入时使用 entry day 收盘后信息
  - Rust 报告: `reports/amv_pb3_regime_gating_rust_summary.json`
  - raw PB3 rolling `+99.62%` / MaxDD `20.70%` -> gated `+109.73%` / MaxDD `16.20%`
  - 过滤 `1380/8140` 条 signal rows、`138/814` 个 signal days；交易数 `1650 -> 1393`
  - 稳健性报告: `reports/amv_pb3_gating_robustness.json`
  - trade-level 当前规则跳过 `258/1650` 笔、贡献 `+23.2K`，无 `>20K` 大赢家误杀；但分年主要靠 `2022/2023`，`2025/2026` 为负
  - walk-forward 合计仍为正: 历史选规则测试年 `+37.5K`，固定当前规则 `+29.5K`
  - 当前判断: 这是目前最值得继续验证的 PB3 风控/降开仓规则；方向有效但不够年年稳定，暂不切到更激进参数
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

- 下一轮上下文因子路线（完整决策背景）:
  - 总判断:
    - 当前不应该先补组合层 allocation；组合层等上下文因子路线走完一轮后再统一评估
    - 量价仍有空间，但不是继续堆 RSI / RSRS / MACD 这类通用指标
    - 最该补的是“板块、情绪、趋势质量、涨停生态”四类 A 股上下文因子
    - 这条路线的目标不是立刻替换 `manual_p2_k0p5_r0_6td` 或 `candidate_p3_k0p5_b0_c0_r0`，而是解释并减少 P3 的假突破、2026 弱段和 AMV bull 内部赚钱效应不足的问题
  - 第一阶段: 板块/行业顺风因子 + 市场赚钱效应 / 情绪温度因子
    - 目的:
      - 直接对应 P3 当前最大痛点: 假突破和 2026 弱段
      - P3 的 2026 问题不是个股形态完全错，而是选到了水泥、地产、航天这类当时不在主线里的假突破
      - 纯 AMV bull 还不够，需要知道当下是不是“有赚钱效应的牛”，还是“指数没坏但短线很难做”
    - 板块/行业顺风因子:
      - 这是最建议先做的方向
      - 可做因子: 行业 5/10/20 日收益排名、行业内上涨家数占比、行业内新高占比、行业成交额扩张、个股相对行业强弱
      - 用法: 更适合做 gate 或加权，而不是单独排序
      - 当前状态: 已完成第一轮 P3 trade diagnostic -> Rust rerank -> cadence -> complete expression
      - 当前候选: `mix10/20 + linear + rel20_under0 + p0.03` 可列为 P3 主线 challenger，但不直接替换默认 P3
      - 未完成验收: 历史行业分类 / 行业映射版本复核、forward 样本、以及最终组合层影响
    - 市场赚钱效应 / 情绪温度因子:
      - 这是下一步优先做的方向
      - 可做因子: 全市场涨停数、跌停数、炸板率、连板高度、昨日涨停次日溢价、20 日新高家数、强势股回撤幅度
      - 初筛结论: P3 不是简单怕“冷市场”；低涨停数、低新高占比、低上涨占比反而承载很多大赢家
      - Rust 结论: `hot_yday_limit_up_premium + high_new_high_20` hard gate 降低收益且不降回撤，不能升级
      - 后续只保留为 soft/rerank 或与行业因子交互的线索，不再单独做 hard gate
  - 第二阶段: 64/128 日中期结构 + 趋势质量 + 流动性冲击 / 成交额异动
    - 目的:
      - 补博主早期 128 日日 K 窗口里可能有、但当前体系还没充分表达的东西
      - 当前 P3 更多是 20 日附近形态，偏“贴近新高”，但还缺“新高质量”和中期结构确认
      - 这类因子不一定直接提高 Top3 收益，更可能帮助区分“真趋势票”和“短期假突破”
    - 中期结构因子，特别是 64/128 日窗口:
      - 可做因子: 128 日收益分位、128 日高低位、趋势效率、回撤后修复比例、波动收缩程度、长期均线斜率
      - 用法: 作为 P3 假突破诊断和候选加权项，优先看是否能提升 2026 与弱行情窗口
    - 趋势质量因子:
      - 不是看涨了多少，而是看涨得是否顺
      - A 股里很多票短期涨幅强，但路径很脏，容易回撤
      - 可做因子: 收益 / 波动、上涨天数占比、K 线实体效率、回撤深度、趋势线性度、均线排列稳定性
      - 用法: 适合补 P3，因为 P3 当前偏“贴近新高”，但还缺“新高质量”
    - 当前状态:
      - 已完成第二阶段 P3 trade diagnostic -> signal-level soft penalty -> Rust static strict -> 参数邻域 -> sector 交互 -> no-cost cadence
      - 新增脚本: `scripts/amv_medium_trend_quality_diagnostic.py`, `scripts/amv_medium_trend_quality_signal_export.py`, `scripts/amv_context_combo_signal_export.py`
      - 报告: `reports/amv_medium_trend_quality_diagnostic.json`, `reports/amv_p3_medium_trend_quality_rust_summary.json`, `reports/amv_p3_context_combo_validation.json`, `reports/amv_p3_context_combo_attribution.json`, `reports/amv_p3_context_combo_cadence.json`
      - Canvas: `reports/canvases/amv-medium-trend-quality-diagnostic.canvas.tsx`
      - 诊断结论: P3 的高 64 日结构、高 128 日趋势质量、高 128 日位置分桶明显更强；但单独 hard skip 会误杀 2026 赢家
      - 单独 128 日中期结构 / 趋势质量候选: `p3_medium128_quality_linear_t0p5_p0p03`
      - 组合上下文候选: `sector mix10/20 linear rel20_under0 p0.02 + 128 日中期结构 / 趋势质量 p0.03`
      - Rust static strict: raw P3 `+201.69%` / MaxDD `13.52%` -> combo `+272.06%` / MaxDD `14.05%`
      - 归因: exact overlap `226/274`; raw-only `48` 笔 `-83.8K`; combo-only `48` 笔 `+163.1K`; unique trade delta `+246.9K`
      - cadence: no-cost 7 offset `7/7` 优于 raw，median delta `+31.69pp`
      - 风险: 2026 年度仍小幅弱于 raw `-0.57pp`，且 2026-01 trade delta `-110.4K`
      - 当前判断: 第二阶段验收复核通过，可以阶段性收口；仍不直接替换默认 P3，保留 forward 监控与最终路线对比
    - 流动性冲击 / 成交额异动因子:
      - 当前已有一些成交额 / 波动相关东西，但还可以更系统
      - 可做因子: 成交额相对 20/60 日分位、换手率突增、放量不涨、缩量回调、突破日量能确认、流动性枯竭后恢复
      - 用法: 对 PB3 可能尤其有用，因为 pullback 成功往往需要“缩量回调 + 再放量”；对 P3 则更偏突破质量确认
  - 第三阶段: 涨停生态 / 强势股事件因子
    - 目的:
      - 这是 A 股特色，和普通量价因子不完全一样
      - 它可能形成一个真正低相关的新策略族，不一定和 P3/PB3 混在一起
    - 可做因子: 近 N 日是否涨停、涨停后第几天、炸板后修复、首板后回踩、反包、连板断板后承接
    - 用法: 优先当作独立 sleeve 的候选方向，而不是先混入 P3/PB3 排序
  - 中期后置: 基本面 / 另类数据因子
    - 这条不作为当前第一优先，因为数据治理成本更高
    - 可做因子: 盈利质量、营收 / 利润增速、ROE、估值分位、机构持仓变化、股东户数、年报长度、CEO 年龄等另类数据
    - 用法: 可能降低纯量价失效风险，但应在量价上下文路线更稳定后再接入
- 第一优先: `P3/K0.5/R0` 主基线替换验证
  - Rust 静态 strict 已从 reference `+170.80%` / MaxDD `15.30%` 提升到 `+201.69%` / MaxDD `13.52%`
  - 年度/交易归因显示优势主要来自 `30` 笔边际换票；2026-01 与 2023-04 的换票机制已解释，下一步看更多样本和 forward 监控后再正式切换
  - P3 static cadence 敏感性报告: `reports/amv_p3_static_cadence_sensitivity.json`
  - 7 个起始 offset 的 no-cost Python-like static 路径最差 `+260.74%`、中位 `+285.60%`、最好 `+297.79%`；粗扣 `0.35%` 单轮往返成本后最差约 `+162.19%`、中位 `+181.25%`，说明 static 不是单一起点侥幸，但最终仍以 Rust net 为准
  - P3 early-stop Rust 报告: `reports/amv_p3_early_stop_rust_summary.json`
  - `d2 < -3% 且 d1 为负` 被真实账户口径否决: net `+201.69% -> +134.75%`，MaxDD 仅 `13.52% -> 12.78%`
  - 当前判断: P3 不适合简单价格止损；若做风控，应结合板块顺风/个股相对 AMV 弱势/结构位，而不是单独用 d2 浮亏
- 当前组合候选: `P3 static + PB3 gated rolling`
  - 组合报告: `reports/amv_p3_pb3_gated_allocation.json`
  - Canvas: `reports/canvases/amv-p3-pb3-gated-allocation.canvas.tsx`
  - `P3 80% / PB3 gated 20%`: total `+185.25%`, MaxDD `11.64%`, 2026 `+2.20%`
  - `P3 70% / PB3 gated 30%`: total `+176.50%`, MaxDD `10.87%`, 2026 `+3.70%`
  - 当前判断: `80/20 gated` 是最自然起点；`70/30 gated` 更防守但收益牺牲更明显
- P3 sector tailwind 初筛:
  - 新增脚本: `scripts/amv_sector_tailwind_diagnostic.py`
  - 报告: `reports/amv_sector_tailwind_diagnostic.json`
  - Canvas: `reports/canvases/amv-sector-tailwind-diagnostic.canvas.tsx`
  - 数据口径: 静态东方财富行业映射 `5,549` 只股票、`86` 个行业 + QMT 日线；行业特征在 `signal_date` 计算
  - P3 行业 10 日收益 rank bottom 40%: `67` 笔合计 `-43.9K`；trade-level skip delta `+43.9K`
  - 但严格只保留 `tailwind_ok` 会误杀 `169` 笔合计 `+369.2K` 的交易；PB3 bottom 40% 仍合计 `+257.0K`，不适合套同一过滤
  - P3 rerank 已接 Rust: `scripts/amv_sector_tailwind_signal_export.py`
  - rerank 报告: `reports/amv_p3_sector_tailwind_rerank_summary.json`
  - rerank 归因: `reports/amv_p3_sector_tailwind_rerank_attribution.json`
  - rerank Canvas: `reports/canvases/amv-p3-sector-tailwind-rerank.canvas.tsx`
  - 稳健性报告: `reports/amv_p3_sector_tailwind_robustness.json`
  - 稳健性 Canvas: `reports/canvases/amv-p3-sector-tailwind-robustness.canvas.tsx`
  - 最佳 `penalty=0.02`: total `+242.10%`, CAGR `25.91%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`
  - 相对 raw P3: total `+40.41pp`, MaxDD `-0.08pp`；exact overlap `216/274`，unique trade delta `+144.4K`
  - focused robustness: `10d/bottom40` 下 `penalty=0.018~0.025` 均明显优于 raw；`bottom50` 过度惩罚导致 total `+138.70%`, MaxDD `20.65%`
  - rank window: `5d` total `+222.04%` / MaxDD `16.62%`; `10d` total `+242.10%` / MaxDD `13.44%`; `20d` total `+264.26%` / MaxDD `16.09%`
  - cadence 报告: `reports/amv_p3_sector_tailwind_cadence.json`, `reports/amv_p3_sector_tailwind_cadence_w20.json`
  - cadence Canvas: `reports/canvases/amv-p3-sector-tailwind-cadence.canvas.tsx`
  - `10d/bottom40/0.02` no-cost 7 offset: 仅 `2/7` 个 offset 优于 raw，median delta `-10.56pp`，worst delta `-24.16pp`
  - `20d/bottom40/0.02` no-cost 7 offset: `5/7` 个 offset 优于 raw，median delta `+9.10pp`，但 worst delta `-22.25pp` 且最差 DD 到 `15.42%`
  - 当前判断: sector tailwind 离散扣分不能直接升级为默认 P3；`10d/bottom40/0.02` 降级为“默认起点表现好但 cadence 不稳”，`20d` 保留为 challenger；下一步如继续，应试连续型行业 rank penalty 或 10d/20d 混合项
- 第二优先: rolling pure pullback sleeve 验证 + gating
  - 已选 `PB3/CP1/RV0 rolling21 refill` 作为代表袖子，rolling21 refill Top10 为 `+99.62%`, MaxDD `20.70%`
  - PB3 rolling gating Rust 已验证: raw `+99.62%` / MaxDD `20.70%` -> gated `+109.73%` / MaxDD `16.20%`
  - 稳健性快筛显示方向有效但不年年稳定，暂不切到更激进参数；`PB2/CP0.5/RV0` 仅作为 challenger 监控
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
- RSRS 备注:
  - 新增 `scripts/amv_executable_rsrs_scan.py`，RSRS beta/z/R2/right 因子构造使用 Polars rolling covariance/variance/correlation
  - 报告: `reports/amv_rsrs_executable_scan_summary.json`
  - 传统 `RSRS beta high` 在 AMV bull pool 中很弱: refill exec NAV `-4.89%`
  - 最好单因子为 `rsrs_z_18_120_low`: refill exec NAV `+99.88%`, MaxDD `31.46%`, close 涨停污染 `0%`
  - 当前判断: RSRS 更像 pullback/reversion 辅助线索，不是独立主线；后续只考虑把 `rsrs_z_18_120_low` 放入 PB/CP/RV 辅助组合或用于 P3 假突破诊断
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
