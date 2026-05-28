# Progress

## 如何阅读

- 本文件是实验流水账，按日期倒序记录探索过程、修正、失败路线和阶段性结论。
- 当前状态、主线策略、归档路线和下一步优先级以 `project-status.md` 为准。
- 已归档探索的集中索引见 `experiments/archive-index.md`。
- 历史小节中的“下一步”只代表当时判断；如果与顶部最新收口或 `project-status.md` 冲突，以最新收口为准。

## 最新收口索引

- AMV 下一轮上下文因子路线: 已明确不先补组合层 allocation，而是继续按“板块/行业顺风 + 市场赚钱效应 -> 64/128 日中期结构 + 趋势质量 -> 涨停生态 sleeve”的顺序推进。核心判断是量价还有空间，但不再堆 RSI/RSRS/MACD 这类通用指标；优先补“板块、情绪、趋势质量、涨停生态”四类 A 股上下文因子。
- AMV 128 日中期结构 / 趋势质量第二阶段验收: 新增 `scripts/amv_medium_trend_quality_diagnostic.py`、`scripts/amv_medium_trend_quality_signal_export.py`、`scripts/amv_context_combo_signal_export.py`。说明: artifact 名里的 `medium128` 指“128 日中期结构 + 128 日趋势质量”。参数邻域确认 `p0.03` 是局部峰值: `p0.025` total `+251.32%`, `p0.03` `+264.90%`, `p0.035` `+240.97%`, `p0.04` `+225.43%`。与 sector-tailwind 交互后，最佳组合为 `sector mix10/20 linear rel20_under0 p0.02 + 128 日中期结构 / 趋势质量 p0.03`: Rust total `+272.06%`, MaxDD `14.05%`, 相对 raw P3 `+70.37pp`；no-cost cadence `7/7` 个 offset 优于 raw，median delta `+31.69pp`。当前判断: 第二阶段可以阶段性收口，保留 2026-01 被牺牲作为 forward/final route 风险。
- AMV market sentiment 初筛: 新增 `scripts/amv_market_sentiment_diagnostic.py`，在 `signal_date` 构造全市场涨停/跌停、炸板、20 日新高、昨日涨停次日溢价、强势股回撤等情绪特征。P3 的简单冷市场过滤不成立，低涨停/低新高/低上涨占比反而误杀大赢家；“昨日涨停溢价过热 + 20 日新高家数偏高”拥挤过滤在 trade-level 看似有效，跳过 `54` 笔、delta `+132.5K`，2026 delta `+62.9K`，但严格 date gate 接 Rust 后被否决：raw P3 `+201.69%` / MaxDD `13.52%` -> gated `+182.02%` / MaxDD `13.53%`。当前判断: 市场情绪单独硬 gate 不成立，只能作为后续 soft/rerank 或与行业因子交互的候选线索。
- AMV sector tailwind 初筛: 新增 `scripts/amv_sector_tailwind_diagnostic.py`，使用静态东方财富行业映射 + QMT 日线在 `signal_date` 构造行业顺风特征。P3 的行业 10 日收益排名底部 40% 交易为 `67` 笔、合计 `-43.9K`，作为弱行业过滤有价值；但严格 `tailwind_ok` 会误杀大量赢家，PB3 不适合直接套该过滤。
- AMV P3 sector tailwind rerank: 新增 `scripts/amv_sector_tailwind_signal_export.py`，对 P3 候选中行业 10 日收益 rank bottom 40% 做 soft penalty 后重排。Rust static strict 显示 `penalty=0.02` 最佳: total `+242.10%`, CAGR `25.91%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`；相对 raw P3 total `+40.41pp`，但需要 cadence/行业映射/窗口稳健性复核。
- AMV P3 sector tailwind robustness: focused grid 显示方向不是单点好运气。`10d/bottom40` 下 `penalty=0.018~0.025` 均明显优于 raw P3；`bottom50` 过度惩罚导致 total `+138.70%`、MaxDD `20.65%`；`20d` 窗口 total 更高 `+264.26%` 但 MaxDD 扩到 `16.09%`。当前最均衡仍是 `10d / bottom40 / penalty=0.02`。
- AMV P3 sector tailwind cadence: no-cost 7 offset 检查推翻了直接升级 `10d/bottom40/0.02` 的想法，只有 `2/7` 个 offset 优于 raw，median delta `-10.56pp`。`20d/bottom40/0.02` 相对更稳，`5/7` 个 offset 优于 raw，median delta `+9.10pp`，但最差 offset `-22.25pp` 且 drawdown 可到 `15.42%`，暂只能作为 challenger。
- AMV P3 sector tailwind complete expression: 将离散 bucket 扣分改为连续型行业 rank penalty，并加入 `10d/20d` 混合与个股相对行业弱势确认。推荐候选 `mix10/20 + linear + rel20_under0 + p0.03`: Rust total `+219.55%`, Sharpe `1.279`, MaxDD `14.07%`，相对 raw P3 `+17.86pp`；Python no-cost cadence `7/7` 个 offset 优于 raw，median delta `+20.94pp`。行业因子可升为 P3 主线 challenger；当前阶段暂不启动历史行业分类数据工程，完整走完下一轮策略路线后再作为最终验收复核。
- AMV P3 annual restart cadence 快速诊断: 新增 `scripts/amv_annual_restart_cadence.py` 与 `reports/amv_p3_annual_restart_cadence_quick.json`。raw P3 在 2021-2025 每年独立重启 offset 均为 `7/7` 正收益，但 2026 只有 `2/7`、median `-2.16%`；`medium128 p0.03` 将 2026 改善到 `5/7`、median `+1.85%`，但最差 offset 仍为 `-9.72%`。明确待办: 在有完整行业映射/信号 artifact 的设备上补跑 `sector complete` 与 `context combo` annual restart，尤其检查 2026 最差 offset。
- AMV P/K/M Rust 验证: 标签侧动量增强未兑现为更好主基线；可解释为 2025 补充袖子，但未解决 2026。
- AMV 年度权重网格: 2025/2026 弱势更像旧 `P2/K0.5/R0` 缺少动量项，但该结论已被 Rust 验证降级为标签侧线索。
- AMV 手数修正后主基线: `manual_p2_k0p5_r0_6td` 更新为净收益 `+170.80%`, MaxDD `15.30%`。
- B3 TDX: 已归档为事件型补充候选，不作为当前主线继续深挖。
- Direct LTR Top3: 因 executable label 与真实执行口径错配，暂不接交易。
- `attack_ok` 第一版: 当前状态特征未证明可稳定学习进攻切换，需收紧标签后再回看。
- AMV 受约束 Oracle: 仍是当前 sleeve switching / gating 的理论锚点。
- `manual_p2_k0p5_r0_6td`: 仍是当前 AMV bull pool 主策略底座。
- AMV executable/pullback Rust 验证: `P3/K0.5/R0` 静态 strict 已跑到 `+201.69%` / MaxDD `13.52%`，成为替换主基线候选；pure pullback 在 rolling21 口径明显成立但回撤仍约 `20%+`。
- AMV P3 vs Ref 交易归因: P3 的 `+30.89pp` 优势主要来自 `30` 笔边际替换交易，尤其 `2026-01` 与 `2023-04`；机制上是提高 P-block 权重，把 Top3 边缘从“K 线更好但离高点略远”推向“更贴近 20 日高点/新高但 K 线项较弱”的突破延续票。
- 项目级 Skill: `.agents/skills/amv-trade-attribution/SKILL.md` 已沉淀 AMV 回测对比、交易归因、换票解释、成本损耗和互补性分析的标准流程。
- 通用回测归因脚本: `scripts/backtest_trade_attribution.py` 已支持两个 `bt-amv-topn` artifact 的收益、回撤、成本、年度/月度、trade overlap、unique winners/losers 和日收益相关性对比。
- AMV rolling pullback 代表: 暂选 `PB3/CP1/RV0 rolling21 refill` 作为后续 allocation/gating 的唯一 pullback 代表；`PB2/CP0.5/RV0` 保留为 forward challenger，不与 PB3 同时堆叠。
- PB3 rolling gating 稳健性: `reports/amv_pb3_gating_robustness.json` 显示，当前 `(aged+非加速) OR (neg>=3 & amp>2.5)` 在 trade-level 跳过 `258/1650` 笔、贡献 `+23.2K`，且无大赢家误杀；但分年主要靠 `2022/2023`，`2025/2026` 为负贡献。walk-forward 合计仍为正，说明方向可用但不宜激进加码。
- P3 early stop Rust 验证: `reports/amv_p3_early_stop_rust_summary.json` 显示，`d2 < -3% 且 d1 为负` 规则被真实账户口径否决；净收益 `+201.69% -> +134.75%`，MaxDD 仅 `13.52% -> 12.78%`，27 笔 early-stop 中 12 笔相对原始持有为负贡献，合计少赚约 `149K`。
- RSRS executable 初筛: 新增 `scripts/amv_executable_rsrs_scan.py`，RSRS 因子构造全程使用 Polars rolling covariance/variance/correlation；`reports/amv_rsrs_executable_scan_summary.json` 显示传统 `RSRS beta high` 在 AMV bull pool 中很弱（refill exec NAV `-4.89%`），低/标准化 RSRS 更强，最好 `rsrs_z_18_120_low` refill exec NAV `+99.88%`、MaxDD `31.46%`、close 涨停污染 `0%`。当前判断: RSRS 可作为 pullback/reversion 辅助特征，不宜作为独立主线或突破增强。
- P3 + PB3 gated allocation: 新增 `scripts/amv_allocation_diagnostic.py` 和 Canvas `reports/canvases/amv-p3-pb3-gated-allocation.canvas.tsx`。daily rebalance 组合显示，`P3 80% / PB3 gated 20%` total `+185.25%`, MaxDD `11.64%`, 2026 `+2.20%`；`70/30` total `+176.50%`, MaxDD `10.87%`, 2026 `+3.70%`。当前判断: 80/20 gated 是自然起点，70/30 更防守。
- AMV rolling cohort 多统计诊断: 新增 `scripts/amv_signal_cohort_stats.py`，报告 `reports/amv_signal_cohort_stats_main_pullback_trend.json`，Canvas `amv-signal-cohort-stats.canvas.tsx`。event-time cohort 结果显示 trend-only top 信号质量最高、PB3 次之、P3/reference 更低；但 trend-only 到 Rust account NAV 仍受 no-repeat、资金暴露和成本压制。
- P3 static cadence 敏感性: `reports/amv_p3_static_cadence_sensitivity.json` 显示，P3 static 不是单一起点侥幸；7 个起始 offset 的 no-cost Python-like static 路径最差 `+260.74%`、中位 `+285.60%`、最好 `+297.79%`，粗扣 `0.35%` 单轮往返成本后最差约 `+162.19%`、中位 `+181.25%`、最好 `+190.14%`。
- 原始 B1 executable-aware lab: 原始三条件在 AMV bull + liquidity 下不是废信号，全部候选平均 exec NAV `+53.28%`；但 Top3 需要 pullback 排序增强，`B1 base + PB2/CP0.5` 6td exec NAV `+89.21%`，仍弱于独立 pullback sleeve，暂归类为 pullback 变体线索而非新主线。去掉 `J <= 13` 后，趋势池 `close > YL / WL > YL` 明显扩张，`P3/K0.5` refill 诊断升至 exec NAV `+118.74%` / MaxDD `7.27%`，提示 `J <= 13` 会压制突破排序，但该结果仍需 Rust 真实回测验证。
- AMV trend-only 网格扫描: 新增 `scripts/amv_executable_trend_filter_grid.py`，focused grid 已跑通 `301` 个 ranker；修正导出脚本的 combo rank 母体后，最强候选 `trend P1/K0/PB1/CP0/RV0.5` static refill 提升到 `+112.54%`, rolling refill `+41.47%`。rank 母体差异解释了部分 Python -> Rust 损耗，但仍低于 Python label `+213.74%` 和当前 P3/PB3 主候选，暂不进入主线候选。
- Trend-only Python label -> Rust rolling 损耗归因: `reports/amv_trend_vs_pb3_signal_trade_overlap.json` 显示，`trend P1/K1/PB1/CP0/RV1` 真实 rolling refill 只有 `49.7%` 买入仍在 Python Top3，`824` 个 Python Top3 未买原因是已持有；`PB3/CP1/RV0` 对照有 `70.0%` 买入仍在 Top3。因此 trend-only 的主要损耗不是涨停/高开污染，而是 Top3 重复度高、真实账户不能重复加仓，导致大量补位到 rank 4-10。
- bt-amv-topn duplicate lot 诊断: 新增 `allow_duplicate_positions` 配置，默认 `false` 保持旧口径；duplicate rolling 诊断显示 `trend P1/K1/PB1/CP0/RV1` 从 `+60.17%` 升至 `+106.50%`，PB3 仅从 `+99.62%` 小升至 `+102.93%`，坐实 trend-only 的主要损耗来自 no-repeat 持仓语义。

## 2026-05-28

### [AMV] P3 annual restart cadence 快速诊断

- 背景:
  - 用户指出此前 static cadence sensitivity 只改变 2021 起始 offset，本质上仍是一条连续 6td 节奏的相位平移
  - 真正担心是: 2022/2023/2024/2025/2026 如果每年独立重启，P3 是否仍稳；这比全样本起点敏感性更贴近实盘中断/重启风险
- 新增脚本:
  - `scripts/amv_annual_restart_cadence.py`
  - 口径: Python-side no-cost；每年单独重启；每年 offset `0..6`；`T+1 open -> D+7 close`；跳过执行日开盘涨停；不含成本
- 当前 Mac 限制:
  - 本机缺少另一台设备上的 sector/context signal artifacts，且 `sector_map_em.csv` 不存在
  - 东方财富行业映射网络刷新失败，因此今晚只快速跑 `raw P3` 与 `medium128 p0.03`
  - sector complete 与 context combo annual restart 留待有完整行业映射/信号 artifact 的设备补跑
- 信号:
  - raw P3: `artifacts/amv_static_sleeve_signals/20260528_213312_candidate_p3_k0p5_b0_c0_r0/`
  - medium128: `artifacts/amv_static_sleeve_signals/20260528_213422_p3_medium128_quality_linear_t0p5_p0p03/`
- 报告:
  - `reports/amv_p3_annual_restart_cadence_quick.json`
- raw P3 annual restart:
  - 2021: worst `+5.06%`, median `+12.30%`, best `+15.85%`, positive `7/7`
  - 2022: worst `+26.04%`, median `+39.97%`, best `+51.44%`, positive `7/7`
  - 2023: worst `+13.83%`, median `+21.86%`, best `+41.43%`, positive `7/7`
  - 2024: worst `+40.67%`, median `+44.89%`, best `+53.18%`, positive `7/7`
  - 2025: worst `+2.45%`, median `+4.62%`, best `+19.55%`, positive `7/7`
  - 2026: worst `-8.55%`, median `-2.16%`, best `+14.57%`, positive `2/7`
- medium128 p0.03 annual restart:
  - 2021: worst `+11.85%`, median `+18.57%`, best `+25.53%`, positive `7/7`
  - 2022: worst `+24.42%`, median `+38.17%`, best `+49.49%`, positive `7/7`
  - 2023: worst `+6.87%`, median `+19.54%`, best `+31.58%`, positive `7/7`
  - 2024: worst `+46.95%`, median `+50.84%`, best `+59.47%`, positive `7/7`
  - 2025: worst `+1.58%`, median `+4.91%`, best `+36.17%`, positive `7/7`
  - 2026: worst `-9.72%`, median `+1.85%`, best `+16.20%`, positive `5/7`
- 初步判断:
  - raw P3 并没有被 annual restart 直接推翻: 2021-2025 每年 `7/7` offset 为正，说明它不是单纯依赖 2021 初始 cadence
  - 但 2026 暴露出明显路径脆弱性: raw P3 只有 `2/7` offset 为正，median 为负
  - medium128 改善了 2026 的正 offset 数量和 median，但没有改善 2026 最差 offset；同时略弱于 raw 的年份包括 2022/2023
  - 结论不是“P3 可放心升级”，而是“P3 的年度重启风险主要集中在 2026；上下文增强需要重点看是否修复 2026，而不能只看全样本总收益”
  - 下一步必须补跑 sector complete 与 context combo 的 annual restart；若 context combo 仍在 2026 最差 offset 上受伤，则不能作为默认主策略，只能作为 challenger/forward 监控

### [AMV] 下一轮上下文因子路线

- 背景:
  - 上一轮行业顺风因子已阶段性收口，`mix10/20 + linear + rel20_under0 + p0.03` 可作为 P3 主线 challenger
  - 但当前约定不是马上补组合层 allocation，而是继续完成下一轮 A 股上下文因子路线
  - 核心判断: 量价还有空间，但不是继续堆 RSI/RSRS/MACD 这类通用指标；最该补的是“板块、情绪、趋势质量、涨停生态”四类 A 股上下文因子
  - 路线目标: 解释并减少 P3 的假突破、2026 弱段和 AMV bull 内部赚钱效应不足的问题，而不是立刻替换当前主底座或马上做组合层 allocation
- 板块/行业顺风因子:
  - 这是当前最建议先做的方向
  - P3 2026 的问题不是个股形态完全错，而是选到了水泥、地产、航天这类当时不在主线里的假突破
  - 可做因子: 行业 5/10/20 日收益排名、行业内上涨家数占比、行业内新高占比、行业成交额扩张、个股相对行业强弱
  - 用法: 更适合做 gate 或加权，而不是单独排序
  - 当前状态: 已完成第一轮从 trade diagnostic 到 Rust rerank/cadence 的技术收口
  - 当前候选: `mix10/20 + linear + rel20_under0 + p0.03` 可列为 P3 主线 challenger，但不直接替换默认 P3
  - 未完成验收: 历史行业分类 / 行业映射版本复核、forward 样本、以及最终组合层影响
- 市场赚钱效应 / 情绪温度因子:
  - 这是下一步优先做的方向
  - A 股很吃这个；纯 AMV bull 还不够，需要知道当下是不是“有赚钱效应的牛”，还是“指数没坏但短线很难做”
  - 可做因子: 全市场涨停数、跌停数、炸板率、连板高度、昨日涨停次日溢价、20 日新高家数、强势股回撤幅度
  - 用法: 对 P3 的入场过滤很有价值，先做 trade-level diagnostic，再决定是否做 gate / rerank / soft penalty
  - 预期验证问题: P3 在低赚钱效应、高跌停、高炸板、昨日涨停无溢价、强势股回撤较深的环境下，是否更容易出现假突破和亏损交易
- 中期结构因子，特别是 64/128 日窗口:
  - 博主早期强调 128 日日 K 窗口，当前体系更多是 20 日附近的形态，可能缺一个中期结构视角
  - 可做因子: 128 日收益分位、128 日高低位、趋势效率、回撤后修复比例、波动收缩程度、长期均线斜率
  - 用法: 不一定直接提高 Top3 收益，更可能帮助区分“真趋势票”和“短期假突破”；优先作为 P3 假突破诊断和候选加权项
- 趋势质量因子:
  - 不是看涨了多少，而是看涨得是否顺；A 股里很多票短期涨幅强，但路径很脏，容易回撤
  - 可做因子: 收益 / 波动、上涨天数占比、K 线实体效率、回撤深度、趋势线性度、均线排列稳定性
  - 用法: 适合补 P3，因为 P3 当前偏“贴近新高”，但还缺“新高质量”
- 流动性冲击 / 成交额异动因子:
  - 当前已有一些成交额/波动相关东西，但还可以更系统
  - 可做因子: 成交额相对 20/60 日分位、换手率突增、放量不涨、缩量回调、突破日量能确认、流动性枯竭后恢复
  - 用法: 对 PB3 可能尤其有用，因为 pullback 成功往往需要“缩量回调 + 再放量”；对 P3 则更偏突破质量确认
- 涨停生态 / 强势股事件因子:
  - 这是 A 股特色，和普通量价因子不完全一样
  - 可做因子: 近 N 日是否涨停、涨停后第几天、炸板后修复、首板后回踩、反包、连板断板后承接
  - 用法: 可能形成一个独立 sleeve，而不是混进 P3/PB3；优先当作独立策略族候选
- 基本面/另类数据因子:
  - 这条放中期，不是马上第一优先
  - 可做因子: 盈利质量、营收/利润增速、ROE、估值分位、机构持仓变化、股东户数、年报长度、CEO 年龄等另类数据
  - 用法: 可能降低纯量价失效风险，但数据治理成本更高
- 执行顺序:
  - 第一阶段: 板块顺风 + 市场赚钱效应，直接对应 P3 当前最大痛点，即假突破和 2026 弱段
  - 第二阶段: 128 日中期结构 + 趋势质量，补的是博主早期窗口里可能有、但当前体系还没充分表达的东西
  - 第二阶段偏后: 流动性冲击 / 成交额异动，优先服务 PB3，也可辅助 P3 的突破质量确认
  - 第三阶段: 涨停生态 sleeve，可能是真正低相关的新策略族，不一定和 P3/PB3 混在一起
  - 基本面/另类数据留作中期数据治理分支，不打断当前量价上下文路线

### [AMV] Market sentiment / 情绪温度初筛

- 目标:
  - 继续第一阶段“板块顺风 + 市场赚钱效应”的另一半
  - 验证 P3 假突破是否来自 AMV bull 内部的短线赚钱效应不足，或来自过热拥挤后的回撤
  - 只做 trade-level diagnostic，不直接改信号、不接组合层 allocation
- 新增脚本: `scripts/amv_market_sentiment_diagnostic.py`
  - 数据: QMT 复权日线 + `build_feature_frame()` 的非 ST 全市场股票池
  - join 口径: 所有情绪特征按 `signal_date` 合成并 join 到交易，避免使用 T+1 entry day 收盘信息
  - 输出: `reports/amv_market_sentiment_diagnostic.json`
  - Canvas: `reports/canvases/amv-market-sentiment-diagnostic.canvas.tsx`
- 构造特征:
  - `limit_up_count`: 按主板 10%、创业板/科创板 20% 识别收盘涨停数量
  - `limit_down_count`: 按板块涨跌幅限制识别收盘跌停数量
  - `failed_limit_up_ratio`: 日内触及涨停但未收盘封住的近似炸板率
  - `new_high_20_ratio`: 收盘创 20 日新高的股票占比
  - `yday_limit_up_close_premium`: 昨日收盘涨停股在今日收盘的平均溢价
  - `strong_stock_drawdown_20d_median`: 20 日收益 top20% 强势股相对 20 日高点的中位回撤
- P3 static strict 结果:
  - 总样本: `274` 笔，trade PnL `+1,008.5K`
  - 简单冷市场过滤不成立:
    - 低涨停数 bucket: `57` 笔，PnL `+272.3K`
    - 低 20 日新高占比 bucket: `63` 笔，PnL `+497.2K`
    - 低市场上涨占比 bucket: `78` 笔，PnL `+495.3K`
    - 这些规则会大量误杀大赢家，不能作为 P3 cash/gate 方向
  - 反而是过热/拥挤环境更值得继续:
    - `skip_hot_yday_limit_up_premium`: 跳过 `95` 笔，跳过 PnL `-120.6K`，trade-level delta `+120.6K`
    - `skip_hot_yday_premium_and_new_high`: 跳过 `54` 笔，跳过 PnL `-132.5K`，trade-level delta `+132.5K`
    - 后者 2026 delta `+62.9K`，但 2022 delta `-18.0K`
    - 误杀 `3` 笔 `>20K` 大赢家，避开 `6` 笔 `<-20K` 大亏损
  - 当前判断: P3 的市场情绪因子更像“过热/拥挤过滤”，不是“冷市场过滤”
- PB3 rolling raw 对照:
  - 总样本: `1650` 笔，trade PnL `+498.1K`
  - `skip_low_limit_up_count`: 跳过 `396` 笔，跳过 PnL `-24.0K`，trade-level delta `+24.0K`
  - 无 `>20K` 大赢家误杀，但分年并不稳定，2022/2026 为负
  - 当前判断: PB3 对全市场涨停数量低迷有一点敏感，但信号弱于既有 regime gating，暂不优先升级
- 下一步:
  - 将 `hot_yday_premium_and_new_high` 做成 P3 signal-level gate 或 soft penalty
  - 接 `bt-amv-topn` static strict Rust 验证，看 trade-level delta 是否能兑现到账户路径
  - 若 Rust 成立，再做阈值邻域、cadence、与 sector-tailwind 的交互验证

### [AMV] P3 market sentiment hot gate 接 Rust

- 目标:
  - 将 market sentiment 初筛里最强的 trade-level 规则接到可执行信号层
  - 规则是 AMV bull 内部的过热/拥挤过滤，不替代 AMV bull
  - 第一版采用 date-level hard gate: `yday_limit_up_close_premium_rank >= 0.67` 且 `new_high_20_ratio_rank >= 0.67` 的 signal_date 不开新仓
- 新增脚本: `scripts/amv_market_sentiment_signal_export.py`
  - 输出 artifact: `artifacts/amv_static_sleeve_signals/20260528_165842_p3_sentiment_hot_yday_premium_newhigh_gate_yp0p67_nh0p67`
  - 修正点: 初版 summary 的 raw overlap 没有限定候选池，已改为候选池内 raw rank；重导出后 `raw_top3_overlap_ratio = 1.0`
  - 导出优化: 避免重复构建全市场 feature frame，Windows 上重导出耗时从卡住降到约 `38s`
- Rust 回测:
  - 配置: `backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml`
  - 输出: `artifacts/amv_static_sleeve_signals/20260528_165842_p3_sentiment_hot_yday_premium_newhigh_gate_yp0p67_nh0p67/backtests/6td_static_strict_top3_no_stop_20260528_165842_p3_sentiment_hot_yday_premium_newhigh_gate_yp0p67_nh0p67`
  - raw P3: total `+201.69%`, MaxDD `13.52%`, trades `274`, win rate `52.55%`
  - sentiment hard gate: total `+182.02%`, MaxDD `13.53%`, trades `264`, win rate `53.03%`
  - delta: total `-19.67pp`, MaxDD `+0.01pp`, trades `-10`
- 当前判断:
  - trade-level 规则没有兑现到账户路径；严格 date-level hard gate 被否决
  - 它没有降低回撤，反而减少了复利路径上的收益
  - 市场情绪不是 P3 的独立硬过滤器；如果继续，只能改成更温和的 soft penalty / rerank，或与 sector-tailwind 交互，而不是单独删除整天信号

### [AMV] 第二阶段 64/128 日中期结构 + 趋势质量

- 目标:
  - 进入上下文因子路线第二阶段，补当前 P3 主要依赖 20 日附近形态、但缺少中期结构和“新高质量”确认的问题
  - 验证 64/128 日窗口是否能区分“真趋势票”和“短期假突破”，尤其是 P3 在贴近新高时是否需要更顺的中期路径
  - 先做 trade-level diagnostic，再用 soft penalty/rerank 接 Rust；不直接做 hard gate
- 新增诊断脚本: `scripts/amv_medium_trend_quality_diagnostic.py`
  - 输出: `reports/amv_medium_trend_quality_diagnostic.json`
  - Canvas: `reports/canvases/amv-medium-trend-quality-diagnostic.canvas.tsx`
  - 数据口径: 使用 `build_feature_frame()` 的 QMT 复权日线、非 ST 与 AMV bull 候选池特征，在 `signal_date` 将个股级中期结构/趋势质量 join 到 P3 static strict 和 PB3 rolling trades
  - 中期结构分: 64/128 日收益分位、区间位置分位、长期均线斜率分位的均值
  - 趋势质量分: 趋势效率、上涨天数占比、收益/波动、K 线实体效率分位的均值
  - 趋势效率定义: 中期绝对收益 / 同窗口逐日绝对收益和，用来近似“涨得是否顺”，而不是只看涨幅
- P3 diagnostic 发现:
  - 总样本: `274` 笔，trade PnL `+1,008.5K`
  - `structure64_high`: `124` 笔，PnL `+1,075.7K`，avg `+2.80%`，win `55.6%`
  - `structure64_low`: `33` 笔，PnL `-167.0K`，avg `-1.37%`，win `39.4%`
  - `quality128_high`: `37` 笔，PnL `+626.1K`，avg `+5.57%`，win `64.9%`
  - `pos128_high`: `150` 笔，PnL `+979.3K`；`pos128_low`: `43` 笔，PnL `-122.7K`
  - 单独 `low_structure_128` 或 `low_quality_128` hard skip 不成立，会误杀 2026 大赢家；`skip_medium_structure_and_quality_weak` 在 trade-level 跳过 `90` 笔、delta `+97.2K`，但 2026 delta `-28.7K`
  - 当前解释: 中期结构/质量对 P3 有解释力，但第一版应做 soft penalty/rerank，而不是删除信号
- PB3 对照:
  - PB3 的 `quality128_high` 和 `retvol128_high` 也更好，但最佳规则 `skip_64_128_quality_both_low` 只贡献 `+6.3K`
  - 当前判断: 第二阶段先服务 P3，PB3 暂不优先接入
- 新增 signal export: `scripts/amv_medium_trend_quality_signal_export.py`
  - 规则: 对 `structure_score_128d < 0.5` 且 `trend_quality_score_128d < 0.5` 的 P3 候选做 linear soft penalty，再重新排序 Top3
  - 导出 artifact:
    - `artifacts/amv_static_sleeve_signals/20260528_171844_p3_medium128_quality_linear_t0p5_p0p01`
    - `artifacts/amv_static_sleeve_signals/20260528_171848_p3_medium128_quality_linear_t0p5_p0p02`
    - `artifacts/amv_static_sleeve_signals/20260528_171853_p3_medium128_quality_linear_t0p5_p0p03`
  - raw Top3 overlap: `0.01` 为 `93.24%`，`0.02` 为 `87.43%`，`0.03` 为 `83.54%`
- Rust static strict 回测:
  - 配置: `backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml`
  - 汇总报告: `reports/amv_p3_medium_trend_quality_rust_summary.json`
  - 归因报告: `reports/amv_p3_medium_trend_quality_attribution.json`
  - cadence 报告: `reports/amv_p3_medium_trend_quality_cadence.json`
  - raw P3: total `+201.69%`, MaxDD `13.52%`, trades `274`, win `52.55%`
  - `linear p0.01`: total `+201.10%`, MaxDD `14.71%`, trades `274`, win `54.74%`
  - `linear p0.02`: total `+261.77%`, MaxDD `15.33%`, trades `274`, win `54.74%`
  - `linear p0.03`: total `+264.90%`, MaxDD `14.05%`, trades `274`, win `54.38%`
  - 当前最佳是 `p0.03`: 相对 raw total `+63.21pp`，MaxDD `+0.53pp`
- `p0.03` 归因:
  - exact overlap `227/274 = 82.85%`
  - raw-only `47` 笔合计 `-68.6K`; 128 日中期结构 / 趋势质量 rerank-only `47` 笔合计 `+161.3K`; unique trade delta `+230.0K`
  - 共同交易仍贡献 `+86.1K`，来自资金路径/仓位规模差异
  - 年度收益: 2021 `+3.92pp`, 2022 `-0.92pp`, 2023 `+1.96pp`, 2024 `+10.96pp`, 2025 `+9.30pp`, 2026 `-0.56pp`
  - 最强正贡献月份: 2025-08 trade delta `+139.8K`; 2026-04 delta `+59.2K`; 2024-11 delta `+51.5K`
  - 最大负贡献月份: 2026-01 trade delta `-110.7K`
- cadence:
  - no-cost Python-like static 7 offset: `7/7` 个 offset 优于 raw
  - raw worst/median/best: `+260.74% / +285.60% / +297.79%`
  - 128 日中期结构 / 趋势质量 p0.03 worst/median/best: `+284.61% / +307.71% / +331.63%`
  - delta worst/median/best: `+21.24pp / +22.64pp / +33.85pp`
- 当前判断:
  - 第二阶段首轮产生了强 challenger，强度超过上一轮 sector-tailwind 单独 challenger
  - 但它仍牺牲 2026-01，说明 128 日结构/质量惩罚会压掉某些短期爆发票
  - 暂不直接替换默认 P3；下一步先做 128 日中期结构 / 趋势质量与 sector-tailwind 的交互/组合复核，再决定是否升级为主线 challenger

### [AMV] 第二阶段验收复核: 128 日中期结构 / 趋势质量参数邻域 + sector 交互

- 目标:
  - 回答第二阶段是否可以收口
  - 验证 `medium128 p0.03` 是否只是边界偶然；这里的 `medium128` 指 128 日中期结构 / 趋势质量
  - 验证第一阶段 sector-tailwind challenger 与第二阶段 128 日中期结构 / 趋势质量是互补还是互相打架
- 参数邻域:
  - 新导出 artifact:
    - `artifacts/amv_static_sleeve_signals/20260528_174416_p3_medium128_quality_linear_t0p5_p0p025`
    - `artifacts/amv_static_sleeve_signals/20260528_174421_p3_medium128_quality_linear_t0p5_p0p035`
    - `artifacts/amv_static_sleeve_signals/20260528_174425_p3_medium128_quality_linear_t0p5_p0p04`
  - Rust static strict:
    - raw P3: total `+201.69%`, MaxDD `13.52%`
    - `medium p0.025`: total `+251.32%`, MaxDD `15.33%`
    - `medium p0.03`: total `+264.90%`, MaxDD `14.05%`
    - `medium p0.035`: total `+240.97%`, MaxDD `16.07%`
    - `medium p0.04`: total `+225.43%`, MaxDD `16.13%`
  - 结论: `p0.03` 是局部峰值；`p0.025` 仍强于 raw，但 `p0.035/0.04` 收益回落且回撤扩大，说明不是惩罚越强越好
- 新增组合导出脚本: `scripts/amv_context_combo_signal_export.py`
  - 组合逻辑: P3 基础分 - sector-tailwind penalty - 128 日中期结构 / 趋势质量 penalty
  - sector 口径: `mix10/20 + linear + bottom40 + rel20_under0`
  - 128 日中期结构 / 趋势质量口径: `structure_score_128d < 0.5` 且 `trend_quality_score_128d < 0.5` 的 linear penalty
  - 组合 artifact:
    - `artifacts/amv_static_sleeve_signals/20260528_174808_p3_ctx_sectormix1020_linear_b0p4_sp0p02_medium128_linear_t0p5_mp0p03_rel20_under0`
    - `artifacts/amv_static_sleeve_signals/20260528_174814_p3_ctx_sectormix1020_linear_b0p4_sp0p03_medium128_linear_t0p5_mp0p03_rel20_under0`
- 组合 Rust:
  - `sector p0.02 + 128 日中期结构 / 趋势质量 p0.03`: total `+272.06%`, MaxDD `14.05%`, win `54.74%`
  - `sector p0.03 + 128 日中期结构 / 趋势质量 p0.03`: total `+247.51%`, MaxDD `14.05%`, win `54.01%`
  - 结论: sector 与 128 日中期结构 / 趋势质量有互补，但叠加后 sector penalty 要降档；`sector p0.03` 在组合中惩罚过强
- 最佳组合归因:
  - 汇总报告: `reports/amv_p3_context_combo_validation.json`
  - raw vs combo 归因: `reports/amv_p3_context_combo_attribution.json`
  - 单独 128 日中期结构 / 趋势质量 vs combo 归因: `reports/amv_p3_context_combo_vs_medium_attribution.json`
  - 相对 raw: total `+70.37pp`, MaxDD `+0.53pp`, exact overlap `226/274`
  - raw-only `48` 笔合计 `-83.8K`; combo-only `48` 笔合计 `+163.1K`; unique trade delta `+246.9K`
  - 相对单独 128 日中期结构 / 趋势质量: total `+7.16pp`, MaxDD 几乎不变，exact overlap `272/274`
  - 年度相对 raw: 2021 `+4.64pp`, 2022 `-0.79pp`, 2023 `+3.44pp`, 2024 `+11.07pp`, 2025 `+9.24pp`, 2026 `-0.57pp`
  - 最强正贡献月份: 2025-08 trade delta `+144.2K`; 2026-04 `+58.5K`; 2024-11 `+53.8K`
  - 最大负贡献月份仍是 2026-01 trade delta `-110.4K`
- cadence:
  - 报告: `reports/amv_p3_context_combo_cadence.json`
  - no-cost Python-like static 7 offset: `7/7` 个 offset 优于 raw
  - raw worst/median/best: `+260.74% / +285.60% / +297.79%`
  - combo worst/median/best: `+292.95% / +316.55% / +340.99%`
  - delta worst/median/best: `+29.74pp / +31.69pp / +43.21pp`
- 当前判断:
  - 第二阶段验收复核通过，可以阶段性收口
  - 最佳候选从单独 `128 日中期结构 / 趋势质量 p0.03` 升级为 `sector p0.02 + 128 日中期结构 / 趋势质量 p0.03` 的组合上下文 rerank
  - 仍不直接替换默认 P3，因为 2026-01 的短期爆发票被牺牲，后续需 forward 监控和最终路线对比

### [AMV] P3 sector tailwind rerank 接 Rust

- 目标: 将 sector tailwind 从事后 trade-level 过滤推进到可执行信号层，在 P3 候选池内做 soft penalty + 重新排序，再接 `bt-amv-topn` 静态 strict Top3。
- 新增脚本: `scripts/amv_sector_tailwind_signal_export.py`
  - 基础: `candidate_p3_k0p5_b0_c0_r0`
  - 规则: 对行业 10 日收益 rank bottom 40% 候选扣分，扣分强度扫描 `0.01 / 0.02 / 0.03 / 0.05`
  - 产物: `artifacts/amv_static_sleeve_signals/20260528_143430_p3_sector_bottom40_penalty_0p01` 等
  - 输出字段仍兼容 `bt-amv-topn` 的 `signal.parquet + signal.meta.json`
- Rust 回测:
  - 配置: `backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml`
  - 汇总报告: `reports/amv_p3_sector_tailwind_rerank_summary.json`
  - 归因报告: `reports/amv_p3_sector_tailwind_rerank_attribution.json`
  - Canvas: `reports/canvases/amv-p3-sector-tailwind-rerank.canvas.tsx`
- penalty grid:
  - raw P3: total `+201.69%`, CAGR `22.98%`, Sharpe `1.22`, MaxDD `13.52%`, 2026 `-0.77%`
  - `0.01`: total `+196.62%`, CAGR `22.59%`, Sharpe `1.20`, MaxDD `15.84%`, 2026 `-1.86%`
  - `0.02`: total `+242.10%`, CAGR `25.91%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`
  - `0.03`: total `+223.56%`, CAGR `24.60%`, Sharpe `1.26`, MaxDD `13.47%`, 2026 `-5.50%`
  - `0.05`: total `+226.92%`, CAGR `24.84%`, Sharpe `1.27`, MaxDD `13.47%`, 2026 `-5.48%`
- 归因:
  - `0.02` vs raw P3: total `+40.41pp`, MaxDD `-0.08pp`
  - exact overlap `216/274 = 78.83%`
  - raw-only `58` 笔合计 `-60.4K`; rerank-only `58` 笔合计 `+84.0K`; unique trade delta `+144.4K`
  - 共同交易仍贡献 `+57.7K`，来自资金路径/仓位规模差异
  - 年度: 2021 `+12.22pp`, 2024 `+11.21pp`, 2025 `+7.10pp`, 2026 `+1.40pp`; 2022 `-11.18pp`, 2023 `-5.40pp`
- 当前判断:
  - sector tailwind soft penalty 是 P3 方向目前最有价值的新因子增强之一，明显优于硬 gate
  - 最佳点暂为 `0.02`，但 `0.03/0.05` 打坏 2026，说明参数并非越强越好
  - 进入候选前还必须做起始 offset / cadence、行业映射版本、rank 窗口与 bottom 阈值稳健性验证；当前静态东方财富行业映射存在历史分类偏差

### [AMV] P3 sector tailwind focused robustness

- 目标: 验证 `10d / bottom40 / penalty=0.02` 是否只是单点参数拟合。
- 脚本更新: `scripts/amv_sector_tailwind_signal_export.py`
  - 新增 `--rank-window 5/10/20`
  - artifact 名称写入 `rank_window / bottom_threshold / penalty`，避免 grid 产物混淆
- 产物:
  - JSON: `reports/amv_p3_sector_tailwind_robustness.json`
  - Canvas: `reports/canvases/amv-p3-sector-tailwind-robustness.canvas.tsx`
- penalty 邻域，固定 `10d / bottom40`:
  - raw P3: total `+201.69%`, Sharpe `1.22`, MaxDD `13.52%`, 2026 `-0.77%`
  - `0.015`: total `+191.28%`, Sharpe `1.18`, MaxDD `15.79%`, 2026 `-1.86%`
  - `0.018`: total `+227.55%`, Sharpe `1.28`, MaxDD `13.44%`, 2026 `-1.85%`
  - `0.020`: total `+242.10%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`
  - `0.022`: total `+239.74%`, Sharpe `1.31`, MaxDD `13.47%`, 2026 `+0.64%`
  - `0.025`: total `+239.74%`, Sharpe `1.31`, MaxDD `13.47%`, 2026 `+0.64%`
- threshold 敏感性，固定 `10d / penalty=0.02`:
  - bottom30: total `+226.99%`, Sharpe `1.31`, MaxDD `13.52%`, 2026 `+1.29%`
  - bottom40: total `+242.10%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`
  - bottom50: total `+138.70%`, Sharpe `0.96`, MaxDD `20.65%`, 2026 `+0.71%`
- rank window 敏感性，固定 `bottom40 / penalty=0.02`:
  - `5d`: total `+222.04%`, Sharpe `1.26`, MaxDD `16.62%`, 2026 `+1.56%`
  - `10d`: total `+242.10%`, Sharpe `1.32`, MaxDD `13.44%`, 2026 `+0.63%`
  - `20d`: total `+264.26%`, Sharpe `1.36`, MaxDD `16.09%`, 2026 `+0.41%`
- 当前判断:
  - sector tailwind rerank 的方向通过第一轮 focused robustness：不是单点 `0.020` 的孤立结果
  - 可用 penalty 区间大致为 `0.018~0.025`；`0.015` 太弱，`bottom50` 太宽会过度惩罚
  - `20d` 窗口是进攻 challenger，收益更高但回撤更深；当前默认候选仍保留 `10d / bottom40 / 0.02`
  - 未完成项: static cadence offset、行业映射版本（申万/中信历史行业）、更细窗口/阈值 walk-forward

### [AMV] P3 sector tailwind cadence check

- 目标: 检查 P3 sector tailwind rerank 是否只是默认 static 起始 offset 占优。
- 新增脚本: `scripts/amv_sector_tailwind_cadence.py`
  - 口径: Python-side no-cost static cadence sensitivity
  - 选择: `rank <= 3`, `is_signal`, `is_bull_regime`, skip open limit-up, no refill
  - entry: shifted `signal.parquet` execution date open
  - exit: entry date + `6` trading days close
  - 注意: 该口径用于起始节奏敏感性，不替代 Rust account NAV
- 产物:
  - `reports/amv_p3_sector_tailwind_cadence.json`
  - `reports/amv_p3_sector_tailwind_cadence_w20.json`
  - `reports/canvases/amv-p3-sector-tailwind-cadence.canvas.tsx`
- `10d / bottom40 / penalty=0.02`:
  - raw P3 median/worst offset total: `+285.60%` / `+260.74%`
  - rerank median/worst offset total: `+275.03%` / `+237.56%`
  - positive delta offsets: `2/7`
  - median delta: `-10.56pp`
  - worst delta: `-24.16pp`
  - 判断: 没有通过 cadence 检查，不能直接升级为默认 P3
- `20d / bottom40 / penalty=0.02` challenger:
  - rerank median/worst offset total: `+292.07%` / `+238.49%`
  - positive delta offsets: `5/7`
  - median delta: `+9.10pp`
  - worst delta: `-22.25pp`
  - 判断: 比 10d 更稳，但仍有单个 offset 明显受伤，且最差路径弱于 raw；只能作为 challenger
- 当前判断:
  - sector tailwind 的“弱行业离散扣分”不够平滑，默认 Rust 起点的漂亮结果有 cadence 偶然性
  - 下一步若继续，应尝试连续型行业 rank penalty、`10d/20d` 混合项、或只在特定 AMV/行业状态下启用，而不是继续押注 `10d/bottom40/0.02`

### [AMV] P3 sector tailwind complete expression

- 目标: 不停在离散弱行业扣分失败点，改做更平滑、局部、更少扰动的行业表达。
- 脚本更新:
  - `scripts/amv_sector_tailwind_diagnostic.py`: 新增 `stock_ret_10d` 与 `stock_rel_sector_ret_10d`
  - `scripts/amv_sector_tailwind_signal_export.py`: 新增 `--rank-source 5d/10d/20d/mix_10_20`, `--penalty-mode bucket/linear`, `--relative-confirm none/rel5_under0/rel10_under0/rel20_under0`
- focused grid:
  - `10d linear`, `mix10/20 linear`, `20d linear`, `mix10/20 linear + rel20_under0`
  - Rust 配置: `backtest-engine/crates/amv-topn/config_6td_static_strict_top3_no_stop.toml`
  - 汇总报告: `reports/amv_p3_sector_tailwind_complete_grid.json`
  - Canvas: `reports/canvases/amv-p3-sector-tailwind-complete.canvas.tsx`
- Rust 结果:
  - raw P3: total `+201.69%`, Sharpe `1.225`, MaxDD `13.52%`
  - `mix10/20 linear rel20 p0.03`: total `+219.55%`, Sharpe `1.279`, MaxDD `14.07%`, excess `+17.86pp`
  - `mix10/20 linear rel20 p0.04`: total `+219.55%`, Sharpe `1.279`, MaxDD `14.07%`, excess `+17.86pp`
  - `10d linear p0.04`: total `+218.69%`, Sharpe `1.265`, MaxDD `13.51%`, excess `+17.00pp`
  - 直接 `mix10/20 linear` 或 `20d linear` 不加相对行业确认时效果明显变弱，说明“少动、只扣交集弱势样本”比宽泛行业弱势扣分更稳。
- cadence:
  - 报告: `reports/amv_p3_sector_tailwind_complete_cadence_p0p03.json`
  - `mix10/20 linear rel20 p0.03` 的 Python no-cost 7 offset 结果: `7/7` 个 offset 优于 raw
  - raw median/worst offset total: `+285.60%` / `+260.74%`
  - rerank median/worst offset total: `+306.53%` / `+280.32%`
  - median delta `+20.94pp`, worst delta `+19.59pp`
- 当前判断:
  - 行业因子没有被否决；被否决的是最早的离散 bucket 表达
  - 推荐将 `mix10/20 + linear + rel20_under0 + p0.03` 升为 P3 主线 challenger，而不是直接替换默认 P3
  - `p0.03` 与 `p0.04` 当前回测/cadence 结果相同，优先选择更保守的 `p0.03`
  - 当前阶段先技术收口，不立即接历史行业分类数据源；原因是该项更偏数据工程，容易打断下一轮策略路线
  - 历史行业分类/行业映射版本复核保留为最终验收项；完整走完下一轮路线后，再用申万/中信/历史成分数据复核静态东方财富行业映射带来的历史分类偏差

### [AMV] Sector tailwind P3/PB3 初筛

- 目标: 验证“板块/行业顺风”是否能解释 P3 假突破，并判断是否值得进入 P3 gating / rerank。
- 新增脚本: `scripts/amv_sector_tailwind_diagnostic.py`
  - 数据: 静态东方财富行业映射 `data/sector_map_em.csv` + QMT `v_stock_daily_qfq_qmt`
  - 行业映射: `5,549` 只股票、`86` 个行业；QMT 日线匹配 `5,180` 只股票，缺失日线行数约 `0.10%`
  - 口径: 行业因子在 `signal_date` 合成并 join 到交易，避免 T+1 open 使用 entry day 收盘信息
  - 特征: 行业 5/10/20 日等权收益、收益横截面 rank、MA20 宽度、20/60 日新高占比、成交额扩张、个股相对行业强弱
- 产物:
  - JSON: `reports/amv_sector_tailwind_diagnostic.json`
  - Canvas: `reports/canvases/amv-sector-tailwind-diagnostic.canvas.tsx`
- P3 结果:
  - 行业 10 日收益 rank top 30%: `123` 笔，PnL `+661.9K`，avg pnl `+1.78%`
  - mid 30%: `84` 笔，PnL `+390.4K`，avg pnl `+1.60%`
  - bottom 40%: `67` 笔，PnL `-43.9K`，avg pnl `+0.09%`
  - trade-level what-if `skip_sector_bottom_40pct`: delta `+43.9K`，跳过 `67` 笔，误杀 `2` 笔 >20K 大赢家，避开 `6` 笔 <-20K 大亏损；年度上 2021/2024/2025/2026 为正，2022/2023 为负
  - 严格只保留 `tailwind_ok` 不成立: 会跳过 `169` 笔合计 `+369.2K` 的交易，说明行业顺风适合作为弱过滤/加权，不适合作为硬 gate
- PB3 结果:
  - bottom 40% 行业 rank 交易仍合计 `+257.0K`，直接过滤会显著变差
  - `aged/old + bottom industry` 也只是接近打平，delta `-6.5K`
- 当前判断:
  - 行业顺风对 P3 有初步价值，尤其能解释一部分弱行业假突破；下一步应做 P3 rerank / soft penalty，而不是简单删除所有非顺风行业
  - PB3 是 pullback/reversion sleeve，不应套同一套顺风过滤
  - 当前行业映射是静态东方财富行业，存在历史分类偏差；若后续信号稳定，应再换成申万/中信历史行业分类复核

## 2026-05-27

### [AMV] P3 + PB3 gated allocation 诊断

- 目标: 从“单策略研究”收敛到当前最像实盘雏形的组合结构，验证 P3 static + PB3 rolling gated 在组合层是否改善 2025/2026 与回撤
- 新增脚本: `scripts/amv_allocation_diagnostic.py`
  - 输入: 三条 Rust `daily_equity.csv`
    - P3 static strict: `20260520_092049_candidate_p3_k0p5_b0_c0_r0`
    - PB3 rolling raw: `20260521_090945_pullback_p0_k0_pb3_cp1_rv0`
    - PB3 rolling gated: `20260526_184023_pullback_p0_k0_pb3_cp1_rv0`
  - 方法: 使用 daily return 做 daily rebalanced synthetic allocation
  - 权重: P3 `100/90/80/70/60/50%`，PB3 为剩余权重
- 产物:
  - JSON: `reports/amv_p3_pb3_gated_allocation.json`
  - Canvas: `reports/canvases/amv-p3-pb3-gated-allocation.canvas.tsx`
- 相关性:
  - P3 vs PB3 raw: `0.255`
  - P3 vs PB3 gated: `0.260`
  - PB3 raw vs PB3 gated: `0.958`
  - 结论: gated PB3 仍然是同一个 pullback sleeve 的风控增强，不是新 sleeve
- standalone:
  - P3 static: total `+201.69%`, MaxDD `13.52%`, 2025 `+13.89%`, 2026 `-0.77%`
  - PB3 raw: total `+99.62%`, MaxDD `20.70%`, 2025 `+29.73%`, 2026 `+15.15%`
  - PB3 gated: total `+109.73%`, MaxDD `16.20%`, 2025 `+25.15%`, 2026 `+14.28%`
- gated allocation:
  - P3 90 / PB3 gated 10: total `+193.66%`, MaxDD `12.41%`, 2026 `+0.71%`
  - P3 80 / PB3 gated 20: total `+185.25%`, MaxDD `11.64%`, 2026 `+2.20%`
  - P3 70 / PB3 gated 30: total `+176.50%`, MaxDD `10.87%`, 2026 `+3.70%`
  - P3 60 / PB3 gated 40: total `+167.47%`, MaxDD `10.10%`, 2026 `+5.20%`
  - P3 50 / PB3 gated 50: total `+158.19%`, MaxDD `9.33%`, 2026 `+6.70%`
- raw vs gated 同权重差异:
  - gated 在全周期 total 上更好: 80/20 `+2.56pp`, 70/30 `+3.76pp`, 60/40 `+4.90pp`
  - gated 对 MaxDD 基本小幅更优，但 50/50 时略差 `+0.66pp`
  - gated 对 2025/2026 年度收益略弱于 raw: 80/20 的 2026 差异 `-0.15pp`
- 当前判断:
  - 组合层最自然候选是 `P3 80% / PB3 gated 20%`: 保留大部分 P3 收益，把 2026 转正到 `+2.20%`，MaxDD 降至 `11.64%`
  - 若更重视回撤与 2026 修复，`70/30 gated` 更均衡，但全周期收益牺牲更明显
  - PB3 gating 在组合层的价值主要是改善 PB3 自身全周期风险质量，不是提高 2025/2026 进攻性

### [AMV] RSRS executable 初筛

- 目标: 验证网红指标 RSRS 在 AMV bull pool 中是否有单因子价值，以及是否值得进入后续 P3/PB3 组合网格
- 新增脚本: `scripts/amv_executable_rsrs_scan.py`
- RSRS 构造:
  - `beta = rolling_cov(low_adj, high_adj, N) / rolling_var(low_adj, N)`
  - `r2 = rolling_corr(low_adj, high_adj, N)^2`
  - `z = (beta - rolling_mean(beta, M)) / rolling_std(beta, M)`
  - `r2adj = z * r2`
  - `right = z * beta * r2`
- 实现备注:
  - RSRS 因子构造全程使用 Polars rolling expressions: `rolling_cov`, `rolling_var`, `rolling_corr`, `rolling_mean`, `rolling_std`
  - 未用 numpy/pandas 实现回归
  - 使用 `feature_start_date=2019-01-01` 做 warmup，再评估 `2021-01-01` 后样本，避免 `M=120/250` 长窗口起始样本不足
- 扫描:
  - 窗口: `(N=18, M=120)`, `(N=18, M=250)`
  - 因子: `beta`, `z`, `z*R2`, `z*beta*R2`
  - 同时评估 high/low 两个排序方向
  - 评估口径: `T+1 open -> D+7 close`, Top3, skip close limit-up refill
- 产物:
  - artifact: `artifacts/amv_executable_rsrs_scan/20260527_133509/`
  - 报告: `reports/amv_rsrs_executable_scan_summary.json`
- 关键结果:
  - 传统趋势方向 `rsrs_beta_18_high` 很弱: refill exec NAV `-4.89%`, ctc NAV `-26.82%`, stable positive years `1`
  - `rsrs_beta_18_low` 明显更好: refill exec NAV `+50.12%`, 但 edge 仍弱且 2026 为负
  - 最好 tradeoff: `rsrs_z_18_120_low`
    - refill exec NAV `+99.88%`
    - MaxDD `31.46%`
    - ctc NAV `+72.56%`
    - close limit-up day share `0.0%`
    - high-open day share `0.53%`
    - rank q95 `3`
    - stable positive years `4`
    - 2025 exec edge `+0.75%`, 2026 exec edge `+2.21%`
  - `rsrs_z_18_120_high` 也有一定表现: refill exec NAV `+93.36%`, 但 2026 edge 为负
- 当前判断:
  - RSRS 不是涨停污染型因子；污染很低
  - 但在 AMV bull pool 里，传统“高 RSRS 趋势突破”方向并不成立，低/标准化 RSRS 更像 pullback/reversion 线索
  - 最好 RSRS 单因子仍弱于既有 PB3/PB1 pullback label 上限，且 MaxDD 偏大，不适合作为独立 sleeve 直接接 Rust
  - 后续更合理用法: 把 `rsrs_z_18_120_low` 作为 PB/CP/RV 组合的辅助候选，或用于解释/过滤 P3 假突破；暂不做 RSRS 全量组合权重网格

### [AMV] P3 early stop 接 Rust 验证

- 目标: 将 P3 exit what-if 中“早期止损可能节省大额亏损”的想法接入 `bt-amv-topn` 真实账户回测
- 代码:
  - `bt-amv-topn` 新增 `[early_stop]` 配置段，默认关闭，不影响旧回测
  - 新增 `ExitReason::EarlyStop`
  - 支持 `trigger_hold_trading_days`, `loss_pct`, `require_previous_close_below_entry`
  - 新增 `reserve_slot_until_max_hold`，用于 P3 static: early stop 后保留仓位槽到原 `max_hold_trading_days`，避免提前释放仓位后变成 refill 策略
- 配置: `backtest-engine/crates/amv-topn/config_6td_static_strict_top3_early_stop_d2_prevneg.toml`
  - `trigger_hold_trading_days = 2`
  - `loss_pct = 0.03`
  - `require_previous_close_below_entry = true`
  - `reserve_slot_until_max_hold = true`
- 结果报告: `reports/amv_p3_early_stop_rust_summary.json`
  - raw P3 static strict: net `+201.69%`, gross `+260.86%`, MaxDD `13.52%`, trades `274`, WR `52.55%`
  - early-stop P3: net `+134.75%`, gross `+185.77%`, MaxDD `12.78%`, trades `274`, WR `51.09%`
  - 差异: net `-66.94pp`, gross `-75.09pp`, MaxDD 仅改善 `-0.75pp`
  - 触发 early stop `27` 笔；相对原始持有，`15` 笔减少亏损、`12` 笔误杀/少赚，合计少赚 `149.2K`
  - 典型误杀:
    - `sz.002456` 2024-10-22: early `-27.3K` vs raw `+50.1K`, delta `-77.5K`
    - `sh.601607` 2022-03-14: early `-8.3K` vs raw `+50.9K`, delta `-59.2K`
    - `sz.002217` 2026-04-21: early `-43.6K` vs raw `-1.9K`, delta `-41.7K`
- 关键发现:
  - P3 的突破延续机制存在“先杀后拉”的强反转样本，`d1<0 + d2<-3%` 仍不足以识别真正坏票
  - trade-level what-if 高估了止损价值；真实账户口径下，提前止损带来的资金路径、成本、误杀大反转共同压低收益
  - 如果继续做 P3 风控，优先方向不应是简单价格止损，而是结合板块顺风/个股相对 AMV 弱势/跌破结构位的复合退出
- 当前判断: 该 early-stop 规则被 Rust 否决，不进入 P3 默认配置

### [AMV] PB3 rolling gating 稳健性快筛

- 目标: 检查 PB3 AMV regime gate 是否只是 `aged / neg>=3 / amp>2.5` 的单点拟合
- 新增脚本: `scripts/amv_pb3_gating_robustness.py`
  - 输入: raw PB3 rolling Rust trades + raw signal parquet
  - 口径: actual bought trades join `signal_date`，再 join AMV phase；所有 gate 特征都按 signal date 收盘已知信息计算
  - 输出: `reports/amv_pb3_gating_robustness.json`
- 扫描范围:
  - duration gate: `aged`, `old`, `aged_or_old` 的非加速状态
  - chaos gate: `amv_neg_streak in [2,3,4]` × `amplitude_pct > [2.0,2.5,3.0,3.5]`
  - 组合: duration gate 单独、chaos 单独、duration OR chaos
- 当前 Rust 已验证规则: `aged_nonaccel_or_chaos_n3_amp2p5`
  - trade-level skip approximation: 跳过 `258/1650` 笔，跳过交易合计 PnL `-23.2K`，即贡献 `+23.2K`
  - skipped trades 平均 `-0.535%`, WR `39.9%`
  - 无 `>20K` 大赢家误杀
  - 分年: `2022 +13.8K`, `2023 +56.1K`; 但 `2021 -6.3K`, `2024 -11.3K`, `2025 -23.8K`, `2026 -5.2K`
- 最强 in-sample trade-level 规则:
  - `aged_or_old_nonaccel_or_chaos_n3_amp3p0`: `+54.5K`, skip `20.1%`, 无大赢家误杀
  - 但最差分年 `-48.4K`，明显更激进，不宜直接替换当前规则
- 更稳但弱的 chaos-only:
  - `chaos_n4_amp3p0`: 仅跳过 `0.55%` 交易，贡献 `+5.8K`，样本太少，更像风险提示而不是可单独使用的 gate
- Walk-forward:
  - 逐年用历史样本选规则，测试年合计 `+37.5K`，`3/5` 个测试年为正
  - 当前固定规则同口径测试年合计 `+29.5K`，`2/5` 个测试年为正
- 当前判断:
  - PB3 gating 的方向不是纯随机：walk-forward 仍为正，且 Rust account 口径已经验证收益与回撤同步改善
  - 但规则并不“年年有效”，核心贡献集中在 `2022/2023` 的 AMV 老化/混沌段
  - 不建议立刻切到更激进的 `aged_or_old + amp3.0`；下一步如要推进，应只做少数候选 Rust 复核，并优先关注是否降低 MaxDD，而不是追求 trade-level PnL 最大化

## 2026-05-26

### [AMV] 牛市内部阶段诊断与 gating 探索

- 目标: AMV 牛市内部 early/mid/late 阶段是否偏好不同 sleeve，能否前向构建 gating 规则
- 脚本: `scripts/amv_regime_phase_diagnostic.py`
- 产物: `reports/amv_regime_phase_diagnostic.json`
- 事后阶段（hindsight）: early 0-25%, mid 25-75%, late 75-100%, pulse <5d
  - P3: early +454K(75t,WR=48%), mid +715K(114t,WR=64%), **late -221K(76t,WR=40%)**
  - PB3: early +123K(297t), mid +384K(833t), late -4K(499t) ← late 基本打平
  - Ref: early +402K(75t), mid +609K(114t), late -218K(76t)
- 前向规则（forward-observable）:
  - P3 static: 纯 AMV 特征无法构建正向 gating（最优规则仅 +13K，4 误杀）
  - PB3 rolling: UNION 规则 `(aged+非加速) OR (neg>=3 & amp>2.5)` 跳过 18.5% 交易、净赚 **+52K (+10.5%)、零误杀**
  - 原因: P3 6d cadence 与混沌期（3-5 天）时间窗口不匹配；PB3 高频 rolling 匹配
- AMV 混沌期发现: bear trigger 前 5-10 天 AMV 日均 ret 持续为负、振幅从 2.5%→2.9%，可检测但 P3 频率不匹配
- regime_maturity（经验生存 CDF）: d10=35%, d17=49%, d30=81%

### [AMV] PB3 rolling regime gating 接 Rust 验证

- 目标: 将 PB3 rolling 的前向 gating 规则接入真实 `bt-amv-topn` 账户回测，而不是只看 trade-level what-if
- 代码:
  - `scripts/amv_static_sleeve_signal_export.py` 新增 `--pb3-regime-gate aged_non_accel_or_chaos`
  - gate 在 `signal_date` 收盘后计算，再随信号 shift 到 T+1 open，避免使用 entry day 收盘后的 AMV 信息
- 规则: `(aged + 非加速) OR (amv_neg_streak >= 3 AND amplitude_pct > 2.5)`
  - `aged + 非加速`: `fwd_duration_bucket == aged` 且 `fwd_momentum_bucket in [cruising, stalling, retreating]`
  - `chaos`: AMV 连续下跌天数 `>=3` 且当日振幅 `>2.5%`
- 导出 artifact: `artifacts/amv_static_sleeve_signals/20260526_184023_pullback_p0_k0_pb3_cp1_rv0/`
  - 原始 signal rows `8140` -> gated `6760`，过滤 `1380` 行
  - 原始 signal days `814` -> gated `676`，过滤 `138` 个信号日
- Rust 配置: `config_6td_rolling21_refill_top10_no_stop.toml`
- 结果报告: `reports/amv_pb3_regime_gating_rust_summary.json`
  - raw PB3 rolling: net `+99.62%`, gross `+130.78%`, MaxDD `20.70%`, trades `1650`, WR `48.06%`
  - gated PB3 rolling: net `+109.73%`, gross `+138.12%`, MaxDD `16.20%`, trades `1393`, WR `49.53%`
  - 改善: net `+10.10pp`, gross `+7.34pp`, MaxDD `-4.50pp`, 成本减少约 `13.8K`
  - 年度: 主要修复 `2023`（equity `-3.37% -> +7.59%`），但牺牲 `2021/2024/2025/2026` 的一部分收益
- 当前判断:
  - PB3 rolling gating 在真实账户口径下有效，不只是 hindsight trade-level 归因幻觉
  - 规则更像“避开 AMV 老化/混沌期继续开新仓”，不是卖出规则；已有持仓仍按原 6td 生命周期退出
  - 下一步如果继续推进，应做 walk-forward/阈值敏感性，确认 `aged`、`amp>2.5`、`neg>=3` 不是单点拟合

### [AMV] 板块宽度诊断

- 目标: 板块宽度是否能解释 P3 在 AMV 牛市不同阶段的收益差异
- 脚本: `scripts/amv_sector_breadth_diagnostic.py`
- 产物: `reports/amv_sector_breadth_diagnostic.json`
- 行业数据: 东方财富 86 个行业, 5552 只股票
- 关键发现:
  - Bull 宽度 61% vs Non-bull 39%，区分度良好
  - 但 P3 在**窄基牛市**里反而单笔收益更高（narrow 1.51% vs broad 1.24%）
  - 板块 OK 过滤在全历史上净亏（误杀 3 笔大赢家，含 +53K）
  - 2026 年 P3 亏损真相: 4 月选了水泥、地产、航天等非科技行业假突破，**一支半导体都没选到**
  - 行业中性化不是答案；P3 需要的是"板块顺风过滤"而非"板块中性化"

### [AMV] P3 退出逻辑（止损/延长持有）what-if

- 脚本: `scripts/amv_exit_logic_whatif.py`
- 目标: 模拟早期止损 + 延长持有对 P3 的改善效果
- 早期止损（d2 cum_ret < -3% → 卖出）:
  - 24 笔触发, 原始合计 -337K, 提前止损约省 **+166K**
  - 2 笔误杀大赢家（均因 d0 正 d1 跌 → 可加"d1 也为负"修复）
- 延长持有（d6 cum_ret > 10% & near high → extend 3d trailing -5%）:
  - 111 笔触发（40%，太多），平均额外仅 +0.2%, 51 笔亏更多
  - 结论: 延长持有当前信号信噪比不够，暂不推进
- 合并近似: +250K（主要来自止损）

## 2026-05-21

### [AMV] Python rolling cohort multi-stat diagnostics

- 背景:
  - 用户指出 Python rolling cohort NAV 不应只看平均收益或最终 NAV
  - 需要同时观察中位数、分位数、胜率、6 条 cohort sleeve 的最差/中位/最好路径，以及粗成本调整
- 新增脚本:
  - `scripts/amv_signal_cohort_stats.py`
  - 输入: 一个或多个 `signal.parquet` / signal artifact
  - 输出:
    - `strict_top3`
    - `refill_top10`
    - daily return distribution: mean/median/std/p05/p10/p25/p75/p90/p95/win rate
    - pick return distribution
    - event-time cohort NAV
    - event-time cohort cost-adjusted NAV
    - 6 条 sleeve 的 worst/median/best NAV
    - dense calendar zero-return cohort NAV
    - yearly diagnostics
- 重算对象:
  - `reference_p2`
  - `p3`
  - `pb3_cp1`
  - `trend_label_top`: `trend_p1_k0p5_pb1_cp0_rv1`
  - `trend_rust_top`: `trend_p1_k1_pb1_cp0_rv1`
- 补导出:
  - `artifacts/amv_static_sleeve_signals/20260521_122805_reference_p2_k0p5_b0_c0_r0/`
  - `artifacts/amv_static_sleeve_signals/20260521_122807_trend_p1_k0p5_pb1_cp0_rv1/`
- 报告:
  - `reports/amv_signal_cohort_stats_main_pullback_trend.json`
- Canvas:
  - `amv-signal-cohort-stats.canvas.tsx`
- 关键结果（`refill_top10`, event-time cohort）:
  - `reference_p2`: NAV `+98.55%`, median daily `+0.282%`, p10 `-3.330%`, win `55.1%`, sleeve worst `+12.22%`
  - `p3`: NAV `+102.93%`, median daily `+0.305%`, p10 `-3.182%`, win `55.3%`, sleeve worst `+31.75%`
  - `PB3/CP1`: NAV `+186.48%`, median daily `+0.512%`, p10 `-5.194%`, win `53.8%`, sleeve worst `+119.61%`
  - `trend label top`: NAV `+229.26%`, median daily `+0.443%`, p10 `-2.376%`, win `57.4%`, sleeve worst `+157.84%`
  - `trend Rust top`: NAV `+215.39%`, median daily `+0.441%`, p10 `-2.533%`, win `57.6%`, sleeve worst `+145.90%`
- 粗成本调整（`0.35%` 单轮往返成本，`refill_top10`）:
  - `reference_p2`: `+45.16%`
  - `p3`: `+48.36%`
  - `PB3/CP1`: `+109.64%`
  - `trend label top`: `+141.10%`
  - `trend Rust top`: `+130.90%`
- dense calendar zero-return 对比（`refill_top10`）:
  - `reference_p2`: `+89.98%`
  - `p3`: `+105.19%`
  - `PB3/CP1`: `+176.58%`
  - `trend label top`: `+227.85%`
  - `trend Rust top`: `+219.09%`
- 当前判断:
  - Python rolling cohort 作为信号质量诊断可以更细: trend-only top 的统计分布确实最强，PB3 次之
  - 但这仍不能替代 Rust account NAV；trend-only 的真实承接问题已由 no-repeat、duplicate、资金暴露和成本诊断解释
  - P3/reference 的 event-time cohort 不惊艳，但 P3 static Rust 强，说明 P3 的价值主要在 static cadence 与低换手真实组合，而不是 rolling cohort 信号质量
- 校验:
  - `uv run python -m py_compile scripts/amv_signal_cohort_stats.py`

### [AMV] P3 static cadence sensitivity

- 背景:
  - 用户对 P3 static “单日起点路径”是否存在侥幸提出质疑
  - Python executable 平均值只能证明信号日整体 edge，不足以证明某条 static cadence 不是运气
- 重导出:
  - `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves candidate_p3_k0p5_b0_c0_r0 --top-n 10`
  - artifact: `artifacts/amv_static_sleeve_signals/20260521_113554_candidate_p3_k0p5_b0_c0_r0/`
- 诊断:
  - 报告: `reports/amv_p3_static_cadence_sensitivity.json`
  - 口径: `T+1 open -> D+7 close`, strict Top3, 跳过执行日开盘涨停，不补位，不含成本
  - all signal-day Top3 分布:
    - mean `+0.859%`
    - median `+0.299%`
    - p10 `-3.253%`
    - p90 `+5.246%`
    - worst `-17.494%`
    - positive day share `55.27%`
  - 7 个 static cadence offset:
    - no-cost 最差 `+260.74%`
    - no-cost 中位 `+285.60%`
    - no-cost 最好 `+297.79%`
    - MaxDD 均约 `8.93%`
  - 粗略扣 `0.35%` 单轮往返成本:
    - 最差约 `+162.19%`
    - 中位约 `+181.25%`
    - 最好约 `+190.14%`
- 当前判断:
  - P3 static 的收益不是单一起点侥幸；不同起始 offset 都有明显正收益
  - 但 Rust net `+201.69%` 仍应作为主结论，因为它包含真实成本、手数、现金和执行细节

### [AMV] bt-amv-topn duplicate lot diagnostic

- 背景:
  - Python executable label 允许每天重复选择同一只强票，Rust rolling 原口径禁止对已持有 code 重复买入
  - 为验证 trend-only 是否被 no-repeat 语义压制，给 `bt-amv-topn` 增加配置开关，默认保持旧行为
- 代码变更:
  - 新增配置字段: `allow_duplicate_positions`
  - 默认值: `false`
  - 开启后同一 code 可生成多个 `Position` lot；每个 lot 独立记录 entry、持有天数、成本和退出
  - 原有配置不写该字段时结果语义不变
- 新增配置:
  - `backtest-engine/crates/amv-topn/config_6td_rolling21_refill_top10_duplicate_no_stop.toml`
- 验证:
  - `cargo check -p bt-amv-topn`
  - 通过
- duplicate 诊断结果:
  - 报告: `reports/amv_duplicate_position_diagnostic_summary.json`
  - `trend P1/K1/PB1/CP0/RV1`: no-repeat `+60.17%` -> duplicate `+106.50%`, gross `+95.13%` -> `+147.17%`, MaxDD `11.58%` -> `13.33%`
  - `PB3/CP1/RV0`: no-repeat `+99.62%` -> duplicate `+102.93%`, gross `+130.78%` -> `+134.12%`, MaxDD `20.70%` -> `20.46%`
  - duplicate 口径下 trend 最大同 code 并发 lot 数 `7`，PB3 为 `5`
- 当前判断:
  - no-repeat 持仓语义解释了 trend-only 很大一部分 Python -> Rust 损耗
  - duplicate lot 口径让 trend-only 明显改善，但也引入更高单票集中度，暂作为诊断/可选进攻口径，不替代默认分散持仓口径
  - PB3 对 duplicate 不敏感，说明它本身更兼容“不同 code rolling”的真实组合语义

### [AMV] Trend-only vs PB3 Python/Rust daily trade overlap

- 背景:
  - 用户指出 trend-only full top 的涨停/高开污染很低，理论上 Rust 损耗不应明显大于 pullback
  - 为确认原因，重跑最小可复现集并比较 Python executable-label 每日 Top3 与 Rust rolling21 refill 实际买入逐日/逐票 overlap
- 重跑:
  - `trend_p1_k1_pb1_cp0_rv1`
  - `pullback_p0_k0_pb3_cp1_rv0`
  - 配置: `config_6td_rolling21_refill_top10_no_stop.toml`
- 产物:
  - overlap 报告: `reports/amv_trend_vs_pb3_signal_trade_overlap.json`
  - trend artifact: `artifacts/amv_static_sleeve_signals/20260521_090943_trend_p1_k1_pb1_cp0_rv1/`
  - PB3 artifact: `artifacts/amv_static_sleeve_signals/20260521_090945_pullback_p0_k0_pb3_cp1_rv0/`
- 核心对比:
  - `trend P1/K1/PB1/CP0/RV1`:
    - Rust rolling refill net `+60.17%`, gross `+95.13%`
    - Python bull Top3 6td open-to-exit mean `+1.35%`
    - Rust 实际买入 mean: gross `+1.01%`, net `+0.65%`
    - 实买 rank 1-3 占比 `49.7%`，rank 4-10 占比 `50.3%`
    - 每日 exact Top3 完全一致仅 `14.9%`，日均 Top3 overlap `49.1%`
    - Python Top3 未买原因: 已持有 `824`，执行日涨停 `2`，其他 `5`
  - `PB3/CP1/RV0`:
    - Rust rolling refill net `+99.62%`, gross `+130.78%`
    - Python bull Top3 6td open-to-exit mean `+1.33%`
    - Rust 实际买入 mean: gross `+1.39%`, net `+1.03%`
    - 实买 rank 1-3 占比 `70.0%`，rank 4-10 占比 `30.0%`
    - 每日 exact Top3 完全一致 `37.1%`，日均 Top3 overlap `70.0%`
    - Python Top3 未买原因: 已持有 `410`，执行日涨停 `1`，其他 `79`
- 当前判断:
  - trend-only 的 Python label 没有兑现，不是因为涨停/高开污染；执行日涨停只解释极少数缺口
  - 真正关键是 trend-only Top3 重复度高，滚动真实账户不能对已持有股票重复买入，导致约一半实际交易来自 rank 4-10 补位
  - PB3/CP1/RV0 的 Top3 更“可滚动”，真实买入保留了约 `70%` 的 Python Top3，因此 Python label 到 Rust 的转换更好
  - 后续如果继续 trend-only，应新增“持仓去重后的 executable label”或在 Python grid 阶段加入 no-repeat rolling 仓位约束，否则 label 会继续高估真实 rolling 表现

## 2026-05-20

### [AMV] Trend-only executable grid full scan on Mac

- 背景:
  - 白天 Windows 设备上 trend-only full grid 耗时过长中止，未产生有效 summary
  - 晚上在当前 Mac 上复跑全量 `factor + pullback full + yearly`
- 运行:
  - `PYTHONUNBUFFERED=1 uv run python scripts/amv_executable_trend_filter_grid.py --ranker-set all --grid-preset full --horizons 6 --top-n 3 --top-k 30`
  - Ranker sets: `factor,pullback,yearly`
  - Rankers: `755`
  - Trend-only pool: `260,164` 行，`516` 天，`2,282` 只股票
  - 使用本地 ST 缓存 `261` 只
  - 耗时约 `244s`
- 产物:
  - `artifacts/amv_executable_trend_filter_grid/20260520_212625/summary.json`
  - `compact.csv`
  - `daily.csv`
- 全部候选基准:
  - trend-only all candidates: exec NAV `+70.40%`, MaxDD `21.20%`, close limit-up day share `94.0%`
- Full grid `skip_close_limit_refill_top3` Top 结果:
  - `P1/K0.5/PB1/CP0/RV1`: exec NAV `+256.94%`, MaxDD `4.51%`, CTC NAV `+207.10%`, rank q95 `3`
  - `P1/K1/PB1/CP0/RV1`: exec NAV `+243.77%`, MaxDD `4.38%`, CTC NAV `+201.91%`, rank q95 `3`
  - `P1/K1/PB0.5/CP0/RV1`: exec NAV `+235.41%`, MaxDD `5.97%`, CTC NAV `+180.69%`, rank q95 `3`
  - `P1/K1/PB1/CP0.5/RV1`: exec NAV `+229.19%`, MaxDD `6.23%`, CTC NAV `+190.40%`, rank q95 `3`
  - `P2/K0.5/PB2/CP0/RV1`: exec NAV `+216.01%`, MaxDD `4.11%`, CTC NAV `+171.89%`, rank q95 `3`
  - focused 旧最强 `P1/K0/PB1/CP0/RV0.5`: exec NAV `+213.74%`, MaxDD `4.26%`, CTC NAV `+161.58%`
- Full grid `original_top3` 观察:
  - 最强 tradeoff 也集中在 `P + K + PB + RV`，例如 `P1/K1/PB0.5/CP0.5/RV1` exec NAV `+250.21%`, MaxDD `7.41%`
  - close 涨停/高开污染显著低于 momentum 类候选，Top refill 多数 rank q95 `3`
- 当前判断:
  - Mac full run 已跑通，并显著抬高 trend-only label 侧上限
  - 新的最强族群不是 focused 里的 `P1/K0/PB1/CP0/RV0.5`，而是 `P1 + K0.5/1 + PB0.5/1 + RV1`，即趋势池内的轻价格位置 + K 线确认 + 回调 + 风险约束
  - 但 focused 旧最强在 Rust 修正后仍只有 static refill `+112.54%`、rolling refill `+41.47%`，说明 Python label -> Rust 真实组合损耗很大
  - 下一步如果继续 trend-only，只应挑 full grid Top 2-3 个新候选导出 Rust 验证；在 Rust 兑现前，trend-only 仍不进入当前主线候选

#### Full top 新候选 Rust 验证

- 背景:
  - 用户指出: trend-only full top label 与普通 pullback 都在 `200%+`，且 trend-only top 涨停污染不重；如果 label -> Rust 损耗相近，trend-only 不应明显弱于普通 pullback
  - 因此补充导出 full grid Top 新候选并接真实 Rust 回测
- 新增 sleeve:
  - `trend_p1_k0p5_pb1_cp0_rv1`
  - `trend_p1_k1_pb1_cp0_rv1`
  - `trend_p2_k0p5_pb2_cp0_rv1`
- 信号导出:
  - `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves trend_p1_k0p5_pb1_cp0_rv1,trend_p1_k1_pb1_cp0_rv1,trend_p2_k0p5_pb2_cp0_rv1 --top-n 10`
  - 输出:
    - `artifacts/amv_static_sleeve_signals/20260520_214039_trend_p1_k0p5_pb1_cp0_rv1/`
    - `artifacts/amv_static_sleeve_signals/20260520_214040_trend_p1_k1_pb1_cp0_rv1/`
    - `artifacts/amv_static_sleeve_signals/20260520_214040_trend_p2_k0p5_pb2_cp0_rv1/`
  - 每个信号约 `776` 个执行日、Top10 shift 后 `7,759~7,760` 行
- Rust 回测:
  - 每个 sleeve 跑 `static strict Top3`、`static refill Top10`、`rolling21 strict Top3`、`rolling21 refill Top10`
  - 汇总报告: `reports/amv_trend_filter_full_top_rust_summary.json`
  - Canvas: `reports/canvases/amv-trend-full-rust-conversion.canvas.tsx`
- 核心结果:
  - `trend P1/K0.5/PB1/CP0/RV1`:
    - static strict/refill: net `+85.85%`, gross `+126.15%`, MaxDD `43.24%`
    - rolling21 refill: net `+56.97%`, gross `+90.55%`, MaxDD `11.88%`
  - `trend P1/K1/PB1/CP0/RV1`:
    - static strict/refill: net `+76.15%`, gross `+116.26%`, MaxDD `42.12%`
    - rolling21 refill: net `+60.17%`, gross `+95.13%`, MaxDD `11.58%`
  - `trend P2/K0.5/PB2/CP0/RV1`:
    - static strict/refill: net `+123.58%`, gross `+171.28%`, MaxDD `38.27%`
    - rolling21 refill: net `+56.73%`, gross `+91.57%`, MaxDD `10.36%`
- 与普通 pullback 对照:
  - `PB1/CP0/RV0` static strict: net `+190.28%`, gross `+230.07%`, MaxDD `43.43%`
  - `PB3/CP1/RV0` rolling21 refill: net `+99.62%`, gross `+130.78%`, MaxDD `20.70%`
  - trend full top 的成本损耗并非唯一原因；rolling gross 已从 label `+216%~257%` 掉到 `+90%~95%`
- 当前解释:
  - trend-only full top 的涨停污染确实很低，不应归因于涨停污染
  - 真正差异在于 Python executable label 到 Rust 真实组合的 gross edge 转换率:
    - Python label 是每日 Top3 cohort 的重叠净值诊断
    - Rust static 会受持仓占用影响，每次买满后 6td 内无法继续捕捉新信号
    - Rust rolling 更接近 label，但仍有 no-repeat、真实资金、手数、成本、重复代码补位等约束
  - 普通 pullback 的 label 年度结构更稳: `PB1/CP0/RV0` / `PB3/CP1/RV0` label 侧 `stable_positive_years = 5`，且 2025/2026 edge 为正
  - trend-only full top label 侧虽然总 NAV 更高，但 stable years 只有 `3~4`，且 2025/2026 edge 多为负；转成真实 rolling 后 gross edge 保留不足
  - 结论: trend-only 不是“废”，但当前更像 label 侧平滑强、真实组合承接弱；暂不替代 `P3 static` 或 `PB3 rolling`，后续若继续应做 Python daily cohort 与 Rust actual trades 的逐日/逐票损耗归因

### [AMV] Trend-only focused candidates Rust verification

- 背景:
  - Python focused scan 中 `trend-only + P/PB/RV` 组合显示很强的 executable label 表现
  - 用户要求导出信号并接真实 `bt-amv-topn` 验证
- 代码:
  - `scripts/amv_static_sleeve_signal_export.py` 新增 trend-only sleeve:
    - `trend_p1_k0_pb1_cp0_rv0p5`
    - `trend_p1_k0p5_pb1_cp0_rv0p5`
    - `trend_p3_k0p5_pb2_cp1_rv0p5`
  - trend sleeve 在原 AMV bull + liquidity 过滤之外额外应用 `close > YL / WL > YL`
  - `YL = mean(MA14, MA28, MA57, MA114)`
  - `WL = EMA(EMA(close, 10), 10)`
- 信号导出:
  - 命令: `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves trend_p1_k0_pb1_cp0_rv0p5,trend_p1_k0p5_pb1_cp0_rv0p5,trend_p3_k0p5_pb2_cp1_rv0p5 --top-n 10`
  - 输出:
    - `artifacts/amv_static_sleeve_signals/20260520_152712_trend_p1_k0_pb1_cp0_rv0p5/`
    - `artifacts/amv_static_sleeve_signals/20260520_152714_trend_p1_k0p5_pb1_cp0_rv0p5/`
    - `artifacts/amv_static_sleeve_signals/20260520_152717_trend_p3_k0p5_pb2_cp1_rv0p5/`
  - 每个 signal: `776` 个执行日，Top10 shift 后 `7760` 条信号
- Rust 回测:
  - 配置:
    - `config_6td_static_strict_top3_no_stop.toml`
    - `config_6td_static_refill_top10_no_stop.toml`
    - `config_6td_rolling21_strict_top3_no_stop.toml`
    - `config_6td_rolling21_refill_top10_no_stop.toml`
  - 运行 `3` 个 sleeve × `4` 个配置，共 `12` 个 release 回测，全部完成
  - 汇总报告: `reports/amv_trend_filter_rust_backtest_summary.json`
- 核心结果:
  - `trend P1/K0/PB1/CP0/RV0.5`:
    - static strict: net `+64.18%`, MaxDD `40.68%`, trades `276`
    - static refill: net `+64.46%`, MaxDD `41.06%`, trades `276`
    - rolling21 strict: net `+24.42%`, MaxDD `7.06%`, trades `740`
    - rolling21 refill: net `+35.02%`, MaxDD `14.10%`, trades `1533`
  - `trend P1/K0.5/PB1/CP0/RV0.5`:
    - static strict/refill: net `+60.74%`, MaxDD `43.78%`
    - rolling21 strict: net `+23.63%`, MaxDD `8.48%`
    - rolling21 refill: net `+29.94%`, MaxDD `12.58%`
  - `trend P3/K0.5/PB2/CP1/RV0.5`:
    - static strict: net `-3.44%`, MaxDD `30.26%`
    - static refill: net `+0.61%`, MaxDD `26.50%`
    - rolling21 strict: net `+26.89%`, MaxDD `9.86%`
    - rolling21 refill: net `+25.54%`, MaxDD `15.69%`
- 当前判断:
  - Python focused label 的 `+195% ~ +214%` 没有在真实 Rust 组合中兑现
  - `trend-only + P/PB/RV` 至少当前 Top3/Top10 真实口径不如 `P3/K0.5/R0` 静态 strict，也不如 `PB3/CP1/RV0` rolling refill
  - 该路线暂时降级为“label 线索未兑现”，不进入下一轮 allocation/gating 候选
  - 若后续继续，只适合做差异归因: Python executable label 与 Rust static/rolling 之间的损耗来自执行日 regime 阻塞、真实资金/持仓约束、重复代码补位还是年度集中性

#### 导出 score scope 修正后复跑

- 问题定位:
  - Python trend-only grid 是先过滤到 `AMV bull + liquidity + trend_only` 候选池，再在候选池内部做组件 rank
  - 初版 `scripts/amv_static_sleeve_signal_export.py` 是先在全市场上算组件 rank，再过滤候选池
  - 对 `P/PB/RV` 这种 rank 组合，rank 母体不同会改变 TopN
- 修正:
  - 对 `trend_*` sleeve，导出脚本先物理过滤 `_is_signal_candidate`
  - 再在该候选 DataFrame 内计算 `pullback_combo_score_expr`
  - 从而使 `_score_component()` 的 `rank().over("date") / pl.len().over("date")` 与 Python grid 口径一致
- 复跑对象:
  - `trend_p1_k0_pb1_cp0_rv0p5`
  - 新信号: `artifacts/amv_static_sleeve_signals/20260520_153918_trend_p1_k0_pb1_cp0_rv0p5/`
  - 汇总报告: `reports/amv_trend_filter_corrected_export_rust_summary.json`
- 修正后 Rust 结果:
  - static strict: net `+112.54%`, MaxDD `38.27%`, trades `276`
  - static refill: net `+112.54%`, MaxDD `38.27%`, trades `276`
  - rolling21 strict: net `+26.85%`, MaxDD `8.83%`, trades `842`
  - rolling21 refill: net `+41.47%`, MaxDD `11.66%`, trades `1575`
- 对比:
  - 修正前 static refill `+64.46%` -> 修正后 `+112.54%`
  - 修正前 rolling21 refill `+35.02%` -> 修正后 `+41.47%`
  - Python focused label 仍为 exec NAV `+213.74%`, MaxDD `4.26%`
- 当前判断:
  - rank 母体差异确实解释了一部分损耗，尤其 static Top3 显著抬升
  - 但修正后仍没有接近 Python label，也没有超过当前 `P3/K0.5/R0` 静态 strict 或 `PB3/CP1/RV0` rolling refill
  - trend-only 方向继续保持降级；如果继续，只做损耗归因，不作为 allocation/gating 新候选

### [AMV] Trend-only executable grid focused scan

- 新增脚本: `scripts/amv_executable_trend_filter_grid.py`
- 目标:
  - 将 `trend_only = close > YL / WL > YL` 从 B1 对照实验提升为可复用候选池过滤器
  - 在该池内扫描 early factor、pullback focused grid、yearly P/K/R/P/K/M rankers
  - 仍使用 executable-aware v2: 主指标 `D+1 open -> D+7 close`，辅助 `D close -> D+6 close`
- Smoke test:
  - `uv run python scripts/amv_executable_trend_filter_grid.py --ranker-set all --grid-preset focused --horizons 6 --top-n 3 --top-k 10 --max-rankers 20`
  - 产物: `artifacts/amv_executable_trend_filter_grid/20260520_145439/summary.json`
  - 通过，前 20 个 ranker 中 `factor_atr_14_pct_asc` refill exec NAV `+144.59%`, MaxDD `5.58%`
- Focused run:
  - `uv run python scripts/amv_executable_trend_filter_grid.py --ranker-set all --grid-preset focused --horizons 6 --top-n 3 --top-k 30`
  - 产物: `artifacts/amv_executable_trend_filter_grid/20260520_145545/summary.json`
  - compact: `artifacts/amv_executable_trend_filter_grid/20260520_145545/compact.csv`
  - daily: `artifacts/amv_executable_trend_filter_grid/20260520_145545/daily.csv`
- 候选池:
  - trend-only rows `260,164`
  - 信号日 `516`
  - 覆盖股票 `2,282`
  - 全部候选平均 exec NAV `+70.40%`, MaxDD `21.20%`
  - 全部候选 close limit-up day share `94.0%`，所以必须优先看 `skip_close_limit_refill_top3`
- Focused Top refill:
  - `pullback P1/K0/PB1/CP0/RV0.5`: exec NAV `+213.74%`, MaxDD `4.26%`, ctc NAV `+161.58%`, rank q95 `3`
  - `pullback P1/K0.5/PB1/CP0/RV0.5`: exec NAV `+204.40%`, MaxDD `4.87%`, ctc NAV `+166.65%`, rank q95 `3`
  - `pullback P3/K0.5/PB2/CP1/RV0.5`: exec NAV `+195.26%`, MaxDD `4.78%`, ctc NAV `+163.43%`, rank q95 `3`
  - `pullback P2/K0.5/PB2/CP0/RV0.5`: exec NAV `+192.52%`, MaxDD `5.33%`, ctc NAV `+142.43%`, rank q95 `3`
  - `yearly P1/K0.5/R1.5`: exec NAV `+194.00%`, MaxDD `7.11%`, ctc NAV `+162.50%`, rank q95 `3`
- Full run:
  - 尝试 `uv run python scripts/amv_executable_trend_filter_grid.py --ranker-set all --grid-preset full --horizons 6 --top-n 3 --top-k 30`
  - 当前 Windows 设备前置构建/评估耗时过长，已中止，未产生有效 summary
  - full grid 留待 Mac 夜跑或改成分组分批执行
- 当前判断:
  - `trend-only` 不是单独策略，而是比 AMV bull 更窄、比 classic B1 更宽的趋势候选池
  - 在该候选池中，强候选主要表现为 `P + PB + RV` 的轻回调低风险结构，而不是纯 `P3/K0.5`
  - Top focused 候选满足低污染、rank q95 浅、回撤低，值得导出信号接 `bt-amv-topn` strict/refill 验证
  - 由于这仍是 Python executable label grid，暂不能替代 Rust 真实组合结论

### [B1] Original base executable-aware lab

- 新增脚本: `scripts/b1_executable_base_lab.py`
- 运行:
  - `uv run python scripts/b1_executable_base_lab.py --horizons 6 --top-n 3 --top-k 12`
  - 产物: `artifacts/b1_executable_base_lab/20260520_142833/summary.json`
- 原始 B1 base 定义:
  - `close_adj > YL`
  - `WL > YL`
  - `J <= 13`
  - `WL = EMA(EMA(close, 10), 10)`
  - `YL = mean(MA14, MA28, MA57, MA114)`
  - `J = 3K - 2D`, `K/D` 来自 9 日 RSV 的 `ewm_mean(com=2)`
- 口径:
  - 复用 AMV executable-aware v2
  - 样本 `2021-01-01` 到 `2026-05-10`
  - AMV bull + ST 剔除 + liquidity 过滤后评估
  - 主指标 `D+1 open -> D+7 close`，辅助 `D close -> D+6 close`
- 候选规模:
  - B1 base rows `28,260`
  - 可满足 Top3 的信号日 `489`
  - 覆盖股票 `1,914`
  - 平均每天候选约 `57.7`
- 核心结果:
  - 全部 B1 候选平均: exec NAV `+53.28%`, MaxDD `20.66%`, close limit-up day share `2.86%`
  - `J` 越低 Top3: exec NAV `+36.71%`, MaxDD `24.62%`
  - `J` 越高 Top3: exec NAV `+64.31%`, MaxDD `20.95%`
  - `B1 base + ma_bias_20 asc`: exec NAV `+81.26%`, MaxDD `28.92%`
  - `B1 base + disp_bias_20 asc`: exec NAV `+76.28%`, MaxDD `26.59%`
  - `B1 base + PB3/CP1`: exec NAV `+80.87%`, MaxDD `27.19%`, close limit-up day share `0.0%`
  - `B1 base + PB2/CP0.5`: exec NAV `+89.21%`, MaxDD `26.07%`, close limit-up day share `0.0%`
  - `B1 base + P3/K0.5`: exec NAV `+43.80%`, MaxDD `10.36%`
- 当前判断:
  - 原始 B1 三条件在 AMV bull 里确实更像“趋势中的回调候选池”，不是现版复杂 B1 回测失败所暗示的完全无效信号
  - 但原始 `J <= 13` 更像候选过滤，不适合作为 Top3 排序主轴；在 `J <= 13` 内继续追求更低 J 反而弱于全部候选平均
  - 在 B1 base 池内，pullback 排序显著优于 P3/K0.5 突破排序，说明它与当前 pure pullback 发现同源
  - 由于 `B1 base + PB2/CP0.5` 仍明显弱于独立 pullback executable/grid 与 Rust rolling pullback，短期不应把 B1 提升为新主线；更合理的定位是 pullback 机制的旁证和后续 ablation 线索

#### 去掉 `J <= 13` 的趋势池对照

- 脚本参数:
  - 新增 `--base-mode classic|trend_only`
  - `classic`: `close > YL / WL > YL / J <= threshold`
  - `trend_only`: `close > YL / WL > YL`
- 运行:
  - `uv run python scripts/b1_executable_base_lab.py --base-mode trend_only --horizons 6 --top-n 3 --top-k 12`
  - 产物: `artifacts/b1_executable_base_lab/20260520_143434/summary.json`
- 候选规模:
  - trend-only rows `260,164`
  - 可满足 Top3 的信号日 `516`
  - 覆盖股票 `2,282`
  - 平均每天候选约 `504.2`
- 核心结果:
  - 全部趋势候选平均: exec NAV `+70.40%`, MaxDD `21.20%`
  - `J` 越低 Top3: exec NAV `+42.66%`, MaxDD `26.09%`
  - `J` 越高 Top3: 原始 Top3 exec NAV `+26.80%` 且 close limit-up day share `56.6%`; 跳过 close 涨停补位后 exec NAV `+90.57%`, MaxDD `23.66%`
  - `close_to_high_20d asc`: 原始 Top3 exec NAV `-3.45%` 且 close limit-up day share `87.2%`; 跳过 close 涨停补位后 exec NAV `+74.82%`, MaxDD `9.04%`
  - `ma_bias_20 asc`: refill exec NAV `+88.98%`, MaxDD `35.34%`
  - `disp_bias_20 asc`: refill exec NAV `+102.61%`, MaxDD `33.11%`
  - `PB3/CP1`: refill exec NAV `+89.34%`, MaxDD `31.91%`
  - `PB2/CP0.5`: refill exec NAV `+89.22%`, MaxDD `34.10%`
  - `P3/K0.5`: 原始 Top3 exec NAV `+116.46%` 但 close limit-up day share `28.9%`; 跳过 close 涨停补位后 exec NAV `+118.74%`, MaxDD `7.27%`
- 当前判断:
  - `J <= 13` 不是单纯“增益过滤”，它会把趋势池强行压到低 J 回调形态，因此明显压制 `P3/K0.5` 这类突破排序
  - 去掉 J 后，trend-only + `P3/K0.5` 的 refill 诊断很强，且回撤很低，是值得后续导出信号接 Rust 的候选
  - 但 trend-only 候选池非常大，部分原始 Top3 排序存在严重 close 涨停污染；后续如果验证，只应使用 strict/refill 可成交口径，不应看原始 close-to-close 或未过滤 Top3
  - B1 方向可以拆成两条: `classic B1 = 趋势中的低 J 回调`，`trend-only = 趋势过滤器/候选池`；二者不应混为同一个策略假设

### [AMV] Rolling pullback 代表袖子选择

- 新增:
  - 决策报告: `reports/amv_pullback_representative_choice.json`
  - Pairwise 归因:
    - `reports/amv_pullback_pb3_vs_pb2_rolling_refill_attribution.json`
    - `reports/amv_pullback_pb3_vs_pb1_rolling_refill_attribution.json`
  - 更新 Canvas: `reports/canvases/amv-executable-sleeve-rust-complement.canvas.tsx`
- 候选:
  - `PB3/CP1/RV0 rolling21 refill`
  - `PB2/CP0.5/RV0 rolling21 refill`
  - `PB1/CP0/RV0 rolling21 refill`
- rolling21 refill 核心结果:
  - `PB3/CP1/RV0`: net `+99.62%`, MaxDD `20.70%`, trades `1650`, 2025 `+29.73%`, 2026 `+15.15%`
  - `PB2/CP0.5/RV0`: net `+96.06%`, MaxDD `22.74%`, trades `1647`, 2025 `+32.21%`, 2026 `+15.25%`
  - `PB1/CP0/RV0`: net `+89.41%`, MaxDD `21.28%`, trades `1522`, 2025 `+21.11%`, 2026 `+12.32%`
- Pairwise 归因:
  - `PB3/CP1` vs `PB2/CP0.5`:
    - total return delta `+3.57pp`
    - MaxDD delta `-2.03pp`
    - cost delta `+4,941`
    - exact overlap `1455` 笔，daily return corr `0.988`
    - PB3 全周期小胜，PB2 在 `2025/2026` 年度略胜
  - `PB3/CP1` vs `PB1`:
    - total return delta `+10.22pp`
    - MaxDD delta `-0.57pp`
    - cost delta `+12,351`
    - exact overlap `524` 笔，daily return corr `0.921`
    - PB3 明显更活跃，收益提升主要来自 unique trades
- 当前判断:
  - 后续 allocation/gating 暂以 `PB3/CP1/RV0 rolling21 refill` 作为唯一 pullback 代表
  - 选择理由是全周期收益最高、回撤低于 PB2、strict 口径与 PB2 近似持平，且与 P3 static 保持低相关
  - `PB2/CP0.5/RV0` 不废弃，作为 forward challenger 监控；但由于与 PB3 相关性和交易重合过高，不应和 PB3 同时堆叠
  - `PB1/CP0/RV0` 更简单且涨停过滤为 `0`，但 rolling return 与 2025/2026 修复能力均弱于 PB3，暂不作为代表

### [Workflow] Generic backtest trade attribution script

- 新增: `scripts/backtest_trade_attribution.py`
- 目标:
  - 将此前手写的 P3 vs Ref 交易归因逻辑沉淀成可复用脚本
  - 保持为 `bt-amv-topn` artifact 通用，不写死 AMV/P3/Ref
- 输入:
  - `--left-backtest <dir>`
  - `--right-backtest <dir>`
  - `--left-label / --right-label`
  - `--out <json>`
- 依赖文件:
  - `report.json`
  - `trades.csv`
  - `daily_equity.csv`
- 输出内容:
  - summary metrics 与 right-left delta
  - cost drag
  - yearly/monthly equity returns
  - yearly/monthly realized trade PnL
  - exact trade overlap 与 code overlap
  - left/right unique winners/losers
  - common trades delta
  - daily return correlation
- 校验:
  - `uv run python -m py_compile scripts/backtest_trade_attribution.py`
  - 用 `Ref P2/K0.5/R0` vs `P3/K0.5/R0` 跑通，核心结果对齐既有归因:
    - total return delta `+30.89pp`
    - MaxDD delta `-1.78pp`
    - exact overlap `244`

### [Workflow] AMV trade attribution project skill

- 新增项目级 Skill: `.agents/skills/amv-trade-attribution/SKILL.md`
- 适用场景:
  - `bt-amv-topn` artifact 对比
  - AMV 策略交易归因
  - P3 vs Ref 换票机制解释
  - pullback sleeve 互补性 / cost drag 分析
- 约定:
  - 可复用计算逻辑放在 repo `scripts/`，通过 `uv run python ...` 执行
  - Skill 只沉淀工作流、归因口径、文档更新规则和后续脚本入口命名
  - 已补首个通用入口 `scripts/backtest_trade_attribution.py`
  - 后续若高频复用，再补 `scripts/amv_explain_signal_swaps.py`、`scripts/amv_strategy_correlation.py`

### [AMV] P3/K0.5 vs Ref P2/K0.5 主基线替换归因

- 新增:
  - tracked canvas: `reports/canvases/amv-p3-vs-ref-trade-attribution.canvas.tsx`
  - 聚合数据: `reports/amv_p3_vs_ref_trade_attribution.json`
  - 换票特征分解: `reports/amv_p3_ref_swap_feature_explain.json`
- 对比对象:
  - Ref: `artifacts/amv_static_sleeve_signals/20260520_092047_reference_p2_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092131_677`
  - P3: `artifacts/amv_static_sleeve_signals/20260520_092049_candidate_p3_k0p5_b0_c0_r0/backtests/6td_static_strict_top3_no_stop_20260520_092208_801`
  - 口径一致: `bt-amv-topn`, static strict Top3, `T+1 open`, `6td`, no-stop
- 核心指标:
  - Ref: total return `+170.80%`, MaxDD `15.30%`, win `51.09%`, total costs `291,119`
  - P3: total return `+201.69%`, MaxDD `13.52%`, win `52.55%`, total costs `295,846`
  - P3 相比 Ref: total return `+30.89pp`, MaxDD 改善 `1.78pp`, win rate `+1.46pp`
  - P3 开盘涨停过滤更多: Ref `48` vs P3 `60`，说明改善不是靠更少交易约束
- 年度拆解:
  - P3 优于 Ref: `2022 +0.75pp`, `2023 +5.20pp`, `2025 +1.32pp`, `2026 +8.03pp`
  - P3 弱于 Ref: `2021 -1.76pp`, `2024 -3.23pp`
  - 2026 改善最关键: Ref `-8.80%` -> P3 `-0.77%`
- 月度归因:
  - P3 优势月份:
    - `2026-01`: equity delta `+9.23pp`, trade PnL delta `+129,226`
    - `2023-04`: equity delta `+9.83pp`, trade PnL delta `+73,186`
    - `2021-04`: equity delta `+6.67pp`, trade PnL delta `+34,768`
  - P3 弱势月份:
    - `2021-01`: equity delta `-11.27pp`, trade PnL delta `-59,508`
    - `2023-03`: equity delta `-3.78pp`, trade PnL delta `-28,697`
    - `2024-11`: equity delta `-1.75pp`, trade PnL delta `-56,447`
- 交易重合与换票:
  - 两者各 `274` 笔交易
  - exact overlap (`entry_date + code`) 为 `244` 笔，重合率 `89.05%`
  - code overlap `193 / 209`，代码重合率 `92.34%`
  - P3-only `30` 笔合计 PnL `+170,899`
  - Ref-only `30` 笔合计 PnL `+29,737`
  - 边际换票贡献约 `+141,161`，占总 PnL 差额 `+154,447` 的大部分
  - overlap 交易 P3 也略优: common P3 PnL `+837,561` vs common Ref PnL `+824,275`
- 关键换票:
  - P3-only 大赢家:
    - `sz.003035` `2026-01-16 -> 2026-01-26`: PnL `+92,771`, return `+18.19%`
    - `sh.600667` `2023-03-31 -> 2023-04-11`: PnL `+53,322`, return `+20.76%`
    - `sh.601127` `2024-11-05 -> 2024-11-13`: PnL `+50,207`, return `+11.57%`
  - 被替换的 Ref 交易中有赢家也有亏损:
    - 避开 `sh.688789` `2026-01-16 -> 2026-01-26`: PnL `-35,846`
    - 避开 `sz.300919` `2023-03-31 -> 2023-04-11`: PnL `-19,365`
    - 但错过 `sz.000559` `2024-10-31 -> 2024-11-08`: PnL `+83,373`
    - 也错过 `sz.002271` `2021-01-04 -> 2021-01-12`: PnL `+51,608`
- 换票机制解释:
  - `P2/K0.5` 的 P-block 占比约 `80%`，`P3/K0.5` 提到约 `85.7%`；变化不大，但 Top3 边缘足以改变排序。
  - `2026-01-15` 信号日: P3 将 `sz.003035` 从 Ref rank `5` 推到 rank `3`，替代 `sh.688789`。`sz.003035` 的 P-block `0.954` 高于 `sh.688789` 的 `0.944`，但 K-block `0.812` 低于 `0.858`；P3 接受较差 K 线，换取更强“贴近高点/收在高点”状态。
  - `2023-03-30` 信号日: P3 将 `sh.600667` 从 Ref rank `5` 推到 rank `3`，替代 `sz.300919`。`sh.600667` 的 P-block `0.982` 明显高于 `sz.300919` 的 `0.943`，但 K-block `0.744` 低于 `0.953`。
  - 失败样本也符合这个机制: `2020-12-31` P3 用 P-block 更高的 `sh.600570` 挤掉 K-block 更高的 `sz.002271`，后者随后成为 Ref-only 大赢家。
  - 因此 P3 可解释为“更纯的高位突破延续”版本，而不是新 alpha；它在趋势延续窗口收益更好，在 K 线质量更重要的分化窗口可能错过强票。
- 当前判断:
  - P3 是强替换候选，但优势主要来自少量边际换票，而非整体交易池重写
  - 2026 改善非常关键，但高度集中在 `2026-01` 的换票质量；当前已能解释 `sz.003035` 替代 `sh.688789` 的排序机制，但是否可重复仍需更多换票样本和 forward 监控
  - 若后续监控确认“高 P-block / 弱 K-block”边际换票在新样本中不显著恶化，则可以把 `P3/K0.5/R0` 提升为新 reference

### [AMV] Executable/Pullback sleeves Rust TopN 回测

- 背景:
  - Mac full pullback grid 复跑没有推翻 focused grid 结论后，开始把候选接入真实 `bt-amv-topn`
  - 目标是同时验证静态 Top3 sleeve 与 rolling cohort 是否复现 Python executable-aware 结论
- 代码/配置:
  - `scripts/amv_static_sleeve_signal_export.py` 新增 6 个候选 sleeve:
    - `reference_p2_k0p5_b0_c0_r0`
    - `candidate_p3_k0p5_b0_c0_r0`
    - `pullback_p0_k0_pb1_cp0_rv0`
    - `pullback_p0_k0_pb3_cp1_rv0`
    - `pullback_p0_k0_pb2_cp0p5_rv0`
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`
  - `scripts/amv_bull_pool_export_signals.py` 补充保留 pullback/risk 字段: `ma_bias_20 / disp_bias_20 / atr_14_pct / panic_vol_ratio_20d / intraday_pos`
  - 新增 `bt-amv-topn` 6td 配置:
    - `config_6td_static_strict_top3_no_stop.toml`
    - `config_6td_static_refill_top10_no_stop.toml`
    - `config_6td_rolling21_strict_top3_no_stop.toml`
    - `config_6td_rolling21_refill_top10_no_stop.toml`
- 信号导出:
  - 命令: `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves reference_p2_k0p5_b0_c0_r0,candidate_p3_k0p5_b0_c0_r0,pullback_p0_k0_pb1_cp0_rv0,pullback_p0_k0_pb3_cp1_rv0,pullback_p0_k0_pb2_cp0p5_rv0,pullback_p2_k0_pb0_cp0p5_rv0p5 --top-n 10`
  - 输出根目录: `artifacts/amv_static_sleeve_signals/20260520_092047_*` 到 `20260520_092057_*`
  - 每个 signal 约 `813` 个执行日；strict Top3 约 `1732-1734` 行，refill Top10 约 `5778-5779` 行
- Rust 回测:
  - 运行 `6` 个 sleeve × `4` 个配置，共 `24` 个 `bt-amv-topn` release 回测，全部完成
  - 静态 strict Top3:
    - `candidate_p3_k0p5_b0_c0_r0`: net `+201.69%`, MaxDD `13.52%`, win `52.6%`, trades `274`
    - `pullback_p0_k0_pb1_cp0_rv0`: net `+190.28%`, MaxDD `43.43%`, win `52.9%`, trades `276`
    - `reference_p2_k0p5_b0_c0_r0`: net `+170.80%`, MaxDD `15.30%`, win `51.1%`, trades `274`
    - `pullback_p0_k0_pb3_cp1_rv0`: net `+152.84%`, MaxDD `41.85%`, win `51.6%`, trades `275`
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`: net `+72.34%`, MaxDD `48.92%`, win `47.6%`, trades `271`
    - `pullback_p0_k0_pb2_cp0p5_rv0`: net `+66.25%`, MaxDD `45.69%`, win `50.5%`, trades `275`
  - 静态 refill Top10:
    - `pullback_p0_k0_pb1_cp0_rv0`: net `+190.28%`, MaxDD `43.43%`
    - `candidate_p3_k0p5_b0_c0_r0`: net `+158.26%`, MaxDD `15.58%`
    - `pullback_p0_k0_pb3_cp1_rv0`: net `+109.99%`, MaxDD `41.85%`
    - `reference_p2_k0p5_b0_c0_r0`: net `+101.03%`, MaxDD `17.48%`
    - `pullback_p0_k0_pb2_cp0p5_rv0`: net `+60.66%`, MaxDD `45.69%`
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`: net `+33.73%`, MaxDD `54.75%`
  - rolling21 strict Top3:
    - `pullback_p0_k0_pb3_cp1_rv0`: net `+83.06%`, MaxDD `17.05%`, trades `1255`
    - `pullback_p0_k0_pb2_cp0p5_rv0`: net `+82.37%`, MaxDD `16.95%`, trades `1205`
    - `pullback_p0_k0_pb1_cp0_rv0`: net `+43.91%`, MaxDD `11.39%`, trades `646`
    - `candidate_p3_k0p5_b0_c0_r0`: net `+30.65%`, MaxDD `8.00%`, trades `1332`
    - `reference_p2_k0p5_b0_c0_r0`: net `+23.93%`, MaxDD `9.31%`, trades `1335`
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`: net `+11.87%`, MaxDD `12.40%`, trades `1246`
  - rolling21 refill Top10:
    - `pullback_p0_k0_pb3_cp1_rv0`: net `+99.62%`, MaxDD `20.70%`, trades `1650`
    - `pullback_p0_k0_pb2_cp0p5_rv0`: net `+96.06%`, MaxDD `22.74%`, trades `1647`
    - `pullback_p0_k0_pb1_cp0_rv0`: net `+89.41%`, MaxDD `21.28%`, trades `1522`
    - `candidate_p3_k0p5_b0_c0_r0`: net `+22.19%`, MaxDD `11.93%`, trades `1644`
    - `reference_p2_k0p5_b0_c0_r0`: net `+21.44%`, MaxDD `10.89%`, trades `1643`
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`: net `+21.11%`, MaxDD `14.03%`, trades `1640`
- 过滤观察:
  - pure pullback 的开盘涨停过滤极轻: B1 为 `0`，PB3/CP1 与 PB2/CP0.5 仅 `1-3`
  - P/K reference 与 P3 在 strict Top3 下仍有 `48/60` 次开盘涨停过滤，refill Top10 下分别为 `135/163`
  - `pullback_p2_k0_pb0_cp0p5_rv0p5` 在 Rust 中没有复现 Python refill 低回撤优势，涨停过滤也偏高，暂时降级
- 当前判断:
  - `candidate_p3_k0p5_b0_c0_r0` 是新的主基线替换候选: 静态 strict 比 reference 多 `+30.89pct`，且 MaxDD 更低
  - pure pullback 的真实交易 edge 成立，尤其 rolling21 口径下明显强于 P/K reference；但回撤仍在 `20%+`，不应直接替换主策略
  - refill Top10 不是无条件提升: 它不仅补开盘涨停，也会在已有持仓重复代码时补低 rank 新票；静态口径下可能稀释高质量 Top3
  - 下一步优先做 `P3/K0.5/R0` 年度归因和 2025/2026 对照，再决定是否正式替换 `manual_p2_k0p5_r0_6td`

### [AMV] Pullback sleeve 命名收口与复跑

- 背景:
  - 为避免和既有 TDX `B1/B2/B3` 策略混淆，pullback 组合统一收口为 `PB/CP/RV` 命名
  - 已删除旧命名 pullback artifact 目录，并移除导出脚本中的旧 ID 兼容分支
- 命名:
  - `PB`: pullback bias = `ma_bias_20 + disp_bias_20`
  - `CP`: close-position pullback = `KSFT + intraday_pos`
  - `RV`: risk/volatility = `atr_14_pct + panic_vol_ratio_20d`
- 代码:
  - `scripts/amv_static_sleeve_signal_export.py` 仅保留 `PB/CP/RV` pullback ID
  - `scripts/amv_executable_pullback_grid.py` 后续生成 ID 格式: `p/k/pb/cp/rv`
- 信号导出:
  - 命令: `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves pullback_p0_k0_pb1_cp0_rv0,pullback_p0_k0_pb3_cp1_rv0,pullback_p0_k0_pb2_cp0p5_rv0,pullback_p2_k0_pb0_cp0p5_rv0p5 --top-n 10`
  - 输出:
    - `artifacts/amv_static_sleeve_signals/20260520_105222_pullback_p0_k0_pb1_cp0_rv0/`
    - `artifacts/amv_static_sleeve_signals/20260520_105223_pullback_p0_k0_pb3_cp1_rv0/`
    - `artifacts/amv_static_sleeve_signals/20260520_105225_pullback_p0_k0_pb2_cp0p5_rv0/`
    - `artifacts/amv_static_sleeve_signals/20260520_105228_pullback_p2_k0_pb0_cp0p5_rv0p5/`
  - ST 使用本地缓存 `258` 只；每个 signal 仍约 `813` 个执行日
- Rust 复跑:
  - 运行 `4` 个 sleeve × `4` 个配置，共 `16` 个 `bt-amv-topn` release 回测，全部完成
  - `PB1/CP0/RV0`:
    - static strict/refill: net `+190.28%`, MaxDD `43.43%`
    - rolling strict: net `+43.91%`, MaxDD `11.39%`
    - rolling refill: net `+89.41%`, MaxDD `21.28%`
  - `PB3/CP1/RV0`:
    - static strict: net `+152.84%`, MaxDD `41.85%`
    - static refill: net `+109.99%`, MaxDD `41.85%`
    - rolling strict: net `+83.06%`, MaxDD `17.05%`
    - rolling refill: net `+99.62%`, MaxDD `20.70%`
  - `PB2/CP0.5/RV0`:
    - static strict: net `+66.25%`, MaxDD `45.69%`
    - static refill: net `+60.66%`, MaxDD `45.69%`
    - rolling strict: net `+82.37%`, MaxDD `16.95%`
    - rolling refill: net `+96.06%`, MaxDD `22.74%`
  - `P2/CP0.5/RV0.5`:
    - static strict: net `+72.34%`, MaxDD `48.92%`
    - static refill: net `+33.73%`, MaxDD `54.75%`
    - rolling strict: net `+11.87%`, MaxDD `12.40%`
    - rolling refill: net `+21.11%`, MaxDD `14.03%`
- 当前判断:
  - `PB/CP/RV` 已成为后续唯一 pullback 命名
  - `PB3/CP1/RV0` 与 `PB2/CP0.5/RV0` 继续作为 rolling pullback 代表候选

### [AMV] Executable sleeves 年度互补性 Canvas

- 新增:
  - tracked canvas: `reports/canvases/amv-executable-sleeve-rust-complement.canvas.tsx`
  - 聚合数据: `reports/amv_executable_sleeve_rust_yearly.json`
- 口径:
  - 从 24 个 `bt-amv-topn` 回测的 `daily_equity.csv` 聚合年度收益
  - 年度收益 = 当年最后权益 / 上一年最后权益 - 1
  - 互补性 = daily return 相关性 + 2025/2026 年度错位
- 关键发现:
  - `Ref P2/K0.5 static strict` vs `P3/K0.5 static strict` 日收益相关 `0.916`，说明 P3 更多是主线替换候选，不是互补 sleeve
  - P/K 主线 vs rolling pullback 日收益相关明显较低:
    - `P3 static` vs `PB3/CP1/RV0 rolling refill`: `0.255`
    - `P3 static` vs `PB2/CP0.5/RV0 rolling refill`: `0.243`
    - `P3 static` vs `PB1/CP0/RV0 rolling refill`: `0.214`
  - rolling pullback 家族内部高度重叠:
    - `PB3/CP1/RV0` vs `PB2/CP0.5/RV0`: `0.988`
    - `PB3/CP1/RV0` vs `PB1/CP0/RV0`: `0.921`
  - 2026 互补非常明显:
    - `Ref static`: `-8.80%`
    - `P3 static`: `-0.77%`
    - `PB3/CP1/RV0 rolling`: `+15.15%`
    - `PB2/CP0.5/RV0 rolling`: `+15.25%`
    - `PB1/CP0/RV0 rolling`: `+12.32%`
- 简单 50/50 daily rebalance 诊断:
  - `P3 static + PB3/CP1/RV0 rolling` 50/50: total `+152.38%`, MaxDD `9.99%`, 2026 `+7.11%`
  - `P3 static + PB3/CP1/RV0 rolling` 80/20: total `+183.05%`, MaxDD `11.60%`, 2026 `+2.35%`
  - `P3 static + PB3/CP1/RV0 rolling` 70/30: total `+173.03%`, MaxDD `10.81%`, 2026 `+3.93%`
- 当前判断:
  - 互补性成立，但不是“多个 pullback 之间互补”；真正互补的是 `P/K 主线` vs `一个 rolling pullback sleeve`
  - pullback 更适合做 allocation/gating 的补充仓位，而不是替换静态 Top3 主策略

## 2026-05-19

### [AMV] Executable-aware pullback full grid Mac 复跑

- 背景:
  - 白天另一台设备上 full grid `618` 个 ranker 运行过久且没有落产物
  - 晚上在当前 Mac 上复跑 `--grid-preset full`，验证 full grid 是否能正常跑完
- 运行:
  - `PYTHONUNBUFFERED=1 uv run python scripts/amv_executable_pullback_grid.py --grid-preset full`
  - 识别 ranker 数量: `618`
  - 构建数据集: `632,608` 行，`2,321` 只股票，日期 `2021-04-20 -> 2026-04-24`
  - 实时 ST 源失败后自动使用本地缓存，共 `263` 只 ST；运行完成，无 Traceback
  - 耗时约 `295s`
- 产物:
  - `artifacts/amv_executable_pullback_grid/20260519_213813/summary.json`
- `original_top3` 关键结果:
  - `pullback_p0_k0_pb1_cp0_rv0`: exec NAV `+245.37%`, MaxDD `23.29%`, CTC NAV `+124.33%`, close 涨停覆盖 `0.5%`
  - `pullback_p0_k0_pb3_cp1_rv0`: exec NAV `+215.37%`, MaxDD `20.29%`, CTC NAV `+110.09%`, close 涨停覆盖 `0.0%`
  - `pullback_p0_k0_pb2_cp0p5_rv0`: exec NAV `+210.37%`, MaxDD `21.89%`, CTC NAV `+98.82%`, close 涨停覆盖 `0.0%`
  - `pullback_p1_k0_pb3_cp1_rv0`: exec NAV `+179.37%`, MaxDD `18.70%`, CTC NAV `+105.12%`, close 涨停覆盖 `0.0%`
  - `candidate_p3_k0p5_b0_c0_r0`: exec NAV `+160.14%`, MaxDD `5.58%`, CTC NAV `+278.93%`, close 涨停覆盖 `20.2%`
  - `reference_p2_k0p5_b0_c0_r0`: exec NAV `+152.21%`, MaxDD `5.27%`, CTC NAV `+248.54%`, close 涨停覆盖 `16.8%`
- `skip_close_limit_refill_top3` 关键结果:
  - `pullback_p0_k0_pb1_cp0_rv0`: exec NAV `+242.58%`, MaxDD `23.29%`, rank q95 `3`
  - `pullback_p0_k0_pb3_cp1_rv0`: exec NAV `+215.37%`, MaxDD `20.29%`, rank q95 `3`
  - `pullback_p0_k0_pb2_cp0p5_rv0`: exec NAV `+210.37%`, MaxDD `21.89%`, rank q95 `3`
  - `pullback_p2_k0_pb0_cp0p5_rv0p5`: exec NAV `+167.31%`, MaxDD `5.73%`, CTC NAV `+147.75%`, rank q95 `4`
  - `pullback_p2_k0_pb0_cp0p5_rv1`: exec NAV `+154.31%`, MaxDD `5.36%`, CTC NAV `+131.02%`, rank q95 `3`
- 当前判断:
  - full grid 跑通，且没有推翻 focused grid 结论
  - 纯 pullback 最强仍是 `PB1/CP0/RV0` 与 `PB3/CP1/RV0`，说明 `ma_bias_20 + disp_bias_20` 回调线索稳定
  - full grid 新增值得关注的折中候选:
    - `pullback_p0_k0_pb2_cp0p5_rv0`: 纯 pullback 中介于 B1 与 PB3/CP1 之间，收益高但回撤仍深
    - `pullback_p2_k0_pb0_cp0p5_rv0p5`: refill 场景下收益 `+167.31%`、MaxDD `5.73%`，更像低回撤混合候选
  - 下一步 Rust 静态 sleeve 候选可从 4 个扩展到 5-6 个，但优先级仍是先验证 `PB1/CP0/RV0`、`PB3/CP1/RV0`、`P2/CP0.5/RV0.5` 这三类 archetype

### [AMV] Executable-aware pullback combo grid v1

- 背景:
  - 全因子 executable 扫描发现低污染强候选集中在 `ma_bias_20_asc / disp_bias_20_asc / KSFT_asc`
  - 用户同意把这些新发现的回调因子纳入新一轮组合权重探索
- 新增:
  - `scripts/amv_executable_pullback_grid.py`
- 因子组定义:
  - `P`: `price_pos_20d` 高 + `close_to_high_20d` 低
  - `K`: `KLEN` 低 + `KMID2` 高
  - `B`: `ma_bias_20` 低 + `disp_bias_20` 低
  - `C`: `KSFT` 低 + `intraday_pos` 低
  - `R`: `atr_14_pct` 低 + `panic_vol_ratio_20d` 低
  - 本轮不加入 `M` 动量
- 执行过程:
  - 首次 full grid `618` 个 ranker 运行过久且没有落产物，已停止
  - 脚本改为默认 `--grid-preset focused`，focused grid `164` 个 ranker；保留 `--grid-preset full` 供后续局部展开
  - `uv run python scripts/amv_executable_pullback_grid.py`
- 产物:
  - smoke: `artifacts/amv_executable_pullback_grid/20260519_153907/summary.json`
  - focused full: `artifacts/amv_executable_pullback_grid/20260519_160017/summary.json`
  - compact: `artifacts/amv_executable_pullback_grid/20260519_160017/compact.csv`
- Canvas:
  - tracked: `reports/canvases/amv-executable-pullback-grid.canvas.tsx`
  - renderable: `amv-executable-pullback-grid.canvas.tsx`
- `original_top3` 核心结果:
  - `pullback_p0_k0_pb1_cp0_rv0`: exec NAV `+245.37%`, MaxDD `23.29%`, close-to-close NAV `+124.33%`, close 涨停覆盖 `0.5%`
  - `pullback_p0_k0_pb3_cp1_rv0`: exec NAV `+215.37%`, MaxDD `20.29%`, close-to-close NAV `+110.09%`, close 涨停覆盖 `0.0%`
  - `pullback_p1_k0_pb3_cp1_rv0`: exec NAV `+179.37%`, MaxDD `18.70%`, close-to-close NAV `+105.12%`, close 涨停覆盖 `0.0%`
  - `pullback_p0_k0_pb2_cp1_rv0`: exec NAV `+176.51%`, MaxDD `19.39%`, close-to-close NAV `+90.94%`, close 涨停覆盖 `0.0%`
  - `candidate_p3_k0p5_b0_c0_r0`: exec NAV `+160.14%`, MaxDD `5.58%`, close-to-close NAV `+278.93%`, close 涨停覆盖 `20.2%`
  - `reference_p2_k0p5_b0_c0_r0`: exec NAV `+152.21%`, MaxDD `5.27%`, close-to-close NAV `+248.54%`, close 涨停覆盖 `16.8%`
- `skip_close_limit_refill_top3` 核心结果:
  - `pullback_p0_k0_pb1_cp0_rv0`: exec NAV `+242.58%`, MaxDD `23.29%`, close-to-close NAV `+124.83%`, rank q95 `3`
  - `pullback_p0_k0_pb3_cp1_rv0`: exec NAV `+215.37%`, MaxDD `20.29%`, close-to-close NAV `+110.09%`, rank q95 `3`
  - `pullback_p1_k0_pb3_cp1_rv0`: exec NAV `+179.37%`, MaxDD `18.70%`, close-to-close NAV `+105.12%`, rank q95 `3`
  - `pullback_p0_k0_pb2_cp1_rv0`: exec NAV `+176.51%`, MaxDD `19.39%`, close-to-close NAV `+90.94%`, rank q95 `3`
  - `pullback_p3_k0_pb0_cp1_rv0`: exec NAV `+157.93%`, MaxDD `11.05%`, close-to-close NAV `+150.89%`, rank q95 `5`
  - `pullback_p2_k0_pb0_cp1_rv0`: exec NAV `+155.87%`, MaxDD `10.14%`, close-to-close NAV `+149.99%`, rank q95 `5`
- 当前判断:
  - pullback sleeve 成立，且不是涨停污染产物；最强 `PB1/CP0/RV0` 基本等价于 `ma_bias_20 + disp_bias_20` 回调组合
  - 纯 pullback 收益高于现有 P/K reference，但回撤也明显更深，定位更像独立 sleeve 或 attack/counter-trend sleeve，不应直接替换主基线
  - `P/K + CP` 组合收益略低但回撤更可控，值得作为 Rust 静态 sleeve 候选
  - 下一步应挑 2-4 个候选导出 Rust 回测:
    - `pullback_p0_k0_pb1_cp0_rv0`
    - `pullback_p0_k0_pb3_cp1_rv0`
    - `pullback_p2_k0_pb0_cp1_rv0`
    - `pullback_p3_k0_pb0_cp1_rv0`

### [AMV] Executable-aware 早期全因子扫描

- 背景:
  - 用户指出当前 executable-aware 权重网格只覆盖少数解释因子，需要把早期全因子扫描也按新口径重跑
  - 新口径沿用:
    - 主评估: `D+1 open -> D+7 close`
    - 辅助诊断: `D close -> D+6 close`
    - 归因: close 涨停、`T+1 open` 涨停、`T+1` 高开、跳过 close 涨停补位
- 新增:
  - `scripts/amv_executable_factor_scan.py`
- 修正:
  - smoke test 发现 `scripts/amv_executable_weight_grid.py` 的单因子 `descending=False` 方向在通用评估函数里没有正确反映到排序 score
  - 已改为显式构造 `score = factor` 或 `score = -factor` 后统一按 `score` 降序排序
  - 该问题影响单因子 asc/desc 对照；此前 P/K/R、P/K/M 组合结论不受影响，因为组合 score 已是 higher-is-better
- 范围:
  - 早期 `RANKERS + COMBO_RANKERS`，共 `47` 个 ranker
  - AMV bull LF2, Top3, `horizon = 6`
- 校验:
  - `uv run python -m py_compile scripts/amv_executable_weight_grid.py scripts/amv_executable_factor_scan.py`
  - `uv run python scripts/amv_executable_factor_scan.py --max-rankers 8`
  - `uv run python scripts/amv_executable_factor_scan.py`
- 产物:
  - full: `artifacts/amv_executable_factor_scan/20260519_151529/summary.json`
  - compact: `artifacts/amv_executable_factor_scan/20260519_151529/compact.csv`
- Canvas:
  - tracked: `reports/canvases/amv-executable-factor-scan.canvas.tsx`
  - renderable: `amv-executable-factor-scan.canvas.tsx`
- `original_top3` 核心结果:
  - `ret_5d_desc`: exec NAV `+264.07%`, MaxDD `47.11%`, close-to-close NAV `+365.43%`, close 涨停覆盖 `69.0%` 天, `T+1` 高开覆盖 `14.1%` 天
  - `ma_bias_20_asc`: exec NAV `+231.03%`, MaxDD `23.19%`, close-to-close NAV `+132.19%`, close 涨停覆盖 `0.7%` 天
  - `disp_bias_20_asc`: exec NAV `+187.54%`, MaxDD `22.68%`, close-to-close NAV `+94.73%`, close 涨停覆盖 `0.4%` 天
  - `KSFT_asc`: exec NAV `+178.36%`, MaxDD `24.50%`, close-to-close NAV `+2.59%`, close 涨停覆盖 `0.0%` 天
  - `combo_high_pos_kmid2_lowrisk`: exec NAV `+157.02%`, MaxDD `11.44%`, close-to-close NAV `+152.69%`, close 涨停覆盖 `10.3%` 天
  - `combo_high_pos_kbar_confirm`: exec NAV `+140.77%`, MaxDD `5.07%`, close-to-close NAV `+191.25%`, close 涨停覆盖 `13.7%` 天
  - `far_from_high_20d`: exec NAV `+126.72%`, MaxDD `26.88%`, close-to-close NAV `+62.51%`, close 涨停覆盖 `2.0%` 天
  - `ret_20d_asc`: exec NAV `+114.83%`, MaxDD `25.73%`, close-to-close NAV `+73.80%`, close 涨停覆盖 `1.4%` 天
- `skip_close_limit_refill_top3` 核心结果:
  - `ma_bias_20_asc`: exec NAV `+227.14%`, close-to-close NAV `+131.40%`, rank q95 `3`
  - `disp_bias_20_asc`: exec NAV `+188.64%`, close-to-close NAV `+95.87%`, rank q95 `3`
  - `KSFT_asc`: exec NAV `+178.36%`, close-to-close NAV `+2.59%`, rank q95 `3`
  - `ret_5d_desc`: exec NAV `+165.93%`, close-to-close NAV `+0.25%`, rank q95 `7`
  - `combo_high_pos_kmid2_lowrisk`: exec NAV `+138.53%`, close-to-close NAV `+119.77%`, rank q95 `3`
  - `combo_high_pos_kbar_confirm`: exec NAV `+121.80%`, close-to-close NAV `+101.77%`, rank q95 `4`
- 当前判断:
  - 全因子 executable 扫描发现一条与既有 high-pos/kbar 主线不同的强线索: “20 日均线/成本线下回归 + 收盘偏低/回调后反弹”
  - `ma_bias_20_asc / disp_bias_20_asc / KSFT_asc` 的 close 涨停污染极低，且补位后几乎不衰减，优先级高于继续堆 P/K/M 动量权重
  - `ret_5d_desc` 仍有很强可执行收益，但高回撤和高涨停覆盖说明它更像 attack/event archetype，不适合作为稳态主基线
  - 下一步应把低污染强因子纳入新的可解释组合网格，例如 `P/K + pullback` 或 `pullback + lowrisk`，并优先做 Rust 静态 sleeve 验证

### [AMV] Executable-aware 因子/权重网格 v2 首轮

- 背景:
  - 用户决定重开早期 AMV 多头宽池因子、因子组合与权重网格探索
  - 新原则:
    - 主评估: `T+1 open -> horizon close`
    - 辅助诊断: `close-to-close`
    - 强制归因: signal close 涨停、`T+1 open` 涨停、`T+1 open` 高开、跳过 close 涨停后补位表现
  - signal day close 涨停不直接删除；主指标用可执行入场价自然惩罚它
- 新增:
  - `scripts/amv_executable_weight_grid.py`
- 口径:
  - 复用 `amv_yearly_weight_grid.py` 的 `90` 个 ranker:
    - 当前 reference
    - 单因子
    - 旧 P/K/R 网格
    - P/K/M 网格
  - 默认 `horizon = 6`
  - signal day `D close` 排序
  - executable 主标签: `D+1 open -> D+7 close`
  - close-to-close 辅助标签: `D close -> D+6 close`
  - 同时输出:
    - `original_top3`
    - `skip_close_limit_refill_top3`
- 校验:
  - `uv run python -m py_compile scripts/amv_executable_weight_grid.py`
  - `uv run python scripts/amv_executable_weight_grid.py --max-rankers 8`
  - `uv run python scripts/amv_executable_weight_grid.py`
  - `ReadLints scripts/amv_executable_weight_grid.py`: 无问题
- 产物:
  - smoke: `artifacts/amv_executable_weight_grid/20260519_144637/summary.json`
  - full: `artifacts/amv_executable_weight_grid/20260519_144938/summary.json`
  - compact: `artifacts/amv_executable_weight_grid/20260519_144938/compact.csv`
- Canvas:
  - tracked: `reports/canvases/amv-executable-weight-grid-v2.canvas.tsx`
  - renderable: `amv-executable-weight-grid-v2.canvas.tsx`
- `original_top3` 关键结果:
  - `single_ret_5d`: exec NAV `+264.07%`, close-to-close NAV `+365.43%`, close 涨停覆盖 `69.0%` 天, `T+1` 高开覆盖 `14.1%` 天, MaxDD `47.11%`
  - `grid_high_pos_kbar_p3_k0p5_r0`: exec NAV `+160.14%`, close-to-close NAV `+278.93%`, close 涨停覆盖 `20.2%` 天, `T+1` 高开覆盖 `9.0%` 天
  - `ref_p2_k0p5_r0`: exec NAV `+152.21%`, close-to-close NAV `+248.54%`, close 涨停覆盖 `16.8%` 天, `T+1` 高开覆盖 `7.6%` 天
  - `grid_pkm_p1_k0p5_m1`: exec NAV `+32.87%`, close-to-close NAV `+1054.55%`, close 涨停覆盖 `73.5%` 天, `T+1` 高开覆盖 `28.0%` 天
  - `grid_pkm_p2_k0p5_m0p5`: exec NAV `+29.14%`, close-to-close NAV `+768.74%`, close 涨停覆盖 `59.7%` 天, `T+1` 高开覆盖 `26.4%` 天
  - `grid_pkm_p3_k1_m2`: exec NAV `+31.62%`, close-to-close NAV `+1045.97%`, close 涨停覆盖 `76.4%` 天, `T+1` 高开覆盖 `28.2%` 天
- `skip_close_limit_refill_top3` 关键结果:
  - `grid_high_pos_kbar_p3_k0p5_r0`: exec NAV `+151.90%`, close-to-close NAV `+117.44%`, `T+1` 高开覆盖 `0.2%` 天
  - `ref_p2_k0p5_r0`: exec NAV `+114.12%`, close-to-close NAV `+90.33%`, `T+1` 高开覆盖 `0.2%` 天
  - `single_ret_5d`: exec NAV `+165.93%`, close-to-close NAV `+0.25%`, close 涨停覆盖 `0%`, MaxDD `30.93%`
  - `grid_pkm_p1_k0p5_m1`: exec NAV `+90.64%`, close-to-close NAV `+55.71%`
  - `grid_pkm_p2_k0p5_m0p5`: exec NAV `+66.44%`, close-to-close NAV `+53.72%`
  - `grid_pkm_p3_k1_m2`: exec NAV `+57.72%`, close-to-close NAV `+31.32%`
- 当前判断:
  - 可执行口径 v2 重新支持早期 AMV 高位 + K 线确认结构；旧 reference 附近权重仍强
  - `P3/K0.5/R0` 在标签侧可执行口径略强于当前 `P2/K0.5/R0`，但仍需 Rust `T+1 open` 真实回测确认，不能直接替换主基线
  - 带动量的 P/K/M 原始 close-to-close 爆炸，但 executable 仅 `+29% ~ +33%`，污染非常重
  - 跳过 close 涨停并补位后，P/K/M 有所恢复但仍弱于无动量高位 + K 线结构
  - `single_ret_5d` 即使 executable NAV 高，也有高涨停覆盖、高回撤和明显事件票属性，应作为污染/动量 archetype 观察，不作为直接主策略候选

### [AMV] Python close 涨停跳过并顺位补位 rolling NAV 诊断

- 背景:
  - 用户提出在标签侧 rolling NAV 中，如果当日 Top3 有 close 涨停票，则跳过并顺位选择下一个候选，直到补满 Top3
  - 目标是区分“P/K/M 标签侧高收益是否只来自原始 Top3 涨停污染”与“非涨停后排候选是否仍有 alpha”
- 新增:
  - `scripts/amv_limit_refill_rolling_nav.py`
- 诊断口径:
  - AMV bull LF2 池，`mv_min = 100`, `amount_ma20_min = 5e7`
  - close-to-close 标签侧 rolling NAV，默认 `5d / 6d`
  - 每日原始 Top3、剔除原始 Top3 涨停但不补位、跳过 close 涨停并顺位补满 Top3 三种场景
  - close 涨停按主板 `10%`、创业板/科创板 `20%`，容差 `0.001`
  - `max_scan_rank = 0`，即从全候选池顺位补位
  - 本诊断仍不是真实交易口径: 不含 `T+1 open`、成本、手数、no-repeat、资金约束
- 运行:
  - `uv run python scripts/amv_limit_refill_rolling_nav.py`
  - 产物: `artifacts/amv_limit_refill_rolling_nav/20260519_143120/summary.json`
- 6d 核心结果:
  - `manual_p2_k0p5_r0`: 原始 Top3 NAV `+248.54%` -> 跳过涨停补位 `+90.33%`, MaxDD `5.71%`
  - `pkm_p1_k0p5_m1`: 原始 Top3 NAV `+1054.55%` -> 跳过涨停补位 `+55.71%`, MaxDD `19.10%`
  - `pkm_p2_k0p5_m0p5`: 原始 Top3 NAV `+768.74%` -> 跳过涨停补位 `+53.72%`, MaxDD `11.64%`
  - `pkm_p3_k1_m2`: 原始 Top3 NAV `+1045.97%` -> 跳过涨停补位 `+31.32%`, MaxDD `21.35%`
- close 涨停污染:
  - manual 原始 Top3 close 涨停 `135 / 1665` 行，覆盖 `94 / 555` 天
  - `P1/K0.5/M1`: `828 / 1665` 行，覆盖 `408 / 555` 天
  - `P2/K0.5/M0.5`: `610 / 1665` 行，覆盖 `332 / 555` 天
  - `P3/K1/M2`: `870 / 1665` 行，覆盖 `424 / 555` 天
- 补位深度:
  - manual 补位 rank 均值 `2.74`, q95 `4`, max `258`
  - `P1/K0.5/M1` 补位 rank 均值 `5.52`, q95 `16`, max `140`
  - `P2/K0.5/M0.5` 补位 rank 均值 `4.75`, q95 `14`, max `244`
  - `P3/K1/M2` 补位 rank 均值 `5.90`, q95 `17`, max `170`
- 当前判断:
  - P/K/M 原始 rolling NAV 的 `+700%` 到 `+1000%` 级收益主要依赖 close 涨停不可买样本
  - 顺位补位后 P/K/M 仍有正的标签侧 alpha，但只剩几十个百分点，且 6d 不再超过 manual
  - 这说明“动量袖子不是完全没信号”，但强信号集中在不可成交极端票；后排非涨停候选不足以支撑主策略替代
  - 如果后续接 Rust，应把“候选池深度 + 跳过不可买后补位”作为 execution policy 单独测试，但不应再把原始 Python rolling NAV 当作可交易上限

### [Backtest Engine] A 股买入手数口径修正

- 背景:
  - rolling cohort 真实组合会把单票目标仓位压到 `1/18` 或 `1/21`
  - 原 `bt_core::round_to_lot` 使用全局 `LOT_SIZE = 300`，会过度压低小仓位可买入数量，并增加高价股跳单
- 修正:
  - 普通 A 股买入按 `100` 股整数倍向下取整
  - 科创板 `sh.688*` 买入少于 `200` 股则跳过；达到 `200` 股后允许按 `1` 股递增
  - `amv-topn / b1 / b3 / rotation / renko` 的买入逻辑统一传入代码，由 `bt_core::round_to_lot(code, shares)` 判定
- 影响:
  - 这是交易口径修正，不是策略调参
  - 既有 Rust 指标，包括 `manual_p2_k0p5_r0_6td = +168.01%` 与 P/K/M 扫描结果，后续应在新手数口径下重新确认
  - rolling cohort 回测应基于本口径继续推进

### [AMV] 手数修正后 manual_p2_k0p5_r0_6td 校准重跑

- 背景:
  - 买入手数从全局 `300` 股修正为普通 A 股 `100` 股、科创板 `200+1` 后，需要先校准当前主基线
  - 同批重跑了 manual 与三组 P/K/M 静态 sleeve 信号，便于后续滚动持仓回测使用一致信号时间戳
- 信号导出:
  - `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves manual_p2_k0p5_r0,pkm_p1_k0p5_m1,pkm_p2_k0p5_m0p5,pkm_p3_k1_m2 --end-date 2026-05-10`
  - 新产物:
    - `artifacts/amv_static_sleeve_signals/20260519_085446_manual_p2_k0p5_r0/signal.meta.json`
    - `artifacts/amv_static_sleeve_signals/20260519_085448_pkm_p1_k0p5_m1/signal.meta.json`
    - `artifacts/amv_static_sleeve_signals/20260519_085450_pkm_p2_k0p5_m0p5/signal.meta.json`
    - `artifacts/amv_static_sleeve_signals/20260519_085452_pkm_p3_k1_m2/signal.meta.json`
- Rust 回测:
  - `uv run python scripts/amv_topn_backtest.py artifacts/amv_static_sleeve_signals/20260519_085446_manual_p2_k0p5_r0 --config backtest-engine/crates/amv-topn/config_6td_no_stop.toml`
  - 报告: `artifacts/amv_static_sleeve_signals/20260519_085446_manual_p2_k0p5_r0/backtests/6td_no_stop_20260519_085501_573/report.json`
- 结果:
  - 新手数口径: net `+170.80%`, gross `+229.03%`, MaxDD `15.30%`, trades `274`, win rate `51.09%`
  - 旧手数口径: net `+168.01%`, MaxDD `14.97%`, trades `273`, win rate `51.65%`
  - 分年份:
    - 2021 `+10.37%`
    - 2022 `+39.70%`
    - 2023 `+14.31%`
    - 2024 `+49.66%`
    - 2025 `+12.57%`
    - 2026 YTD `-8.80%`
- 当前判断:
  - 手数修正没有改变 `manual_p2_k0p5_r0_6td` 作为当前主策略底座的结论
  - 净收益略升、回撤略升，整体变化很小
  - 下一步可以在同一新手数口径下重跑 P/K/M `6td`，再推进 rolling cohort `5td / 6td`

### [AMV] 手数修正后 P/K/M 6td 校准重跑

- 背景:
  - 用户要求在重跑 manual 后，用同一新手数口径重跑三组 P/K/M `6td no-stop`
  - 信号使用同批 `20260519_0854xx` 静态 sleeve 导出
- Rust 回测:
  - 配置: `backtest-engine/crates/amv-topn/config_6td_no_stop.toml`
  - 汇总产物:
    - `artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_new_lot_summary_20260519_085900.json`
    - `artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_new_lot_summary_20260519_085900.csv`
- 核心结果:
  - `pkm_p1_k0p5_m1`: net `+90.34%`, gross `+128.43%`, MaxDD `48.04%`, trades `269`, win rate `47.21%`
  - `pkm_p3_k1_m2`: net `+103.58%`, gross `+142.04%`, MaxDD `48.08%`, trades `268`, win rate `47.01%`
  - `pkm_p2_k0p5_m0p5`: net `+12.10%`, gross `+47.35%`, MaxDD `62.15%`, trades `270`, win rate `45.93%`
- 分年份:
  - `pkm_p1_k0p5_m1`: 2021 `+5.72%`, 2022 `+23.94%`, 2023 `-13.94%`, 2024 `+23.15%`, 2025 `+59.60%`, 2026 YTD `-14.12%`
  - `pkm_p3_k1_m2`: 2021 `+0.35%`, 2022 `+28.05%`, 2023 `-12.82%`, 2024 `+37.17%`, 2025 `+52.14%`, 2026 YTD `-12.93%`
  - `pkm_p2_k0p5_m0p5`: 2021 `+0.47%`, 2022 `+10.17%`, 2023 `-7.68%`, 2024 `+56.85%`, 2025 `-9.44%`, 2026 YTD `-22.77%`
- 当前判断:
  - 手数修正后 P/K/M 结果整体抬升，但仍没有替代 `manual_p2_k0p5_r0_6td`
  - 最好的 `P3/K1/M2` 约为主基线收益的六成，回撤约为主基线的三倍
  - P/K/M 仍主要表现为 2025 型高波动补充袖子，2026 YTD 仍弱于 manual
  - 下一步应进入 rolling cohort 口径，检验每日 Top3 信号产能是否被 `max_positions = 3` 压制

### [AMV] Rolling cohort 真实组合第一版回测

- 背景:
  - 用户提出用真实账户滚动持仓检验每日 Top3 信号产能
  - 目标是判断 P/K/M 标签侧 rolling NAV 优势是否只是被 `max_positions = 3` 单组持仓限制压掉
- 代码口径确认:
  - `amv-topn` 已支持 `max_positions / max_daily_buys / position_size_pct / max_hold_trading_days`
  - ECS 执行顺序为 `[open] process_buy_signals -> [close] check_exit_conditions -> update_stats`
  - 当前实现会跳过已持有代码，因此 rolling cohort 是“不重复加仓”的真实账户口径，不是标签侧可重复 cohort
- 配置:
  - 临时配置目录: `artifacts/amv_static_sleeve_signals/rolling_cohort_configs_20260519_090300`
  - `config_5td_rolling18_no_stop.toml`:
    - `max_positions = 18`
    - `max_daily_buys = 3`
    - `position_size_pct = 1 / 18`
    - `max_hold_trading_days = 5`
  - `config_6td_rolling21_no_stop.toml`:
    - `max_positions = 21`
    - `max_daily_buys = 3`
    - `position_size_pct = 1 / 21`
    - `max_hold_trading_days = 6`
- 回测范围:
  - `manual_p2_k0p5_r0`
  - `pkm_p1_k0p5_m1`
  - `pkm_p2_k0p5_m0p5`
  - `pkm_p3_k1_m2`
  - 均使用 `20260519_0854xx` 新手数口径信号产物
- 汇总产物:
  - `artifacts/amv_static_sleeve_signals/amv_rolling_cohort_new_lot_summary_20260519_090500.json`
  - `artifacts/amv_static_sleeve_signals/amv_rolling_cohort_new_lot_summary_20260519_090500.csv`
- 核心结果:
  - `manual_p2_k0p5_r0 5td rolling18`: net `+11.90%`, MaxDD `9.25%`, trades `1361`, win rate `46.66%`
  - `manual_p2_k0p5_r0 6td rolling21`: net `+23.61%`, MaxDD `9.33%`, trades `1335`, win rate `46.82%`
  - `pkm_p1_k0p5_m1 5td rolling18`: net `-13.99%`, MaxDD `31.19%`, trades `1248`, win rate `45.75%`
  - `pkm_p1_k0p5_m1 6td rolling21`: net `-7.29%`, MaxDD `28.28%`, trades `1235`, win rate `44.53%`
  - `pkm_p2_k0p5_m0p5 5td rolling18`: net `-6.23%`, MaxDD `31.08%`, trades `1301`, win rate `45.89%`
  - `pkm_p2_k0p5_m0p5 6td rolling21`: net `+0.21%`, MaxDD `26.59%`, trades `1293`, win rate `45.09%`
  - `pkm_p3_k1_m2 5td rolling18`: net `-14.10%`, MaxDD `29.90%`, trades `1241`, win rate `46.41%`
  - `pkm_p3_k1_m2 6td rolling21`: net `-8.57%`, MaxDD `27.50%`, trades `1226`, win rate `44.54%`
- 当前判断:
  - rolling cohort 没有把 P/K/M 标签侧优势兑现为真实账户优势
  - manual 仍是 rolling 口径下唯一有明显正收益的候选，且 `6td rolling21` 好于 `5td rolling18`
  - P/K/M rolling 的 2025 确有正贡献，但被 2022/2023 与高回撤抵消
  - 下一步若继续 rolling，应先诊断收益被摊薄的原因: 重复代码跳过、现金利用率、买入失败、分散化稀释，而不是继续加权重网格
  - 结论上，P/K/M 不替代 manual；rolling cohort 也暂不作为默认主策略替代

### [AMV] close-to-close cohort diagnostic crate

- 背景:
  - 用户指出 Python rolling NAV 与 Rust rolling cohort 差距太大，需要定位损耗断点
  - 用户要求新增独立 crate，避免把 close-to-close 诊断口径混入真实 `amv-topn` 交易引擎
  - 用户提醒必须保留涨停/开盘过滤，否则会重复 rotation 中“不可成交信号被计入收益”的问题
- 新增:
  - Rust crate: `backtest-engine/crates/amv-cohort-diagnostic`
  - Python 诊断信号导出: `scripts/amv_close_to_close_diagnostic_signal_export.py`
  - Python 诊断回测 helper: `scripts/amv_cohort_diagnostic_backtest.py`
  - 默认配置:
    - `config_5td_rolling18.toml`
    - `config_6td_rolling21.toml`
- 第一版诊断口径 B:
  - 输入必须是 unshifted close-to-close diagnostic signal
  - 信号日 close 买入
  - 持有 `max_hold_trading_days` 后 close 卖出
  - 不允许重复持有同一代码
  - 保留真实资金、手数、成本、`max_positions / max_daily_buys / position_size_pct`
  - 信号日 close 涨停不可买
  - 默认开盘过滤: `max_open_gap_pct = 0.098`
  - 继续支持 AMV bull entry gate
- 校验:
  - `uv run python -m py_compile scripts/amv_close_to_close_diagnostic_signal_export.py scripts/amv_cohort_diagnostic_backtest.py`
  - `cargo fmt --package bt-amv-cohort-diagnostic`
  - `cargo check -p bt-amv-cohort-diagnostic`
- 当前判断:
  - 新 crate 已可编译，后续可先导出 manual / P/K/M 的 unshifted diagnostic signal
  - 下一步跑 B 口径，观察加入 close 涨停过滤与开盘过滤后，close-to-close 账户结构收益是否仍接近 Python rolling NAV

### [AMV] close-to-close cohort diagnostic B 口径首轮结果

- 背景:
  - 用户要求直接导出 unshifted close-to-close diagnostic signal 并开跑 B 口径
  - 重点验证 Python rolling NAV 的高收益是否来自大量现实不可成交信号
- 信号导出:
  - `uv run python scripts/amv_close_to_close_diagnostic_signal_export.py --sleeves manual_p2_k0p5_r0,pkm_p1_k0p5_m1,pkm_p2_k0p5_m0p5,pkm_p3_k1_m2 --end-date 2026-05-10`
  - 新产物:
    - `artifacts/amv_close_to_close_diagnostic_signals/20260519_135621_manual_p2_k0p5_r0/signal.meta.json`
    - `artifacts/amv_close_to_close_diagnostic_signals/20260519_135623_pkm_p1_k0p5_m1/signal.meta.json`
    - `artifacts/amv_close_to_close_diagnostic_signals/20260519_135625_pkm_p2_k0p5_m0p5/signal.meta.json`
    - `artifacts/amv_close_to_close_diagnostic_signals/20260519_135627_pkm_p3_k1_m2/signal.meta.json`
- 回测:
  - `uv run python scripts/amv_cohort_diagnostic_backtest.py <signal_dir>`
  - 配置:
    - `config_5td_rolling18.toml`
    - `config_6td_rolling21.toml`
  - 汇总产物:
    - `artifacts/amv_close_to_close_diagnostic_signals/amv_close_to_close_cohort_diagnostic_summary_20260519_140000.json`
    - `artifacts/amv_close_to_close_diagnostic_signals/amv_close_to_close_cohort_diagnostic_summary_20260519_140000.csv`
- 核心结果:
  - `manual_p2_k0p5_r0 5td rolling18`: net `+11.66%`, gross `+38.33%`, MaxDD `10.67%`, trades `1335`
  - `manual_p2_k0p5_r0 6td rolling21`: net `+20.88%`, gross `+44.71%`, MaxDD `11.08%`, trades `1311`
  - `pkm_p1_k0p5_m1 5td rolling18`: net `+5.63%`, gross `+19.90%`, MaxDD `14.58%`, trades `779`
  - `pkm_p1_k0p5_m1 6td rolling21`: net `+5.87%`, gross `+17.97%`, MaxDD `15.38%`, trades `772`
  - `pkm_p2_k0p5_m0p5 5td rolling18`: net `+11.07%`, gross `+30.06%`, MaxDD `12.88%`, trades `980`
  - `pkm_p2_k0p5_m0p5 6td rolling21`: net `+15.31%`, gross `+31.71%`, MaxDD `11.24%`, trades `971`
  - `pkm_p3_k1_m2 5td rolling18`: net `+8.65%`, gross `+22.41%`, MaxDD `11.78%`, trades `739`
  - `pkm_p3_k1_m2 6td rolling21`: net `+8.83%`, gross `+20.44%`, MaxDD `12.38%`, trades `732`
- close 涨停/高开过滤:
  - manual: close 涨停过滤 `138`, 高开过滤 `0`
  - `P1/K0.5/M1`: close 涨停过滤 `812`, 高开过滤 `3`
  - `P2/K0.5/M0.5`: close 涨停过滤 `602`, 高开过滤 `3`
  - `P3/K1/M2`: close 涨停过滤 `856`, 高开过滤 `3`
- 当前判断:
  - close-to-close B 口径没有复刻 Python rolling NAV 的 `+1000%` 级别收益
  - P/K/M 的 Python rolling NAV 高收益很大一部分来自信号日收盘已经涨停、现实无法买入的票
  - 加入 close 涨停不可买、no repeat、资金/手数/成本后，P/K/M 只剩低两位数收益，且 manual 仍更稳
  - 对比真实 `T+1 open` rolling:
    - manual `6td`: close-to-close B `+20.88%` vs T+1 open `+23.61%`
    - `P1 6td`: close-to-close B `+5.87%` vs T+1 open `-7.29%`
    - `P2 6td`: close-to-close B `+15.31%` vs T+1 open `+0.21%`
    - `P3 6td`: close-to-close B `+8.83%` vs T+1 open `-8.57%`
  - 这说明 P/K/M 还存在明显 T+1 open 执行损耗，但更大的第一断点已经是 close 涨停不可买
  - 下一步如果继续，应做过滤归因表，而不是直接把 Python rolling NAV 视作可交易上限

## 2026-05-18

### [AMV] Rolling cohort 持仓口径讨论

- 背景:
  - 当前 `amv-topn` 主基线使用 `max_positions = 3`，等价于只允许一组 Top3 在场
  - 用户提出更接近 rolling NAV 的真实组合口径: 每天最多买 Top3，多组 cohort 同时持有，到期组在收盘卖出
- 当前 Rust 执行顺序确认:
  - 每日流程为 `[open] buy T+1 shifted signals`，再 `[close] sell by risk or max hold days`
  - `max_hold_trading_days = 6` 时，若执行买入日记为 `E`，则 `E+6 close` 触发卖出
  - 因为是开盘先买、收盘再卖，`6td` rolling cohort 若每天都买 Top3，完整槽位约为 `7 * 3 = 21`
  - 若未来设计“先卖再买”或盘中/开盘卖出机制，槽位才可能降为 `6 * 3 = 18`
- 配置可行性:
  - `amv-topn` 已有 `max_positions` 与 `max_daily_buys`
  - 第一版 rolling cohort 可通过配置测试:
    - `max_daily_buys = 3`
    - `max_positions = 21` for `6td` 当前流程
    - `position_size_pct = 1 / max_positions`
  - `5td` 也应重新扫，因为滚动持仓后最佳持有期可能不再等于单组 Top3 的 `6td`
- 手数口径:
  - 当前回测引擎买入时使用 `bt_core::round_to_lot`
  - 当前 `LOT_SIZE = 300`，比真实 A 股常见 `100` 股一手更保守
  - rolling cohort 下单票资金更小，手数向下取整会更显著影响可买入数量与跳单率
- 当前判断:
  - rolling cohort 是值得补测的真实组合口径，能检验 P/K/M 标签侧优势是否被 `max_positions = 3` 的资金槽限制压掉
  - 补测前应明确是否继续沿用保守 `300` 股手数，或改成更贴近 A 股主板的 `100` 股口径

### [AMV] P/K/M 动量增强袖子 Rust 真实回测

- 背景:
  - 年度权重网格发现 `P/K/M` 动量增强组合在 2025/2026 标签侧显著强于旧 `P2/K0.5/R0`
  - 用户提醒 `6td` 命名来自早期自然日/交易日口径迁移，本轮必须确认使用当前更严谨的交易日持有期
- 口径确认:
  - 当前 Rust `bt-amv-topn` 使用 `max_hold_trading_days`
  - `config_6td_no_stop.toml` 中 `max_hold_trading_days = 6`
  - `systems.rs` 中 `hold_trading_days = current_trade_index - entry_trade_index`
  - 因此本轮 `6td` 明确是交易日持有期，不是早期自然日口径
- 新增/调整:
  - `scripts/amv_static_sleeve_signal_export.py` 新增:
    - `pkm_p1_k0p5_m1`
    - `pkm_p2_k0p5_m0p5`
    - `pkm_p3_k1_m2`
  - Canvas: `amv-pkm-sleeve-rust-backtest.canvas.tsx`
- 信号导出:
  - `uv run python scripts/amv_static_sleeve_signal_export.py --sleeves pkm_p1_k0p5_m1,pkm_p2_k0p5_m0p5,pkm_p3_k1_m2 --end-date 2026-05-10`
  - 产物:
    - `artifacts/amv_static_sleeve_signals/20260518_115914_pkm_p1_k0p5_m1/signal.meta.json`
    - `artifacts/amv_static_sleeve_signals/20260518_115916_pkm_p2_k0p5_m0p5/signal.meta.json`
    - `artifacts/amv_static_sleeve_signals/20260518_115918_pkm_p3_k1_m2/signal.meta.json`
  - 三组均为 `812` 个执行信号日；执行日非 bull 阻断 `126` 行
- Rust 回测:
  - 配置: `backtest-engine/crates/amv-topn/config_6td_no_stop.toml`
  - 口径: `T+1 open`, Top3, `6td` trading days, no-stop, AMV bull entry gate
  - 汇总产物:
    - `artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_summary_20260518_120000.json`
    - `artifacts/amv_static_sleeve_signals/pkm_rust_6td_no_stop_summary_20260518_120000.csv`
- 核心结果:
  - 当前主基线 `manual_p2_k0p5_r0_6td`: net `+168.01%`, MaxDD `14.97%`, trades `273`, win rate `51.65%`
  - `pkm_p1_k0p5_m1`: net `+83.18%`, gross `+120.07%`, MaxDD `49.27%`, trades `268`, win rate `47.01%`
  - `pkm_p3_k1_m2`: net `+92.79%`, gross `+130.17%`, MaxDD `48.94%`, trades `267`, win rate `46.82%`
  - `pkm_p2_k0p5_m0p5`: net `+11.45%`, gross `+46.14%`, MaxDD `61.87%`, trades `270`, win rate `45.56%`
- 分年份:
  - `pkm_p1_k0p5_m1`: 2021 `+6.0%`, 2022 `+21.0%`, 2023 `-13.1%`, 2024 `+23.4%`, 2025 `+59.2%`, 2026 YTD `-16.4%`
  - `pkm_p3_k1_m2`: 2021 `+1.2%`, 2022 `+24.6%`, 2023 `-11.2%`, 2024 `+36.7%`, 2025 `+50.9%`, 2026 YTD `-16.5%`
  - `pkm_p2_k0p5_m0p5`: 2021 `+1.4%`, 2022 `+8.6%`, 2023 `-7.3%`, 2024 `+55.1%`, 2025 `-9.1%`, 2026 YTD `-22.6%`
- 当前判断:
  - 年度权重网格的 `P/K/M` 标签侧优势没有兑现为更好的 Rust 主基线
  - `P1/K0.5/M1` 和 `P3/K1/M2` 明显改善 2025，但总收益只有主基线约一半，最大回撤接近 `49%`
  - 2026 并未被修复，P/K/M 候选的 2026 YTD 亏损比当前主基线更深
  - 因此 `P/K/M` 不应替代 `manual_p2_k0p5_r0_6td`
  - 它最多作为“2025 型高波动动量补充袖子”保留观察，不能解释为 2026 解法
  - 下一步应回到执行日/环境确认和 `cash_ok`，尤其是识别入场后上冲空间不足、执行日收弱、短期市场宽度/赚钱效应恶化的状态

### [AMV] P/K/M 动量增强袖子持仓周期扫描

- 背景:
  - 用户提醒 `manual P2/K0.5/R0` 主基线也是先扫持仓周期，才定位到当前 `6td`
  - 因此 P/K/M 不能只看 `6td`，需要同口径扫描 `1td` 到 `7td`
- 口径:
  - 复用三组 P/K/M 静态信号
  - 配置:
    - `config_1td_no_stop.toml`
    - `config_2td_no_stop.toml`
    - `config_3td_no_stop.toml`
    - `config_6td_no_stop.toml`
    - 从 `config_6td_no_stop.toml` 派生临时 `4td / 5td / 7td` no-stop 配置
  - 均为 `T+1 open`, Top3, no-stop, AMV bull entry gate
  - 所有 `td` 均为 `max_hold_trading_days` 交易日口径
- 产物:
  - `artifacts/amv_static_sleeve_signals/pkm_rust_horizon_scan_1td_to_7td_summary_20260518_121300.json`
  - `artifacts/amv_static_sleeve_signals/pkm_rust_horizon_scan_1td_to_7td_summary_20260518_121300.csv`
  - Canvas `amv-pkm-sleeve-rust-backtest.canvas.tsx` 已补充“持仓周期扫描”
- 核心结果:
  - `pkm_p1_k0p5_m1`:
    - `1td`: net `-79.91%`, gross `-32.28%`, MaxDD `86.75%`, trades `784`
    - `2td`: net `-72.90%`, gross `-31.00%`, MaxDD `83.87%`, trades `541`
    - `3td`: net `-39.99%`, gross `+8.93%`, MaxDD `79.38%`, trades `427`
    - `4td`: net `-52.18%`, gross `-23.77%`, MaxDD `73.48%`, trades `359`
    - `5td`: net `-19.32%`, gross `+14.72%`, MaxDD `65.47%`, trades `299`
    - `6td`: net `+83.18%`, gross `+120.07%`, MaxDD `49.27%`, trades `268`
    - `7td`: net `-17.70%`, gross `+15.13%`, MaxDD `72.81%`, trades `231`
  - `pkm_p2_k0p5_m0p5`:
    - `1td`: net `-69.71%`, gross `-16.09%`, MaxDD `77.77%`
    - `2td`: net `-74.68%`, gross `-27.63%`, MaxDD `87.61%`
    - `3td`: net `-45.79%`, gross `+2.54%`, MaxDD `73.67%`
    - `4td`: net `-64.87%`, gross `-40.32%`, MaxDD `81.39%`
    - `5td`: net `-19.43%`, gross `+17.28%`, MaxDD `66.47%`
    - `6td`: net `+11.45%`, gross `+46.14%`, MaxDD `61.87%`
    - `7td`: net `-10.50%`, gross `+22.45%`, MaxDD `67.32%`
  - `pkm_p3_k1_m2`:
    - `1td`: net `-79.99%`, gross `-28.93%`, MaxDD `86.88%`
    - `2td`: net `-78.04%`, gross `-37.77%`, MaxDD `86.96%`
    - `3td`: net `-29.48%`, gross `+25.38%`, MaxDD `76.64%`
    - `4td`: net `-61.11%`, gross `-33.00%`, MaxDD `78.18%`
    - `5td`: net `-15.06%`, gross `+19.61%`, MaxDD `63.45%`
    - `6td`: net `+92.79%`, gross `+130.17%`, MaxDD `48.94%`
    - `7td`: net `-34.44%`, gross `-6.18%`, MaxDD `76.11%`
- 当前判断:
  - P/K/M 和主基线一样，最佳持仓周期仍是 `6td`
  - `4td / 5td / 7td` 补扫后没有改变结论；`5td` 比短持仓有所修复但仍为负，`7td` 又明显回落
  - 这说明标签侧 P/K/M 动量优势不是“换个持仓期就能兑现”的 edge
  - 结论仍然是不替代 `manual_p2_k0p5_r0_6td`；下一步继续回到 `cash_ok` / 入场环境确认

### [AMV] 年度权重网格诊断: 回到早期 TopN 探索框架

- 背景:
  - 用户指出后续 LTR、sleeve oracle、B3 等探索，本质上都在尝试补 `manual_p2_k0p5_r0_6td` 在 2025/2026 的弱势
  - 既然后续复杂路线暂未证明有效，本轮回到更可解释的 `AMV 多头宽池权重网格` 模式，按年份诊断最佳权重
- 环境修复:
  - `pyproject.toml` 中 `pyobjc-framework-cocoa / quartz / vision` 已加 `sys_platform == 'darwin'` marker
  - 这样 Windows 与 macOS 后续都继续统一使用 `uv run`
  - `uv lock` 已重新解析通过
- 新增:
  - `scripts/amv_yearly_weight_grid.py`
  - Canvas: `amv-yearly-weight-grid.canvas.tsx`
- 运行:
  - `uv run python scripts/amv_yearly_weight_grid.py --end-date 2026-05-08`
  - 产物: `artifacts/amv_bull_pool_yearly_weight_grid/20260518_100342/summary.json`
- 口径:
  - AMV bull + LF2
  - Top3
  - `6d close-to-close` 标签侧诊断
  - 覆盖 `2021-04-20 -> 2026-04-27`
  - 候选共 `90` 个:
    - 当前基线 `P2/K0.5/R0`
    - 价格/K线/动量/风险单因子
    - 旧 `P/K/R` 网格
    - 新增去重后的 `P/K/M` 网格，其中 `M` 使用 `ret_5d / ret_20d`
- 当前基线分年份:
  - 2021: mean `+1.10%`, edge `+0.39pp`
  - 2022: mean `+1.17%`, edge `+0.80pp`
  - 2023: mean `+0.66%`, edge `+1.26pp`
  - 2024: mean `+3.34%`, edge `+2.17pp`
  - 2025: mean `+0.69%`, edge `-0.70pp`
  - 2026: mean `+1.10%`, edge `+0.03pp`
- 年度最佳:
  - 2024:
    - `KLEN` 单因子 mean `+3.81%`, edge `+2.64pp`, rolling NAV `+86.78%`
    - `P3/K0.5/M0.5` mean `+3.62%`, edge `+2.44pp`
    - 当前 `P2/K0.5/R0` 仍有 edge `+2.17pp`
  - 2025:
    - `P1/K0.5/M1` mean `+4.76%`, edge `+3.37pp`, rolling NAV `+86.31%`
    - `P3/K1/M2` mean `+4.74%`, edge `+3.35pp`
    - 当前 `P2/K0.5/R0` edge `-0.70pp`
  - 2026:
    - `P1/K0.5/M1` mean `+7.05%`, edge `+5.98pp`, rolling NAV `+39.09%`
    - `P3/K1/M2` mean `+6.94%`, edge `+5.87pp`
    - 当前 `P2/K0.5/R0` edge `+0.03pp`
- 弱年份综合:
  - `P1/K0.5/M1`: 2025/2026 平均 edge `+4.68pp`, 全样本 edge `+2.06pp`, `6/6` 年正 edge
  - `P3/K1/M2`: 2025/2026 平均 edge `+4.61pp`, 全样本 edge `+2.11pp`, `6/6` 年正 edge
  - `P2/K0.5/M1`: 2025/2026 平均 edge `+4.25pp`, 全样本 edge `+2.03pp`, `6/6` 年正 edge
- Walk-forward 检查:
  - 用 2021-2024 训练 edge / tradeoff 选参数，都会选到 `P2/K0.5/M0.5`
    - 2025 测试 edge `+2.07pp`, mean `+3.45%`
  - 用 2021-2025 训练:
    - edge 均值选择 `P3/K1/M2`, 2026 测试 edge `+5.87pp`, mean `+6.94%`
    - tradeoff 选择 `P1/K0.5/M1`, 2026 测试 edge `+5.98pp`, mean `+7.05%`
- 当前判断:
  - 2025/2026 弱势不是 AMV bull 宽池 Top3 框架整体失效，更像旧 `P2/K0.5/R0` 价格位置权重过重、缺少动量项
  - 新增动量后的 `P/K/M` 组合不是单一年份 hindsight 偶然；walk-forward 也能从历史样本选出动量增强权重
  - 但这仍是 `6d close-to-close` 标签侧诊断，不能直接替代 Rust 主基线
  - 下一步应把 `P1/K0.5/M1 / P2/K0.5/M0.5 / P3/K1/M2` 导出为静态 sleeve，接 `T+1 open / 6td / Top3 / no-stop` Rust 真实回测
  - 当前 `manual_p2_k0p5_r0_6td` 仍是已验证主策略底座；`P/K/M` 是新候选，不是已上线替代

## 2026-05-17

### [B3] 最终收口: 归档为事件型补充候选

- 决策:
  - B3 不作为当前主线继续深挖
  - 保留 `backtest-engine/crates/b3`、`scripts/b3_tdx_signal_export.py`、`scripts/b3_candidate_ranking_lab.py` 作为后续事件型补充研究资产
  - 后续只有在做组合/低重合度补充时再回看 B3，不再单独优化排序字段、波段参数或专用 crate 逻辑
- 收口依据:
  - 固定 `6td` B3: 净收益 `+22.61%`，最大回撤 `42.20%`
  - B3 波段 v0: 净收益 `+18.49%`，最大回撤 `43.39%`
  - B1 `rw_dif_pct` 排序迁移失败: 净收益 `-28.32%`，最大回撤 `56.31%`
  - 全量 B3 候选信号 `6td`: 平均收益 `+1.02%`，胜率 `49.87%`，平均盈亏比 `1.50`
  - 单字段 Top3 排序均未跑赢全部候选平均
  - 收益主要依赖少数大涨事件票，不像稳定可反复利用的主策略 alpha
- 对比锚点:
  - `manual_p2_k0p5_r0_6td` 修正后仍为净收益 `+168.01%`，最大回撤 `14.97%`
  - B3 与该底座差距过大，不值得继续占用主线研发时间
- 文档整理:
  - `project-status.md` 已新增“当前决策摘要”
  - B3 在 `project-status.md` 中已标记为归档/非主线
  - 后续应让 `project-status.md` 保持当前状态和最终决策，历史过程继续留在 `progress.md` / `experiments/` / Canvas

### [AMV] attack_ok 第一版结论

- 今日修正:
  - 本文件顶部新增 `2026-05-17` 记录，避免后续误以为最新 AMV sleeve/gating 分析仍属于 `2026-05-10`
- 当前理论锚点:
  - `amv-constrained-oracle.canvas.tsx`（AMV 受约束 Oracle）
  - `attack_ok` 标签来自 `top_ret_dailyized + 3% margin + allow_cash=true`
- 第一版实验:
  - 脚本: `scripts/amv_attack_ok_lab.py`
  - 产物: `artifacts/amv_attack_ok/20260517_150811/summary.json`
  - Canvas: `amv-attack-ok-lab.canvas.tsx`
- 当前结论:
  - 第一版 `attack_ok` 没有证明当前交易前状态特征能稳定学习“什么时候进攻”
  - 2023/2024 AUC 接近随机，2025 仅略高，2026 样本太少
  - 验证集 F1 阈值退化为接近 `always_attack`，固定高阈值又召回过低
  - 下一步需要收紧标签或固定单一进攻袖子后再评估，而不是直接把当前 `attack_ok` 接交易

### [B3] TDX 公式 Polars 复刻 v0

> 历史探索记录: 本节及后续 B3 中间“下一步”已被顶部 `[B3] 最终收口` 替代，B3 当前状态以归档结论为准。

- 背景:
  - 暂停继续堆 AMV sleeve switching，转向研究 AMV bull 下另一个明确 archetype
  - B3 定位为 `AMV bull` 择时下的“强异动 K 后确认接力”子策略，未来可与 `manual_p2_k0p5_r0_6td` 做组合/互补性分析
- 新增:
  - `scripts/b3_tdx_signal_export.py`
  - Canvas: `b3-tdx-signal-backtest.canvas.tsx`
- 翻译口径:
  - 只复刻最终 `B3触发` 直接依赖的变量，不翻译未参与最终决策的 KDJ/RSI/砖型图/异动统计等中间指标
  - 保留 AMV bull 前置过滤
  - 加 ST 剔除、流动性过滤: `market_cap_100m >= 100`, `amount_ma20 >= 5e7`
  - 原始 B3 触发后，按前一日触发 K 涨幅排序，每日取 Top3 写入 Rust `signal.parquet`
  - TDX 原式 `(REF(知行短期趋势线,1) >= REF(知行短期趋势线,2))+0.01` 在布尔上下文近似恒真；当前主信号按 TDX 语义处理，同时输出 strict 版本计数做诊断
- 信号产物:
  - `artifacts/b3_tdx_signals/20260517_173457/signal.meta.json`
  - `signal.parquet`
  - `raw_candidate_signals.csv`
- 信号统计 (`2019-01-02 ~ 2026-05-08`):
  - 原始 B3 触发: `13,171` 行, `1,522` 天
  - strict 短趋势线版本: `10,648` 行
  - AMV bull + 流动性后: `2,026` 行, `529` 天
  - Top3 后执行日信号: `1,166` 行, `528` 天
  - 执行日开盘涨停诊断: `1` 行，占比 `0.086%`
  - Rust `2021+` 实际读取信号: `873` 行；AMV 非 bull 阻止 `24` 行；修正后涨停过滤 `1` 行
- 回测口径修正:
  - 发现 `bt_core::price_limit_pct` 只识别裸代码 `300/301/688/689`，未识别 QMT 格式 `sz.300xxx / sh.688xxx`
  - 已修复 `backtest-engine/crates/core/src/lib.rs`，同时保留裸代码和 `sz./sh.` 前缀代码支持
  - 新增单测 `price_limit_pct_supports_qmt_prefixed_codes`
  - `cargo test -p bt-core` 通过
- Rust 修正后回测 (`bt-amv-topn`, `T+1 open`, Top3, no-stop, config start `2021-01-01`):
  - `1td`: 净收益 `-54.40%`, 毛收益 `-14.58%`, 最大回撤 `63.31%`, 交易 `562`, 胜率 `42.0%`
  - `2td`: 净收益 `-28.79%`, 毛收益 `+4.34%`, 最大回撤 `57.19%`, 交易 `433`, 胜率 `43.6%`
  - `3td`: 净收益 `+0.48%`, 毛收益 `+33.92%`, 最大回撤 `40.30%`, 交易 `352`, 胜率 `46.0%`
  - `6td`: 净收益 `+22.61%`, 毛收益 `+50.57%`, 最大回撤 `42.20%`, 交易 `230`, 胜率 `47.4%`
  - 涨停过滤从 `2` 行修正为 `1` 行；多放入的 20cm 代码样本对 B3 结果有负面影响，说明旧结果不是同一真实口径
- 当前判断:
  - B3 不是废策略；在 AMV bull + 真实执行下，修正后 `6td` 仍能跑出正收益，`3td` 仅小幅为正
  - 但第一版明显弱于 `manual_p2_k0p5_r0_6td`，暂时不能替代主策略底座
  - `1td/2td` 对交易成本非常敏感，短持仓不适合作为当前主方向
  - 下一步不应立刻调参，而应先分析 B3 与 `manual_p2_k0p5_r0_6td` 的信号重合度、年度互补性和组合价值
  - 额外注意: 涨跌停代码前缀 bug 已修复；如果要严谨比较历史 AMV/LTR/oracle 结果，后续应重跑关键基线

### [B3] Rust 波段回测 crate 起步

- 背景:
  - 用户确认 B3 原始视频是波段策略，买入后存在多种应对情况，不应继续只用固定 `1td/2td/3td/6td` 持有期评估
  - 决策: 不重命名 `b1`，新增独立 crate `backtest-engine/crates/b3`
- 新增:
  - `backtest-engine/crates/b3/Cargo.toml`
  - `backtest-engine/crates/b3/config.toml`
  - `backtest-engine/crates/b3/src/{main.rs,resources.rs,components.rs,systems.rs}`
  - `backtest-engine/Cargo.toml` workspace member 加入 `crates/b3`
- 入口/退出 v0:
  - 买入: 消费 `scripts/b3_tdx_signal_export.py` 生成的执行日 `is_signal / score / rank`
  - 结构止损: 默认以前一日触发 K 的 `trigger_low` 为结构止损位
  - 波段退出: 跌破白线、早期失败、弱势未兑现、移动止损、最长持有交易日
  - 保留 AMV bull gate，默认 `require_bull_regime = true`
- 信号导出同步补充:
  - `scripts/b3_tdx_signal_export.py` 现在输出 `white_line / yellow_line / trigger_type / trigger_low / trigger_high / trigger_close / prev_trigger_volume_ratio`
  - 这些字段供 `bt-b3` 做结构止损和趋势线退出，不再要求 Rust 端重算 TDX 指标
- 验证:
  - `cargo check --manifest-path backtest-engine/Cargo.toml -p bt-b3` 通过
  - `uv run python -m py_compile scripts/b3_tdx_signal_export.py` 通过
- 注意:
  - 当前只是第一版机制落地，还没有重导出新字段信号并跑真实 B3 波段结果
  - 后续需要用新 `signal.parquet` 跑 `bt-b3`，再对比固定 6td 和 `manual_p2_k0p5_r0_6td`

### [B3] Rust 波段回测 v0 试跑

- 信号:
  - 新导出: `artifacts/b3_tdx_signals/20260517_180746/signal.parquet`
  - 统计: 原始 B3 触发 `13,171` 行；AMV bull + 流动性后 `2,026` 行；Top3 执行日信号 `1,166` 行
  - 仍使用临时排序 `score = prev_trigger_ret_pct`，不是已验证 alpha 排序字段
- 发现并修复:
  - 第一版试跑发现 `TrailingStop` 激活后未触发时会挡住后续 `Weak/MaxHold` 检查
  - 已修复 `backtest-engine/crates/b3/src/systems.rs`，并在报告中补充 exit reason 汇总
- 最终参考结果:
  - 报告: `artifacts/b3_tdx_signals/20260517_180746/backtests_v0_summary/report.json`
  - 时间: `2021-01-01 ~ 2026-05-08`
  - 初始资金 `50 万`，Top3，最多 3 仓，默认波段出场规则
  - 净收益 `+18.49%`，毛收益 `+39.00%`，最大回撤 `43.39%`
  - 交易 `191` 笔，胜率 `35.08%`，总成本 `10.25 万`
- Exit reason 诊断:
  - `TrailingStop`: `24` 笔，胜率 `100%`，平均 `+19.97%`，贡献 `+72.12 万`
  - `FastFail`: `62` 笔，胜率 `0%`，平均 `-5.15%`，拖累 `-47.18 万`
  - `BreakWhiteLine`: `62` 笔，胜率 `33.9%`，平均 `-0.87%`，拖累 `-9.57 万`
  - `StructuralStop`: `8` 笔，平均 `-7.95%`，拖累 `-10.53 万`
  - `Weak`: `30` 笔，平均约 `0%`，接近资金占用退出
  - `MaxHoldDays`: 最大观察持仓已收敛到 `30td`
- 当前判断:
  - B3 波段 v0 有正收益，但仍弱于固定 `6td` B3 的 `+22.61% / DD 42.20%`，更远弱于 `manual_p2_k0p5_r0_6td`
  - 真正赚钱来自少数走成波段的 `TrailingStop`；主要问题是大量早期失败和跌破白线退出
  - 下一步不应先调止盈，而应先处理入场质量: 执行日开盘是否已跌破触发 K 低点/白线、候选排序字段是否有效、是否需要确认触发后不破结构再买

### [B3] B1 `rw_dif_pct` 排序对照

- 背景:
  - 用户建议参考 `backtest-engine/crates/b1/config.toml` 的 `sort_field = "rw_dif_pct"`
  - 已在 `scripts/b3_tdx_signal_export.py` 中复刻 B1 running weekly MACD 强度字段: `rw_dif_pct / rw_hist / rw_dif`
  - `bt-b3` 已支持 `sort_field / sort_ascending`，可从 parquet 任意 f64 字段排序
- 对照信号:
  - `artifacts/b3_tdx_signals/20260517_184650/signal.parquet`
  - 导出参数: `--sort-field rw_dif_pct`
  - 候选数量不变: Top3 执行日信号 `1,166` 行，排序字段范围 `-7.29 ~ 16.21`
- 波段回测:
  - 报告: `artifacts/b3_tdx_signals/20260517_184650/backtests_rw_dif_pct/report.json`
  - 净收益 `-28.32%`，毛收益 `-13.24%`，最大回撤 `56.31%`
  - 交易 `197` 笔，胜率 `32.99%`
- Exit reason:
  - `TrailingStop`: `23` 笔，平均 `+14.95%`，贡献 `+37.31 万`
  - `FastFail`: `62` 笔，平均 `-4.84%`，拖累 `-32.78 万`
  - `BreakWhiteLine`: `67` 笔，平均 `-1.27%`，拖累 `-10.75 万`
  - `StructuralStop`: `14` 笔，平均 `-8.40%`，拖累 `-14.34 万`
- 当前判断:
  - B1 的 `rw_dif_pct` 不能直接迁移为 B3 排序字段
  - 它会偏向周线动能很强的票，但 B3 是“异动后确认接力”，更怕执行日接力失败和结构破位
  - 已保留字段与配置能力，但默认导出排序恢复为 `prev_trigger_ret_pct`，`bt-b3` 默认 `sort_field = "score"`，避免误用失败口径

### [B3] 候选排序字段诊断 lab

- 新增:
  - `scripts/b3_candidate_ranking_lab.py`
  - Canvas: `b3-candidate-ranking-lab.canvas.tsx`
- 输入:
  - 候选文件: `artifacts/b3_tdx_signals/20260517_184650/raw_candidate_signals.csv`
  - 输出: `artifacts/b3_candidate_ranking_lab/20260517_185254/summary.json`
  - 标签: `T+1 open` 入场，观察 `3/6/10/20td` close/MFE/MAE
  - 有效样本: `1,496` 行，`381` 个 signal days，回测口径从 `2021-01-01` 开始
- 全部候选基准:
  - `6td` 均值 `+1.02%`
  - `10td` 均值 `+1.53%`
  - `20td MFE` 均值 `+12.44%`
  - `3td MAE` 均值 `-3.02%`
  - 快速失败代理率 `39.24%`
- 单字段 Top3 排序结论:
  - 每日按单字段 Top3 都没有跑赢全部候选平均
  - 最好的 `market_cap_100m` 升序 Top3 也只有 `6td +0.82%`，低于全部候选 `+1.02%`
  - `prev_trigger_ret_pct` 降序 Top3 为 `+0.74%`，快速失败率升到 `46.62%`
  - `rw_dif_pct` 降序 Top3 为 `+0.64%`，进一步确认 B1 周动能不适合直接当 B3 主排序
- 过滤/分桶结论:
  - `仅30日振幅Q2`: `6td +1.65%`，快速失败率 `32.62%`
  - `仅B1周线红柱Q2`: `6td +1.64%`，快速失败率 `29.95%`
  - `仅B1周线DIF最低25%`: `6td +1.56%`，快速失败率 `30.93%`
  - `仅触发K涨幅低/中50%`: `6td +1.41%`，快速失败率 `29.24%`
  - `剔除振幅/触发涨幅/白线距离最高25%`: `6td +1.32%`，快速失败率 `29.28%`
- 当前判断:
  - B3 的排序问题不是“找到一个越大越好的字段”
  - 更像是要先剔除过热、距离过远、接力失败风险高的候选
  - 下一步优先把 `drop_overheat_combo` 或 `drop_range_amp_q4` 接入信号导出后跑 Rust；前者覆盖 `871` 行且快速失败率从 `39.24%` 降到 `29.28%`，后者覆盖 `1,122` 行且规则更简单

### [B3] 全量信号盈亏比与Top盈利票

- 新增 Canvas:
  - `b3-all-signal-payoff.canvas.tsx`
- 输入:
  - `artifacts/b3_candidate_ranking_lab/20260517_185254/candidate_labels.parquet`
  - 口径: AMV bull + 流动性后的全部 B3 候选信号，`T+1 open` 入场
  - 样本: `1,496` 行，`381` 个信号日，`2020-12-31 ~ 2026-04-27`
- 全量信号盈亏比:
  - `3td`: 平均收益 `+0.55%`，胜率 `48.13%`，平均盈亏比 `1.43`，利润因子 `1.36`
  - `6td`: 平均收益 `+1.02%`，胜率 `49.87%`，平均盈利 `+6.12%`，平均亏损 `-4.08%`，平均盈亏比 `1.50`，利润因子 `1.50`
  - `10td`: 平均收益 `+1.53%`，胜率 `51.28%`，平均盈亏比 `1.53`，利润因子 `1.63`
  - `20td`: 平均收益 `+2.11%`，胜率 `49.11%`，平均盈亏比 `1.63`，利润因子 `1.58`
- 6td Top10 盈利信号:
  - `sh.601567` `2021-04-29 -> 2021-04-30`: `6td +69.41%`, `20td MFE +155.14%`
  - `sz.300188` `2023-03-27 -> 2023-03-28`: `6td +48.10%`, `20td MFE +62.65%`
  - `sh.688222` `2026-01-06 -> 2026-01-07`: `6td +44.29%`, `20td MFE +55.39%`
  - `sz.002657` `2025-08-12 -> 2025-08-13`: `6td +41.28%`, `20td MFE +53.22%`
  - `sz.002636` `2026-01-16 -> 2026-01-19`: `6td +39.03%`, `20td MFE +45.25%`
- 当前判断:
  - B3 全量信号不是完全没有 edge，但胜率接近五五开，主要靠少数大涨票拉开盈亏比
  - Top盈利票多数是单次或少数事件贡献，说明 B3 更像事件型脉冲，不像稳定可反复交易的股票池 alpha
  - 这进一步支持“B3 不值得作为主线深挖”，最多保留为事件型补充策略候选

### [AMV] manual_p2_k0p5_r0 baseline rerun after price-limit fix

- 背景:
  - 修复 Rust `price_limit_pct` 支持 `sz./sh.` 前缀后，用户要求重跑核心基线 `manual_p2_k0p5_r0`
  - 复用既有信号: `artifacts/amv_static_sleeve_signals/20260516_222301_manual_p2_k0p5_r0/signal.parquet`
- 修正后 Rust 回测 (`bt-amv-topn`, `T+1 open`, Top3, no-stop):
  - `1td`: 净收益 `-51.03%`, 毛收益 `+17.94%`, 最大回撤 `54.79%`, 交易 `835`, 胜率 `41.9%`
  - `2td`: 净收益 `-37.33%`, 毛收益 `+17.82%`, 最大回撤 `43.96%`, 交易 `573`, 胜率 `41.0%`
  - `3td`: 净收益 `+38.11%`, 毛收益 `+102.42%`, 最大回撤 `36.00%`, 交易 `439`, 胜率 `45.8%`
  - `6td`: 净收益 `+168.01%`, 毛收益 `+225.39%`, 最大回撤 `14.97%`, 交易 `273`, 胜率 `51.6%`
- 影响判断:
  - 核心 `manual_p2_k0p5_r0_6td` 基线完全不变，仍是当前最强底座
  - `1td/3td/6td` 与修复前一致
  - `2td` 轻微变化: `-37.05% -> -37.33%`，不影响结论
  - 涨停过滤仍为 `49` 行，说明该基线没有受到 QMT 代码前缀 bug 的实质影响
  - 已同步更新 `static_sleeve_backtest_summary.csv` 和 `amv-static-sleeve-backtest.canvas.tsx` 中 manual_p2 2td 的轻微变化

## 2026-05-10

### [AMV] Bull Pool Listwise LTR v0

> 历史探索记录: 本节及后续 LTR 阶段性“下一步”已被后续 Rust 回测和执行口径标签校准替代，direct LTR Top3 当前状态以 `project-status.md` / `experiments/archive-index.md` 为准。

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_listwise_ranker_lab.py`
- **实验目的**:
  - 按 `Empirical Asset Pricing via Learning-to-Rank` 的思路, 在 AMV bull 宽池内直接优化横截面 topN 排序
  - 验证模型是否能学习 `KLEN/KMID2/动量` 在不同年份和 AMV bull 阶段的有效性切换
- **实现口径**:
  - 数据起点: `2019-01-01`
  - 标签: `fwd_ret_6d`, 每日横截面前 `100` 名赋递减 relevance, 其余为 `0`
  - 模型: `LightGBM LGBMRanker`, `objective=lambdarank`, `metric=ndcg`, `label_gain` 使用线性增益
  - 训练/验证/测试: 年度 expanding walk-forward
    - `2019-2021 -> valid 2022 -> test 2023`
    - `2019-2022 -> valid 2023 -> test 2024`
    - `2019-2023 -> valid 2024 -> test 2025`
    - `2019-2024 -> valid 2025 -> test 2026`
  - 特征: 12 个个股截面 rank 特征 + AMV bull 阶段 / AMV 涨幅 / 宽池赚钱效应状态特征, 全程使用 Polars 构造数据
- **产物**:
  - `artifacts/amv_bull_pool_listwise_ranker/20260516_104849/summary.json`
  - `fold_metrics.csv`, `daily_metrics.csv`, `ltr_topn_selected.csv`, `feature_importance.csv`
  - pool: `827,388` 行, `2019-01-29 -> 2026-04-27`, `808` 个 AMV bull 交易日, `2,397` 只股票
- **默认 v0 结果 (`6d/top3`)**:
  - LTR 平均 top3 edge: `+1.153pp`, Precision@3 `0.040`
  - 年度 edge: 2023 `+0.817pp`, 2024 `+0.118pp`, 2025 `+1.538pp`, 2026 `+2.140pp`
  - 对照: `KLEN` 平均 edge `+1.818pp`, `5日动量` `+1.373pp`, `KMID2` `+1.242pp`, 当前组合 `P2/K0.5/R0` `+0.665pp`
  - 重要特征: `rank_ret_20d`, `rank_atr_14_pct`, `rank_ret_5d`, `pool_candidate_count_scaled`, `rank_KLEN`, `rank_close_to_high_20d`
- **结论**:
  - LTR v0 已经能在 2025/2026 做出正 edge, 说明模型确实捕捉到一部分因子切换
  - 但 v0 未超过最强单因子 `KLEN`, 暂时不能直接替代当前规则进入 Rust 回测
  - 下一步应做 LTR 归因和约束实验: 去掉风险类特征、按年份看 topN 选股重合度、比较 `relevance_top_k` 与 top3/top5 标签, 再决定是否进入真实交易回测

### [AMV] Bull Pool Listwise LTR 特征消融

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py`
  - 新增 Canvas: `amv-ltr-variant-ablation.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-ltr-variant-ablation.canvas.tsx`
- **脚本改动**:
  - 新增 `--variants`, 支持在同一份 dataset 上训练多个 LTR 特征组
  - 默认 variants: `full`, `no_risk`, `stock_only`, `core_state`, `kbar_momentum_state`, `momentum_state`
  - 输出中新增 `feature_variant`, `feature_count`, 分 variant 的 `feature_importance.csv`, 以及 `ltr_topn_selected.csv`
- **正式产物**:
  - `artifacts/amv_bull_pool_listwise_ranker/20260516_105316/summary.json`
  - pool 不变: `827,388` 行, `2019-01-29 -> 2026-04-27`, `808` 个 AMV bull 交易日
- **平均 edge 排名 (`6d/top3`)**:
  - `ltr_no_risk`: `+2.523pp`, Precision@3 `0.045`
  - `ltr_kbar_momentum_state`: `+2.386pp`, Precision@3 `0.036`
  - `ltr_core_state`: `+1.880pp`, Precision@3 `0.036`
  - `baseline_klen`: `+1.818pp`, Precision@3 `0.024`
  - `baseline_ret_5d`: `+1.373pp`, Precision@3 `0.049`
  - `ltr_full`: `+1.153pp`, Precision@3 `0.040`
- **分年重点**:
  - `ltr_no_risk`: 2023 `-0.292pp`, 2024 `+2.731pp`, 2025 `+1.292pp`, 2026 `+6.362pp`
  - `ltr_kbar_momentum_state`: 2023 `-0.186pp`, 2024 `+1.665pp`, 2025 `+1.224pp`, 2026 `+6.842pp`
  - `ltr_core_state`: 2023 `+0.378pp`, 2024 `+3.268pp`, 2025 `-0.321pp`, 2026 `+4.196pp`
  - `baseline_klen`: 2023 `+0.469pp`, 2024 `+2.438pp`, 2025 `+1.560pp`, 2026 `+2.804pp`
- **结论更新**:
  - LTR 方向从“能学到但不够强”升级为“有候选可以继续推进”: `no_risk` 与 `kbar_momentum_state` 已超过最强单因子 `KLEN`
  - `full` 明显弱于 `no_risk`, 说明当前 `atr_14_pct / panic_vol_ratio_20d` 这类风险特征在 listwise 训练中更像噪声或错误约束
  - 单纯 `momentum_state` 不稳, 说明 2026 的动量强并不代表可以放弃 K线结构; 更可靠的是 `KLEN/KMID2 + 动量 + AMV 状态`
  - 暂不直接接 Rust; 下一步先做 `ltr_no_risk` / `ltr_kbar_momentum_state` 的 topN 选股重合度、2026 大贡献拆解和入场后 MFE/MAE 路径归因

### [AMV] Bull Pool LTR 市场状态特征补齐

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_ranker_lab.py`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py`
  - 新增 Canvas: `amv-ltr-state-feature-completion.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-ltr-state-feature-completion.canvas.tsx`
- **用户提醒的问题**:
  - 之前讨论过的状态特征里, v0/v1 已使用:
    - `AMV bull 持续天数`: `bull_day_scaled`
    - `AMV ret_1d / ret_2d`: `amv_ret_1d_scaled / amv_ret_2d_scaled`
    - `AMV bull 初期 / 中期 / 后期`: `bull_phase_code`
    - `宽池近 5 日平均收益`: `trail_pool_ret_5d`
    - `宽池上涨比例`: `pool_up_ratio_5d`
  - 但缺少两个明确状态量:
    - `宽池 topN 动量强度`
    - `宽池成交额变化`
- **实现补齐**:
  - `scripts/amv_bull_pool_ranker_lab.py` 的 `build_dataset()` 现在保留 `amount`, `amount_ma5`, `amount_ma20`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py` 新增:
    - `--state-top-n`, 默认 `20`
    - `pool_topn_ret_5d`: 宽池中 `ret_5d` 前 `state_top_n` 的均值
    - `pool_topn_ret_20d`: 宽池中 `ret_5d` 前 `state_top_n` 的 `ret_20d` 均值
    - `pool_amount_ma5_vs_20`: 宽池成交额 `sum(amount_ma5) / sum(amount_ma20) - 1`
- **正式产物**:
  - `artifacts/amv_bull_pool_listwise_ranker/20260516_111818/summary.json`
  - 只重跑核心候选: `no_risk`, `kbar_momentum_state`
- **补齐后结果 (`6d/top3`, state_top_n=20)**:
  - `ltr_no_risk`: 平均 edge `+2.304pp`, Precision@3 `0.051`
    - 2023 `+0.054pp`, 2024 `+2.821pp`, 2025 `+2.872pp`, 2026 `+3.471pp`
  - `ltr_kbar_momentum_state`: 平均 edge `+1.953pp`, Precision@3 `0.041`
    - 2023 `+0.411pp`, 2024 `+0.910pp`, 2025 `+2.186pp`, 2026 `+4.307pp`
  - 对照 `baseline_klen`: 平均 edge `+1.818pp`
- **重要性观察**:
  - `pool_topn_ret_20d` 与 `pool_amount_ma5_vs_20` 在两个候选里都有实际增益贡献:
    - `no_risk`: `pool_topn_ret_20d` gain `412`, `pool_amount_ma5_vs_20` gain `291`
    - `kbar_momentum_state`: `pool_topn_ret_20d` gain `435`, `pool_amount_ma5_vs_20` gain `330`
- **结论更新**:
  - 之前版本不是完全没用市场状态, 但确实漏掉了“topN 动量强度”和“宽池成交额变化”这两个明确状态维度
  - 补齐后 `no_risk` 仍优于 `KLEN`, 且 2023 从负 edge 转为微正, 2025 明显增强
  - 但补齐后 2026 极端 edge 回落, 说明新增状态更像稳定器, 不是纯收益放大器
  - 下一步应比较补齐前后 `ltr_no_risk` 的选股重合度和 2026 贡献来源, 判断回落是减少偶然暴露, 还是错过真正强势票

### [AMV] LTR 选股重合度与 2026 贡献归因

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py`
  - 新增 `scripts/amv_ltr_selection_analysis.py`
  - 新增 Canvas: `amv-ltr-selection-attribution.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-ltr-selection-attribution.canvas.tsx`
- **实现说明**:
  - `ltr_topn_selected.csv` 新增 `fwd_mae_6d`, 便于做入场后路径归因
  - 重跑状态补齐后的核心候选:
    - `artifacts/amv_bull_pool_listwise_ranker/20260516_112444/summary.json`
  - 归因脚本产物:
    - `artifacts/amv_bull_pool_listwise_ranker/20260516_112444/selection_analysis_20260516_112604/summary.json`
    - 输出 `variant_summary`, `variant_year_summary`, `contribution_2026`, `top_worst_2026`, `top_code_contributors_2026`, `overlap_summary`, `before_after_*`
- **2026 路径归因**:
  - `ltr_kbar_momentum_state`: 单笔均值 `+5.36%`, 中位数 `+3.67%`, 胜率 `56.6%`, Hit15 `61.6%`, 平均 MFE `+19.49%`, 平均 MAE `-7.68%`
  - `ltr_no_risk`: 单笔均值 `+4.52%`, 中位数 `+0.76%`, 胜率 `53.5%`, Hit15 `57.6%`, 平均 MFE `+18.53%`, 平均 MAE `-7.99%`
- **贡献集中度**:
  - `ltr_kbar_momentum_state`: Top10 选股贡献占正收益 `43.3%`, 占净收益 `88.1%`, Worst10 合计 `-248.6pp`
  - `ltr_no_risk`: Top10 选股贡献占正收益 `46.3%`, 占净收益 `102.6%`, Worst10 合计 `-224.0pp`
  - 代码级贡献显示 `sz.002931` 是 2026 核心来源:
    - `kbar_momentum_state`: 入选 `6` 次, 累计贡献 `+239.5pp`, 约占净收益 `45%`
    - `no_risk`: 入选 `5` 次, 累计贡献 `+232.5pp`, 约占净收益 `52%`
- **重合度**:
  - `no_risk` vs `kbar_momentum_state` 日均重合:
    - 2023: `0.88/3`
    - 2024: `1.42/3`
    - 2025: `1.29/3`
    - 2026: `1.42/3`
  - 状态补齐前后 `no_risk` 日均重合:
    - 2023: `0.88/3`, 平均收益差 `+0.35pp`
    - 2024: `1.17/3`, 平均收益差 `+0.09pp`
    - 2025: `1.62/3`, 平均收益差 `+1.58pp`
    - 2026: `1.18/3`, 平均收益差 `-2.89pp`
- **结论更新**:
  - LTR 候选确实有 alpha, 但 2026 的强表现仍明显依赖少数股票重复入选
  - `kbar_momentum_state` 的 2026 路径优于 `no_risk`（均值/中位数/Hit15/MFE 都更高, MAE 略小）, 但同样存在 `sz.002931` 集中贡献
  - 状态补齐让 2023/2025 更稳, 但牺牲了 2026 的极端收益, 更像降低偶然暴露而非单纯增强
  - 接 Rust 前应先做稳健性测试: 去掉最大贡献股票、限制同一股票重复入选次数、或按单票贡献上限做敏感性分析

### [AMV] LTR 新增状态特征精确 A/B

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py`
- **问题**:
  - 用户追问 `宽池 topN 动量强度` 与 `宽池成交额变化` 是否应该保留
  - 需要在同一版代码、同一份数据里做精确 A/B, 避免只用历史 artifact 做近似比较
- **实现**:
  - 新增 `NEW_STATE_FEATURES`:
    - `pool_topn_ret_5d`
    - `pool_topn_ret_20d`
    - `pool_amount_ma5_vs_20`
  - 新增 `OLD_STATE_FEATURES`
  - 新增 variants:
    - `full_old_state`
    - `no_risk_old_state`
    - `kbar_momentum_old_state`
- **产物**:
  - `artifacts/amv_bull_pool_listwise_ranker/20260516_112948/summary.json`
  - 运行 variants: `no_risk`, `no_risk_old_state`, `kbar_momentum_state`, `kbar_momentum_old_state`
- **结果 (`6d/top3`)**:
  - `ltr_no_risk_old_state`: 平均 edge `+2.523pp`, Precision@3 `0.045`
    - 2023 `-0.292pp`, 2024 `+2.731pp`, 2025 `+1.292pp`, 2026 `+6.362pp`
  - `ltr_no_risk`: 平均 edge `+2.304pp`, Precision@3 `0.051`
    - 2023 `+0.054pp`, 2024 `+2.821pp`, 2025 `+2.872pp`, 2026 `+3.471pp`
  - `ltr_kbar_momentum_old_state`: 平均 edge `+2.386pp`, Precision@3 `0.036`
    - 2023 `-0.186pp`, 2024 `+1.665pp`, 2025 `+1.224pp`, 2026 `+6.842pp`
  - `ltr_kbar_momentum_state`: 平均 edge `+1.953pp`, Precision@3 `0.041`
    - 2023 `+0.411pp`, 2024 `+0.910pp`, 2025 `+2.186pp`, 2026 `+4.307pp`
- **结论**:
  - 如果目标是最大化平均 edge / 2026 爆发力, 去掉新增状态特征更强
  - 如果目标是提升年份稳定性和 Precision@3, 保留新增状态特征更好
  - 当前建议: 不删除字段; 将 `old_state` 作为收益上限候选, 将完整状态版作为稳健候选, 后续稳健性测试同时比较两组
  - 接 Rust 前的下一步: 对 `no_risk_old_state`, `kbar_momentum_old_state`, `no_risk`, `kbar_momentum_state` 同时做去最大贡献股票 / 限制同票重复 / 单票贡献上限敏感性

### [AMV] LTR 四候选稳健性测试

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_ltr_selection_analysis.py`
  - 新增 Canvas: `amv-ltr-robustness-test.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-ltr-robustness-test.canvas.tsx`
- **产物**:
  - `artifacts/amv_bull_pool_listwise_ranker/20260516_112948/selection_analysis_20260516_113241/summary.json`
  - 新增 `robustness_2026.csv`
- **稳健性口径**:
  - `remove_top_code`: 删除当年累计贡献最高股票后重算均值
  - `max_repeat`: 同一股票每年最多保留 `3` 次入选
  - `single_pick_cap`: 单笔 `fwd_ret_6d` 上限截断到 `30%`
  - `code_cap`: 单股票年度累计贡献上限封顶到 `100pp`
- **2026 稳健性结果**:
  - `kbar_momentum_old_state`: 原始 `+7.89%`, 去最大票 `+4.50%`, 限制重复 `+6.57%`, 单笔截断 `+5.49%`, 单票封顶 `+5.15%`
  - `no_risk_old_state`: 原始 `+7.41%`, 去最大票 `+3.98%`, 限制重复 `+6.11%`, 单笔截断 `+4.91%`, 单票封顶 `+4.67%`
  - `kbar_momentum_state`: 原始 `+5.36%`, 去最大票 `+3.13%`, 限制重复 `+5.03%`, 单笔截断 `+3.67%`, 单票封顶 `+3.95%`
  - `no_risk`: 原始 `+4.52%`, 去最大票 `+2.29%`, 限制重复 `+4.19%`, 单笔截断 `+2.89%`, 单票封顶 `+3.18%`
- **集中度**:
  - 四个候选的最大贡献股票均为 `sz.002931`
  - `kbar_momentum_old_state`: `sz.002931` 入选 `8` 次, 贡献 `+371.6pp`, 占净收益 `47.6%`
  - `no_risk_old_state`: `sz.002931` 入选 `8` 次, 贡献 `+371.6pp`, 占净收益 `50.6%`
  - 完整状态版通过新增状态特征减少了对 `sz.002931` 的入选次数, 但并未在稳健性口径下反超
- **结论更新**:
  - `old_state` 不只是 2026 爆发更强, 在去最大票、限制重复、收益截断和单票封顶后仍然领先
  - 当前主候选应切到 `kbar_momentum_old_state` 与 `no_risk_old_state`
  - 新增状态特征不删除, 但暂不作为主模型输入; 更适合后续作为 gating / 二次风控条件重新评估
  - 接 Rust 前仍建议加入候选层风控: 同票重复约束、单票贡献监控、以及 top/worst 入场路径复盘

### [AMV] LTR 主候选信号导出

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_ltr_signal_export.py`
- **目的**:
  - 将 LTR 研究结果沉淀为 `bt-amv-topn` 可消费的 `signal.parquet + signal.meta.json`
  - Rust 侧不实现模型逻辑, 只消费 Python 侧已经 OOS 训练并导出的分数/排名
- **实现口径**:
  - 输入: `artifacts/amv_bull_pool_listwise_ranker/20260516_112948/ltr_topn_selected.csv`
  - 输出列对齐 `bt-amv-topn`:
    - `date/code/open_adj/high_adj/low_adj/close_adj/pre_close_adj`
    - `is_bull_regime/amv_mechanical_regime`
    - `score/rank/is_signal`
  - 信号已做 T+1 shift: `signal_date` 为模型选股日, `date` 为次一交易日执行日
  - 导出层支持:
    - `--variants`
    - `--combine-mode separate|union|intersection`
    - `--max-code-repeats`, 当前主候选使用 `3`
  - `separate` 模式已限制只能传入一个 variant, 避免同一 `date/code` 因多个 variant 重复写入 parquet
- **已导出 artifact**:
  - `artifacts/amv_ltr_signals/20260516_113742_separate_kbar_momentum_old_state/signal.meta.json`
    - `signal.parquet`: `7,531,367` 行, `990` 条 T+1 信号, `368` 个执行日
    - label 侧同票最多 3 次后: 全样本年度均值 `+2.85%`; 2026 `+6.57%`, Hit15 `62.64%`, MFE `+20.54%`, MAE `-7.12%`
    - T+1 执行日落入非 bull regime 的信号: `48` 条, 后续 Rust 若 `require_bull_regime=true` 会在执行层过滤
  - `artifacts/amv_ltr_signals/20260516_113806_separate_no_risk_old_state/signal.meta.json`
    - `signal.parquet`: `7,531,367` 行, `918` 条 T+1 信号, `364` 个执行日
    - label 侧同票最多 3 次后: 全样本年度均值 `+3.15%`; 2026 `+6.11%`, Hit15 `53.26%`, MFE `+19.34%`, MAE `-7.05%`
    - T+1 执行日落入非 bull regime 的信号: `39` 条
- **校验**:
  - 已用 Polars 读取两个 `signal.parquet` 核验行数、信号数、日期范围和 `rank` 类型
  - `uv run ruff check scripts/amv_ltr_signal_export.py` 通过
- **下一步**:
  - 用 `bt-amv-topn` 对上述两个 signal artifact 跑真实回测
  - 回测配置建议沿用当前 `6td + no stop`, 并保留 `require_bull_regime=true` 作为执行层二次确认

### [AMV] LTR 主候选 Rust 真实回测首轮

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `backtest-engine/crates/amv-topn/config_6td_no_stop.toml`
  - 新增 Canvas: `amv-ltr-rust-backtest.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-ltr-rust-backtest.canvas.tsx`
- **回测口径**:
  - 引擎: `bt-amv-topn`
  - 配置: `6td + no stop`
  - 执行: T+1 open buy, 6 trading days close sell
  - 执行层仍保留:
    - `require_bull_regime=true`
    - 涨停不可买
    - 跌停不可卖
    - 交易成本: commission `0.025%`, stamp duty `0.1%`, slippage `0.1%`
- **回测产物**:
  - `kbar_momentum_old_state`:
    - signal: `artifacts/amv_ltr_signals/20260516_113742_separate_kbar_momentum_old_state/signal.meta.json`
    - report: `artifacts/amv_ltr_signals/20260516_113742_separate_kbar_momentum_old_state/backtests/6td_no_stop_20260516_114038_751/report.json`
    - analysis: `artifacts/amv_ltr_signals/20260516_113742_separate_kbar_momentum_old_state/backtests/analysis_20260516_114132/summary.json`
  - `no_risk_old_state`:
    - signal: `artifacts/amv_ltr_signals/20260516_113806_separate_no_risk_old_state/signal.meta.json`
    - report: `artifacts/amv_ltr_signals/20260516_113806_separate_no_risk_old_state/backtests/6td_no_stop_20260516_114054_426/report.json`
    - analysis: `artifacts/amv_ltr_signals/20260516_113806_separate_no_risk_old_state/backtests/analysis_20260516_114133/summary.json`
- **真实回测结果**:
  - `kbar_momentum_old_state`: Total Return `-66.97%`, Gross `-55.36%`, MaxDD `74.93%`, Trades `166`, WinRate `43.37%`
  - `no_risk_old_state`: Total Return `-0.01%`, Gross `+20.55%`, MaxDD `52.93%`, Trades `165`, WinRate `43.03%`
  - 当前规则基线 `6td + no stop`: Total Return `+144.02%`, Gross `+195.18%`, MaxDD `14.71%`, Trades `264`, WinRate `52.27%`
- **年度路径**:
  - `kbar_momentum_old_state`: 2023 `-37.97%`, 2024 `-35.95%`, 2025 `-7.44%`, 2026 YTD `-9.62%`
  - `no_risk_old_state`: 2023 `+1.02%`, 2024 `+47.51%`, 2025 `-27.52%`, 2026 YTD `-7.63%`
- **标签兑现诊断**:
  - 原 LTR 标签是 signal-date close-to-close `fwd_ret_6d`
  - 真实执行是 next-day open buy, 6td close sell
  - 将样本改成“执行日 open -> h 日 close”后, edge 基本塌缩:
    - `kbar`: 1td 全样本均值 `+0.16%`, 6td `+0.13%`; 2026 1td `+2.16%`, 6td `-0.01%`
    - `no_risk`: 1td 全样本均值 `+0.23%`, 6td `-0.61%`; 2026 1td `+2.22%`, 6td `-1.76%`
  - LTR 当前学到的更像“信号日收盘后次日短线/冲高”而不是可持有 6td 的执行口径 alpha
- **结论更新**:
  - 当前 `6d close-to-close` LTR 不能直接用于 Rust 回测
  - 问题不是 Rust 接入本身, 而是训练标签与真实执行口径错配
  - 下一步应重做 LTR 标签:
    - label 改成 `T+1 open -> T+N close`
    - 训练样本中加入“次日涨停不可买”过滤
    - 优先测试 `1td/2td/3td` 执行口径标签, 而不是继续沿用当前 `6d close-to-close`
- **Rotation same-day close 理论上限诊断**:
  - 目的: 检查当前 LTR 如果改用 Rotation 的“当日 close 成交”口径是否存在明显上限惊喜
  - 临时产物: `artifacts/amv_ltr_rotation_theory/20260516_135751/`
  - 生成方式: 用已导出的 LTR signal 反向回到 `signal_date`, 不做 T+1 shift; 非 top3 股票设为 `score=0/rank=9999/is_top_n=false`
  - 口径 A: `same-day close + rank-drop rotation + no stop`
    - config: `rotation_theory_top3_close.toml`
    - `kbar_momentum_old_state`: Total Return `-67.90%`, Gross `-21.46%`, MaxDD `76.03%`, Trades `551`
    - `no_risk_old_state`: Total Return `-52.17%`, Gross `-16.16%`, MaxDD `58.74%`, Trades `370`
    - 解读: 当前 LTR 是 6d 标签, strict rank-drop 近似变成 1 日高换手, 成本与噪音很重
  - 口径 B: `same-day close + no rank-drop + max_hold_days=10 calendar + no stop`
    - config: `rotation_theory_top3_close_hold10_no_rankdrop.toml`
    - `kbar_momentum_old_state`: Total Return `-48.88%`, Gross `-34.37%`, MaxDD `69.01%`, Trades `158`
    - `no_risk_old_state`: Total Return `-6.99%`, Gross `+10.91%`, MaxDD `44.64%`, Trades `145`
    - 解读: `no_risk` 在 same-day close + hold10 的毛收益为正, 但扣成本后仍为负, 且远弱于规则基线; `kbar` 无理论上限惊喜
  - 结论: 直接复用当前 Rotation close 口径不能挽救这版 LTR; 最大问题仍然是训练标签/可交易约束与执行口径不匹配, 尤其涨停不可买和高开兑现损失

### [AMV] LTR 执行口径标签校准

- 已更新:
  - `scripts/amv_bull_pool_ranker_lab.py`
  - `scripts/amv_bull_pool_listwise_ranker_lab.py`
  - 本文件 (本条)
  - `project-status.md`
- **实现口径**:
  - 新增 `--label-mode next_open_to_close`
  - 新增 `--execution-lag-days` (默认 `1`)
  - 新增 `--exclude-limit-up-entry`
  - 新增 `--price-limit-tolerance` (默认 `0.001`)
  - 执行标签定义为: `signal_date + 1td open` 买入, `execution_lag_days + horizon` 对应交易日 close 卖出
  - MFE / MAE 使用执行入场 open 到持有窗口内 high / low
  - 次日涨停不可买样本可在训练集和评估集统一剔除
- **重要修正**:
  - 初版实现曾在已过滤 AMV bull pool 后再做 `shift`, 会跳过非候选交易日, 导致执行标签被高估
  - 该错误结果已作废:
    - `artifacts/amv_bull_pool_listwise_ranker/20260516_141014`
    - `artifacts/amv_bull_pool_listwise_ranker/20260516_141041`
    - `artifacts/amv_bull_pool_listwise_ranker/20260516_141110`
  - 正确实现已移动到底层 `build_dataset()`: 在完整日线序列上先计算未来 open/high/low/close/pre_close, 再过滤 AMV bull pool / 市值 / 成交额
- **正式修正后实验**:
  - 公共参数:
    - `--label-mode next_open_to_close`
    - `--exclude-limit-up-entry`
    - `--variants no_risk_old_state,kbar_momentum_old_state`
  - `1td` run: `artifacts/amv_bull_pool_listwise_ranker/20260516_141305`
    - `baseline_ret_5d`: avg edge `+0.662pp`, avg mean `+0.924%`
    - `ltr_kbar_momentum_old_state`: avg edge `+0.571pp`, avg mean `+0.833%`
    - `ltr_no_risk_old_state`: avg edge `+0.518pp`, avg mean `+0.780%`
  - `2td` run: `artifacts/amv_bull_pool_listwise_ranker/20260516_141334`
    - `baseline_ret_5d`: avg edge `+0.686pp`, avg mean `+1.075%`
    - `ltr_kbar_momentum_old_state`: avg edge `+0.454pp`, avg mean `+0.842%`
    - `ltr_no_risk_old_state`: avg edge `+0.446pp`, avg mean `+0.835%`
  - `3td` run: `artifacts/amv_bull_pool_listwise_ranker/20260516_141402`
    - `baseline_ret_5d`: avg edge `+0.660pp`, avg mean `+1.146%`
    - `ltr_no_risk_old_state`: avg edge `+0.261pp`, avg mean `+0.747%`
    - `ltr_kbar_momentum_old_state`: avg edge `+0.259pp`, avg mean `+0.745%`
- **结论更新**:
  - 执行标签校准后, 当前两个 old_state LTR 变体没有超过简单 `ret_5d` 基线
  - 原先 LTR 的高 edge 主要来自不可交易 / 错位标签, 不是稳定可执行 alpha
  - 短线执行口径下, `ret_5d` 本身是更强的候选方向; 后续更应该围绕 `ret_5d` 做执行层 Rust 回测、风控和组合约束, 而不是继续强化当前 LTR old_state

### [AMV] Regime-aware 因子袖子切换实验

- 已更新:
  - 新增 `scripts/amv_bull_pool_regime_sleeve_lab.py`
  - 新增 Canvas: `amv-regime-sleeve-lab.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-regime-sleeve-lab.canvas.tsx`
  - 本文件 (本条)
  - `project-status.md`
- **实验目的**:
  - 不再让 LTR 直接从全股票池选 Top3
  - 改为评估“因子袖子”是否存在可切换空间:
    - `ret_5d`
    - `ret_20d`
    - `klen`
    - `kmid2`
    - `kbar`
    - `kbar_momentum`
    - `manual_p2_k0p5_r0`
    - `manual_p3_k0p5_r0`
  - 用执行口径标签: `T+1 open -> T+N close`, 并剔除次日涨停不可买样本
  - 对比:
    - 静态单袖子
    - 每日 oracle (事后每天选最佳袖子, 仅看上限)
    - `state_classifier` (只用旧状态特征预测当天该选哪个袖子)
    - `train_best_sleeve` (训练期/验证期平均最优袖子)
- **产物**:
  - `1td`: `artifacts/amv_bull_pool_regime_sleeve/20260516_142833`
  - `2td`: `artifacts/amv_bull_pool_regime_sleeve/20260516_142854`
  - `3td`: `artifacts/amv_bull_pool_regime_sleeve/20260516_142915`
- **核心结果**:
  - `1td`:
    - `daily_oracle`: avg edge `+5.138pp`
    - `static_ret_5d`: `+0.662pp`
    - `static_ret_20d`: `+0.593pp`
    - `state_classifier`: `+0.538pp`
  - `2td`:
    - `daily_oracle`: avg edge `+6.138pp`
    - `static_ret_20d`: `+0.705pp`
    - `static_ret_5d`: `+0.686pp`
    - `state_classifier`: `+0.242pp`
  - `3td`:
    - `daily_oracle`: avg edge `+6.991pp`
    - `static_ret_5d`: `+0.660pp`
    - `static_ret_20d`: `+0.483pp`
    - `state_classifier`: `+0.091pp`
- **年度拆解**:
  - oracle 每年都强:
    - `1td`: 2023 `+4.441pp`, 2024 `+4.899pp`, 2025 `+5.371pp`, 2026 `+5.842pp`
    - `2td`: 2023 `+5.164pp`, 2024 `+6.156pp`, 2025 `+6.019pp`, 2026 `+7.214pp`
    - `3td`: 2023 `+5.820pp`, 2024 `+7.257pp`, 2025 `+7.387pp`, 2026 `+7.500pp`
  - `state_classifier` 不稳:
    - `1td`: 2023 `+1.381pp`, 2024 `-0.015pp`, 2025 `+0.494pp`, 2026 `+0.291pp`
    - `2td`: 2023 `+0.423pp`, 2024 `-0.779pp`, 2025 `+0.637pp`, 2026 `+0.689pp`
    - `3td`: 2023 `+0.460pp`, 2024 `-0.827pp`, 2025 `+0.123pp`, 2026 `+0.609pp`
- **模型选择分布**:
  - `state_classifier` 基本只在 `ret_5d` / `ret_20d` 间切换:
    - `1td`: `ret_20d` 361 天, `ret_5d` 13 天
    - `2td`: `ret_20d` 200 天, `ret_5d` 173 天
    - `3td`: `ret_5d` 322 天, `ret_20d` 50 天
- **结论更新**:
  - “因子袖子切换”存在很高事后上限, 所以方向没有错
  - 当前状态特征 + 简单 classifier 没学到可交易切换, 不能直接进入 Rust 回测
  - 下一步不应急着做动态权重实盘信号, 而应先研究 oracle 可预测性:
    - oracle 当天选择了哪些袖子
    - oracle 选择是否能被 AMV 阶段、宽池赚钱效应、动量强度、涨停/高开风险解释
    - 如果可解释, 再做规则化 gating; 如果不可解释, 切换更像噪声

### [AMV] Oracle 袖子 Rust 上限回测

- 已更新:
  - 新增 `scripts/amv_oracle_sleeve_signal_export.py`
  - 新增 `backtest-engine/crates/amv-topn/config_1td_no_stop.toml`
  - 新增 `backtest-engine/crates/amv-topn/config_2td_no_stop.toml`
  - 新增 `backtest-engine/crates/amv-topn/config_3td_no_stop.toml`
  - 新增 Canvas: `amv-oracle-sleeve-rust-backtest.canvas.tsx`
  - 归档 Canvas: `reports/canvases/amv-oracle-sleeve-rust-backtest.canvas.tsx`
  - 本文件 (本条)
  - `project-status.md`
- **重要警告**:
  - oracle 是“事后诸葛亮”, 使用未来收益选择当天最优 sleeve
  - 该回测只表示可交易约束下的理论天花板, 不能作为实盘策略
- **信号产物**:
  - `1td`: `artifacts/amv_oracle_sleeve_signals/20260516_143515_oracle_1td/signal.meta.json`
  - `2td`: `artifacts/amv_oracle_sleeve_signals/20260516_143558_oracle_2td/signal.meta.json`
  - `3td`: `artifacts/amv_oracle_sleeve_signals/20260516_143629_oracle_3td/signal.meta.json`
- **Rust 回测产物**:
  - `1td`: `artifacts/amv_oracle_sleeve_signals/20260516_143515_oracle_1td/backtests/1td_no_stop_20260516_143827_653/report.json`
  - `2td`: `artifacts/amv_oracle_sleeve_signals/20260516_143558_oracle_2td/backtests/2td_no_stop_20260516_143832_676/report.json`
  - `3td`: `artifacts/amv_oracle_sleeve_signals/20260516_143629_oracle_3td/backtests/3td_no_stop_20260516_143837_004/report.json`
- **Rust 真实约束口径**:
  - T+1 open 买入
  - `max_hold_trading_days = 1/2/3`
  - no stop
  - `require_bull_regime=true`
  - 涨停不可买 / 跌停不可卖
  - 成本: commission `0.025%`, stamp `0.1%`, slippage `0.1%`
- **回测结果**:
  - `1td oracle`:
    - Total Return `+15,806,636%`
    - Gross Return `+16,785,773%`
    - MaxDD `31.76%`
    - Trades `825`
    - WinRate `73.58%`
    - Avg trade PnL `+4.59%`
  - `2td oracle`:
    - Total Return `+891,251%`
    - Gross Return `+945,000%`
    - MaxDD `37.37%`
    - Trades `571`
    - WinRate `70.75%`
    - Avg trade PnL `+5.12%`
  - `3td oracle`:
    - Total Return `+639,900%`
    - Gross Return `+671,457%`
    - MaxDD `41.78%`
    - Trades `436`
    - WinRate `71.79%`
    - Avg trade PnL `+6.49%`
- **年度路径**:
  - `1td`: 2021 `+746.8%`, 2022 `+489.3%`, 2023 `+1078.3%`, 2024 `+899.7%`, 2025 `+755.2%`, 2026 YTD `+206.2%`
  - `2td`: 2021 `+446.7%`, 2022 `+357.5%`, 2023 `+446.9%`, 2024 `+483.5%`, 2025 `+436.7%`, 2026 YTD `+106.3%`
  - `3td`: 2021 `+552.6%`, 2022 `+179.2%`, 2023 `+487.7%`, 2024 `+443.4%`, 2025 `+459.8%`, 2026 YTD `+95.5%`
- **结论更新**:
  - 即使加入真实执行约束和交易成本, oracle sleeve 仍有巨大上限
  - 这强烈说明“因子袖子切换”不是没有空间, 问题在于我们当前还无法提前预测切换
  - 后续研究重点应转为 oracle 可解释性:
    - 哪些状态下 oracle 选择 `ret_5d` / `ret_20d` / `kmid2`
    - 选择前是否存在可观测先验, 如 AMV bull 阶段、宽池赚钱效应、涨停/高开风险、市场宽度
    - 若有稳定先验, 再落地成规则 gating 或弱模型; 若无, oracle 主要是噪声/未来函数

### [AMV] Bull Pool 因子分年份 / 分 regime 标签分析

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_factor_regime_analysis.py`
  - 新增 Canvas: `amv-bull-pool-factor-regime.canvas.tsx`
- **实现说明**:
  - 脚本已按项目偏好使用 Polars, 未使用 pandas
  - 分析 10 个 AMV TopN 相关因子 + 当前组合, 统一使用 `6d` 标签和 `top3`
  - 指标: 日度 Spearman IC, top3 edge, Hit15 edge
  - 分组: 年份, AMV bull 初/中/后期, 近 5 日宽池赚钱效应强/弱, 未来宽池收益强/弱
- **产物**:
  - `artifacts/amv_bull_pool_factor_regime/20260510_221119/summary.json`
  - pool: `636,042` 行, `2021-04-20 -> 2026-04-27`, `555` 个 AMV bull 交易日, `2,335` 只股票
- **全样本 top3 edge**:
  - `K线振幅收缩(KLEN)`: `+1.16pp`, IC `+0.0499`
  - `实体占比偏强(KMID2)`: `+1.06pp`, IC `-0.0104`
  - `接近20日新高`: `+0.86pp`, IC `+0.0107`
  - 当前组合 `P2/K0.5/R0`: `+0.82pp`, IC `+0.0035`
  - `5日动量`: `+0.75pp`, IC `-0.0223`
  - `20日高位`: `+0.35pp`
  - `换手放大`: `-0.82pp`
- **分年份重点**:
  - 2024: `KLEN` edge `+2.44pp`, 当前组合 `+2.12pp`
  - 2025: `KMID2` edge `+2.17pp`, `接近20日新高` edge `+2.03pp`, 当前组合 `-0.75pp`
  - 2026: `5日动量` edge `+4.13pp`, `20日动量` edge `+4.00pp`, 当前组合仅 `+0.05pp`
- **分 AMV bull 阶段**:
  - 初期: `KLEN +1.98pp`, 当前组合 `+1.68pp`
  - 中期: `KLEN +1.23pp`, 当前组合仅 `+0.10pp`
  - 后期: `5日动量 +2.24pp`, `KMID2 +2.07pp`, 当前组合 `+1.18pp`
- **结论**:
  - 当前组合不是最稳的单一信号; `KLEN` 和 `KMID2` 更像稳定核心
  - 2026 的有效因子显著切换到动量, 这解释了当前固定权重组合在 2026-04 edge 几乎归零
  - 多数因子 IC 不强但 top3 edge 明显, 说明问题更偏“极值 topN 选择 / 状态条件”而不是简单线性 IC
  - 对 LTR 的启发: LTR 值得做, 但必须加入市场状态特征, 学习何时偏 `KLEN/KMID2`, 何时偏动量, 而不是只让模型学固定个股因子排序

### [AMV] Bull Pool 因子年度稳定性补充

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_yearly_factor_analysis.py`
  - 新增 Canvas: `amv-bull-pool-yearly-factor.canvas.tsx`
- **问题背景**:
  - 当前 `6td + no stop` Rust 回测显示收益高度依赖 2024 大赢家
  - 需要回到接入 Rust 前的 AMV bull pool 排序实验, 补充分年份评价, 检查当前因子/权重是否只是更适配 2024 行情
- **产物**:
  - `artifacts/amv_bull_pool_yearly_factor/20260510_150556/summary.json`
  - 对当前组合 `高位+K线确认 P2/K0.5/R0`, 四个组成因子, 以及 36 组权重网格按年重算
  - 主观察口径为 `top3 / 6d horizon`, 同时输出 `5d/10d/20d`
- **重要口径差异与修正**:
  - 首轮多 horizon (`5/6/10/20d`) yearly factor lab 的 pool 日期为 `2021-04-20 -> 2026-02-02`
  - 原因不是 AMV 数据缺失: AMV 原始数据/机械 regime 都到 `2026-05-08`, QMT 行情也到 `2026-05-08`
  - 真正原因是脚本复用 `build_dataset()` 时使用 `max(args.horizons)=20`, 全局过滤 `fwd_ret_20d is not null`; 在当前 QMT 截止日下 20d 前瞻样本最多到 `2026-04-07`, 而 2026-04 的 AMV bull 从 `2026-04-09` 才重新开始, 因此多 horizon pool 最晚 bull 日期停在上一段 bull 的 `2026-02-02`
  - 已补跑 6d-only: `artifacts/amv_bull_pool_yearly_factor/20260510_151739/summary.json`, pool 日期为 `2021-04-20 -> 2026-04-27`
- **当前组合 6d 分年份 (以 6d-only 修正版为准)**:
  - 2021: 单笔均值 `+1.10%`, 相对随机 `+0.40pp`
  - 2022: 单笔均值 `+1.14%`, 相对随机 `+0.77pp`
  - 2023: 单笔均值 `+0.64%`, 相对随机 `+1.25pp`
  - 2024: 单笔均值 `+3.30%`, 相对随机 `+2.12pp`
  - 2025: 单笔均值 `+0.63%`, 相对随机 `-0.75pp`
  - 2026: 单笔均值 `+1.10%`, 相对随机 `+0.05pp`, 覆盖到 `2026-04-27`, 样本日 `33`
- **组成因子 6d 稳定性**:
  - 当前组合: 全样本 edge `+0.82pp`, `5/6` 年正 edge
  - `20日高位`: 全样本 edge `+0.35pp`, `5/6` 年正 edge, 2026 为 `-0.14pp`
  - `接近20日新高`: 全样本 edge `+0.87pp`, `5/6` 年正 edge, 2026 为 `-0.53pp`
  - `K线振幅收缩`: 全样本 edge `+1.16pp`, `6/6` 年正 edge, 2026 为 `+2.80pp`
  - `实体占比偏强`: 全样本 edge `+1.06pp`, `6/6` 年正 edge, 2026 为 `+1.56pp`
- **权重网格启发**:
  - 当前不是明确“权重天花板”: `P3/K0.5/R0` 在 6d-only 全样本 edge `+0.93pp`, 略优于当前 `P2/K0.5/R0` 的 `+0.82pp`
  - 加风险项的组合没有出现在 6d 稳定性前列, 暂时不支持直接加 `atr/panic` 权重来解决问题
- **结论**:
  - 用户提出的“当前因子更适配 2024”成立: 2024 的 edge 明显高于其他完整年份
  - 但不能简单说组合天花板已到; 更准确的判断是“同一类高位+K线确认框架仍有小幅权重优化空间, 但 2026-04 把当前组合 edge 压到几乎为零”
  - 下一步如果继续做因子层研究, 应用 6d-only / per-horizon 数据集重跑权重网格, 避免 20d 前瞻过滤污染短周期结论; 如果继续做交易规则, 应测试执行日收弱过滤与短期赚钱效应确认

### [AMV] TopN 6td 分段归因: 2024 大赢家 vs 2026 亏损段

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_topn_segment_analysis.py`
  - 新增 Canvas: `amv-topn-segment-attribution.canvas.tsx`
- **分段归因产物**:
  - `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_125746_630/segment_analysis_20260510_130534/summary.json`
  - `enriched_trades.csv`: 每笔交易补充 `score/rank`, 入场 gap, 入场日表现, AMV regime, 市值/成交额, 持有期 `MFE/MAE`
  - `segment_summary.csv`, `monthly_summary.csv`, `top_2024_winners.csv`, `trades_2026.csv`, `loss_2026_worst.csv`
- **核心对照**:
  - 全样本: `264` 笔, PnL `+72.01 万`, 胜率 `52.27%`, 单笔均值 `+1.12%`, 平均 `MFE +5.11%`, `MAE -3.15%`
  - 2024 盈利交易: `25` 笔, PnL `+60.72 万`, 单笔均值 `+8.28%`, 平均 `MFE +14.72%`, `MAE -2.67%`
  - 2024 Top10 盈利: PnL `+51.13 万`, 单笔均值 `+17.10%`, 平均 `MFE +26.81%`, 平均入场 gap `+2.26%`
  - 2026 全部: `18` 笔, PnL `-11.76 万`, 胜率 `33.33%`, 单笔均值 `-1.52%`, 平均 `MFE +1.77%`, `MAE -4.20%`
  - 2026 亏损交易: `12` 笔, PnL `-16.30 万`, 单笔均值 `-3.13%`, 平均 `MFE +1.01%`, `MAE -5.41%`, 入场日表现 `-1.30%`
- **月份结论**:
  - 2024-10 是最大正贡献月份: `6` 笔, PnL `+30.89 万`, 单笔均值 `+17.09%`
  - 2024-09 次强: `9` 笔, PnL `+11.06 万`
  - 2026-04 是最大亏损月份: `9` 笔, PnL `-10.26 万`, 胜率 `11.11%`
- **解释**:
  - 2024 大赢家并不是 `score/rank` 明显更强, 而是持有期内给了极高 MFE; 本质更像 AMV bull 叠加行情窗口
  - 2026 亏损段的 `score` 仍接近全样本, 但入场后几乎没有上冲空间, 且入场日平均走弱; 这说明当前排序信号在弱赚钱效应阶段无法区分“仍在高位但已经没弹性”的票
- **下一步建议**:
  - 不优先继续扫 `gap` 阈值, 因为 2026 平均 gap 并不异常
  - 优先尝试入场确认 / 环境确认: 如执行日不能明显收弱, 或在 AMV bull 中叠加短期市场宽度/赚钱效应确认

### [AMV] TopN 6td 无止损交易归因

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `backtest-engine/crates/amv-topn/src/components.rs`
  - `backtest-engine/crates/amv-topn/src/main.rs`
  - `backtest-engine/crates/amv-topn/src/systems.rs`
  - `scripts/amv_topn_enhancement_sweep.py`
  - 新增 `scripts/amv_topn_trade_analysis.py`
  - 新增 Canvas: `amv-topn-6td-trade-analysis.canvas.tsx`
- **新增回测产物**:
  - Sweep: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_125746_630/summary.json`
  - 6td 无止损: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_125746_630/stop_off_hold_6d/`
  - 6td + 5% stop: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_125746_630/hold_6d/`
  - 交易归因: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_125746_630/analysis_20260510_125808/summary.json`
- **工程修正**:
  - `bt-amv-topn` 回测报告目录新增 `trades.csv`, 记录完整平仓明细
  - `bt-amv-topn` 回测报告目录新增 `daily_equity.csv`, 记录每日 cash / 持仓市值 / 总权益
  - `amv_topn_enhancement_sweep.py` 运行 Cargo 时自动移除 `CARGO_TARGET_DIR`, 避免 Cursor 沙箱 release 构建权限问题
- **6td 对照结果**:
  - `6td + no stop`: 净收益 `+144.02%`, `MaxDD -14.71%`, 胜率 `52.27%`, 交易数 `264`
  - `6td + 5% stop`: 净收益 `+39.64%`, `MaxDD -17.33%`, 胜率 `48.20%`, 交易数 `278`
- **归因结论**:
  - 年度收益: 2021 `+0.48%`, 2022 `+38.36%`, 2023 `+14.18%`, 2024 `+49.00%`, 2025 `+12.53%`, 2026 YTD `-8.79%`
  - 单笔分布: 均值 `+1.12%`, 中位数 `+0.19%`, P10 `-4.74%`, P90 `+6.33%`, payoff ratio `1.56`
  - Top10 盈利占正收益 `40.45%`, 占净 PnL `97.74%`; Top10 亏损占总亏损 `27.19%`
  - 5% 止损同入口匹配到 `29` 笔, 其中 `18` 笔后续反弹优于止损退出, 错杀比例 `62.07%`, 合计 PnL 差额约 `+18.56 万`
- **当前判断**:
  - `6td + no stop` 可作为新基线, 但收益集中度偏高, 需要继续检查 2024 大赢家与 2026 亏损段
  - 下一步优先做分段归因: 大赢家的 AMV 位置 / 买入 gap / 行业主题集中度, 以及 2026 亏损段是否需要过热过滤或弱市降仓

### [AMV] TopN 持有期改为交易日口径并重跑增强消融

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `backtest-engine/crates/amv-topn/src/components.rs`
  - `backtest-engine/crates/amv-topn/src/resources.rs`
  - `backtest-engine/crates/amv-topn/src/systems.rs`
  - `backtest-engine/crates/amv-topn/src/main.rs`
  - `backtest-engine/crates/amv-topn/config*.toml`
  - `scripts/amv_topn_enhancement_sweep.py`
  - Canvas: `amv-topn-enhancement-sweep.canvas.tsx`
- **重要修正**:
  - 旧 `max_hold_days` 使用 `NaiveDate` 差值, 实际是自然日
  - AMV TopN 已删除旧字段, 改为 `max_hold_trading_days`
  - Rust 侧使用全市场 `trading_dates` 生成 `date_index`, 买入时记录 `entry_trade_index`, 卖出时按交易日索引差值判断持有期
- **交易日口径消融 artifact**:
  - 快速核验: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_122252_006/summary.json`
  - 完整消融: `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_122315_991/summary.json`
- **核心结果 (交易日口径)**:
  - `baseline_10d` (`10td + 5% stop`): 净收益 `+25.67%`, `MaxDD -22.21%`, 胜率 `44.28%`, 交易数 `201`
  - `stop_off` (`10td + no stop`): 净收益 `+50.30%`, `MaxDD -18.85%`, 胜率 `47.54%`, 交易数 `183`
  - `stop_off_hold_5d` (`5td + no stop`): 净收益 `+21.56%`, `MaxDD -28.46%`, 胜率 `44.22%`, 交易数 `303`
  - `stop_off_hold_6d` (`6td + no stop`): 净收益 `+144.02%`, `MaxDD -14.71%`, 胜率 `52.27%`, 交易数 `264`
  - `stop_off_hold_7d` (`7td + no stop`): 净收益 `+99.68%`, `MaxDD -14.95%`, 胜率 `47.16%`, 交易数 `229`
  - `stop_off_trailing_8_4`: 净收益 `+78.91%`, `MaxDD -16.94%`
  - `stop_off_bear_exit`: 净收益 `+56.69%`, `MaxDD -19.52%`
  - `stop_3pct`: 净收益 `-5.89%`, `MaxDD -26.06%`
- **结论**:
  - 自然日口径的 `stop_off +224.23%` 已废弃, 不再作为当前判断依据
  - 交易日口径下, 关固定止损仍显著优于 `5% stop`, 但幅度回归正常
  - 当前新主线从 `10d no stop` 调整为 `6td no stop`
  - 自然日持有期异常强势值得单独研究, 但后续应显式新增 `max_hold_calendar_days`, 不恢复有歧义的旧 `max_hold_days`
  - 下一步应输出完整交易明细, 做年度收益/回撤、单笔分布、最大盈利贡献占比、最大亏损、以及 `stop_off vs 5% stop` 的错杀分析

### [AMV] TopN 增强消融首轮完成

- **状态更新**: 本节使用旧自然日持有期口径, 已被上一节交易日口径重跑结果取代, 不再作为当前策略判断依据。
- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `backtest-engine/crates/amv-topn/src/resources.rs`
  - `backtest-engine/crates/amv-topn/src/systems.rs`
  - `backtest-engine/crates/amv-topn/src/main.rs`
  - `backtest-engine/crates/amv-topn/config*.toml`
  - 新增 `scripts/amv_topn_enhancement_sweep.py`
  - 新增 Canvas: `amv-topn-enhancement-sweep.canvas.tsx`
- **新增规则开关**:
  - `entry.max_open_gap_pct`: 执行日高开过滤, 用于跳过开盘涨幅过高的追高买入
  - `exit.sell_on_bear_regime`: AMV 非 bull 时主动退出已有持仓
  - sweep 脚本自动生成临时 TOML, 逐个调用 `bt-amv-topn`, 并汇总 `report.json`
- **最新信号**:
  - `artifacts/amv_topn/20260510_115834/signal.parquet`
  - 日期范围 `2021-01-04 -> 2026-05-08`
  - 执行信号日 `560`, 执行信号行 `1679`
- **首轮消融 artifact**:
  - `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_115853_773/summary.json`
  - `artifacts/amv_topn/20260510_115834/backtests/enhancement_20260510_120008_207/summary.json`
- **核心结果**:
  - `baseline_10d`: 净收益 `+64.74%`, `MaxDD -14.69%`, 胜率 `54.55%`, 交易数 `242`
  - `stop_off`: 净收益 `+224.23%`, `MaxDD -12.69%`, 胜率 `58.74%`, 交易数 `223`
  - `stop_8pct`: 净收益 `+107.53%`, `MaxDD -13.07%`
  - `stop_3pct`: 净收益 `+18.96%`, `MaxDD -24.98%`
  - `stop_off_trailing_8_4`: 净收益 `+215.01%`, `MaxDD -15.19%`
  - `stop_off_trailing_10_5`: 净收益 `+214.77%`, `MaxDD -14.61%`
  - `stop_off_gap_5pct`: 净收益 `+182.35%`, `MaxDD -12.66%`
  - `stop_off_bear_exit`: 净收益 `+152.58%`, `MaxDD -11.41%`
- **结论**:
  - 固定 `5%` 止损是当前最大拖累, 关掉后收益和回撤同时改善
  - `3%` 止损明显过紧; `8%` 止损好于 `5%`, 但仍显著弱于无固定止损
  - 高开过滤、AMV 转空主动清仓、移动止盈均降低收益; 其中 AMV 转空清仓能降回撤, 但收益牺牲较大
  - 下一步应优先对 `stop_off` 做年度拆分与风险暴露检查, 再决定是否需要更温和的风险退出

## 2026-05-03

### [AMV] TopN Rust 真实回测首轮完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_export_signals.py`
  - 新增 `scripts/amv_topn_backtest.py`
  - 新增 Rust crate: `backtest-engine/crates/amv-topn`
  - 新增 Canvas: `amv-topn-rust-backtest.canvas.tsx`
- **命名决定**:
  - crate 名称使用 `bt-amv-topn`
  - 目录为 `backtest-engine/crates/amv-topn`
  - 含义明确为 `AMV gate + topN 排序策略`, 避免和 AMV 数据/OCR 管道混淆
- **信号口径**:
  - 固定研究阶段最优信号: `top3 高位+K线确认 P2/K0.5/R0`
  - `T` 日收盘计算横截面排序信号, `T+1` 开盘买入
  - `AMV bear / 非 bull` 只阻止新开仓, 不强制已有仓位清仓
  - 使用执行日开盘涨停过滤, 收盘跌停禁止卖出
- **artifact**:
  - 信号: `artifacts/amv_topn/20260503_221922/signal.parquet`
  - Meta: `artifacts/amv_topn/20260503_221922/signal.meta.json`
  - 回测:
    - `artifacts/amv_topn/20260503_221922/backtests/5d_20260503_221942_965/report.json`
    - `artifacts/amv_topn/20260503_221922/backtests/10d_20260503_221948_247/report.json`
    - `artifacts/amv_topn/20260503_221922/backtests/20d_20260503_221951_651/report.json`
- **核心结果**:
  - `5d`: 净收益 `+29.88%`, 最大回撤 `-35.99%`, 胜率 `49.48%`, 交易数 `388`
  - `10d`: 净收益 `+65.54%`, 最大回撤 `-14.69%`, 胜率 `54.58%`, 交易数 `240`
  - `20d`: 净收益 `+10.26%`, 最大回撤 `-28.01%`, 胜率 `44.24%`, 交易数 `165`
- **结论**:
  - `10d` 是当前真实回测口径下最优档位: 收益最高、回撤最低、胜率最高
  - `5d` 毛收益高但换手过快, 成本和路径回撤显著侵蚀净收益
  - `20d` 研究口径单笔均值更高, 但真实组合路径明显变差, 暂不适合作为主交易周期
  - 下一步优先围绕 `10d` 细化退出: 止损开关、AMV 转空是否主动清仓、风险因子过滤/仓位控制

### [AMV] Bull 宽池 Top3 持有期兑现曲线完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_horizon_curve.py`
  - 新增 Canvas: `amv-bull-pool-horizon-curve.canvas.tsx`
- **实验目的**:
  - 固定当前候选信号 `top3 高位+K线确认 P2/K0.5/R0`
  - 只改变观察窗口 `1/2/3/5/10/15/20/30d`, 判断 `20d` 是否偏长
- **artifact**:
  - `artifacts/amv_bull_pool_horizon_curve/20260503_215103/summary.json`
- **核心结果**:
  - `1d`: 单笔均值 `+0.413%`, 相对随机 `+0.340pp`, 日均收益 `+0.413%`, rolling NAV `+746.31%`, `MaxDD -21.45%`
  - `5d`: 单笔均值 `+1.304%`, 相对随机 `+0.906pp`, 日均收益 `+0.261%`, rolling NAV `+287.31%`, `MaxDD -4.05%`
  - `10d`: 单笔均值 `+1.789%`, 相对随机 `+0.943pp`, 日均收益 `+0.179%`, rolling NAV `+153.04%`, `MaxDD -4.27%`
  - `20d`: 单笔均值 `+2.625%`, 相对随机 `+1.276pp`, 日均收益 `+0.131%`, rolling NAV `+102.97%`, `MaxDD -2.90%`
  - `30d`: 单笔均值 `+3.173%`, 相对随机 `+1.581pp`, 日均收益 `+0.106%`, rolling NAV `+72.07%`, `MaxDD -2.46%`
- **结论**:
  - 绝对收益到 `30d` 仍在增加, 说明没有明显 `20d` 前回吐
  - 但单位时间收益从 `1d` 起持续下降, `20d/30d` 资金效率明显低于 `5d/10d`
  - 真实回测不应只押 `20d`; 优先做 `5d / 10d / 20d` 三档持有期对照, 并加入 AMV 转空/排名跌出/止损止盈等退出规则

### [Research] Learning-to-Rank 论文对 AMV 宽池排序方向有参考价值

- 已阅读:
  - `~/Downloads/ssrn-6348379.pdf`
  - 论文: `Empirical Asset Pricing via Learning-to-Rank`
  - 新增 Canvas: `ltr-paper-amv-relevance.canvas.tsx`
- **核心启发**:
  - 论文主张不要只预测绝对收益再排序, 而是直接优化横截面相对排序
  - `Listwise / NDCG / Precision@K` 与当前 `AMV bull 宽池 top3/top5/top10` 评估高度相关
  - 当前 `高位+K线确认` 手工组合分数, 本质上是简化版 Learning-to-Rank 排序器
- **对当前路线的影响**:
  - 短期仍优先做 Rust 真实回测兑现, 不因论文直接跳到复杂模型
  - 中期可尝试 LightGBM ranker / LambdaRank, 在 AMV bull 宽池内直接学习 topN 排序
  - 评估指标应加入 `Precision@K / topN 年份拆分 / rolling NAV / MaxDD`, 不再只看单笔均值

### [AMV] Bull 宽池组合权重网格完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_combo_grid.py`
  - 新增 Canvas: `amv-bull-pool-combo-grid.canvas.tsx`
- **实验目的**:
  - 从“最佳 20d 单笔均值”切到更接近组合可交易性的口径
  - 对 `高位+K线确认` 做权重网格与 `top3/top5/top10` 对比
  - 目标指标改为 `tradeoff_score = rolling-sleeve NAV + MaxDD`, 即同时奖励净值、惩罚回撤
- **网格口径**:
  - 价格位置权重 `P`: `1 / 2 / 3`
  - K线确认权重 `K`: `0.5 / 1 / 2`
  - 风险/卖压权重 `R`: `0 / 0.5 / 1 / 1.5`
  - 共 `36` 组权重 × `top3/top5/top10`
- **artifact**:
  - `artifacts/amv_bull_pool_combo_grid/20260503_172052/summary.json`
- **核心结果 (20d, 按 tradeoff_score)**:
  - 全局最优: `top3 高位+K线确认 P2/K0.5/R0`, 单笔均值 `+2.625%`, 相对随机 `+1.276pp`, `NAV +102.97%`, `MaxDD -2.90%`, `tradeoff +100.07%`
  - 单笔均值最高: `top3 P3/K0.5/R0`, 单笔均值 `+2.679%`, 相对随机 `+1.331pp`, `NAV +102.10%`, `MaxDD -3.21%`
  - top5 最优: `top5 P3/K0.5/R0`, 单笔均值 `+2.398%`, `NAV +85.53%`, `MaxDD -3.59%`
  - top10 最优: `top10 P2/K0.5/R1.5`, 单笔均值 `+2.232%`, `NAV +77.42%`, `MaxDD -3.36%`
- **结论**:
  - 当前最佳持仓宽度是 `top3`, 不是更分散的 `top5/top10`
  - 最优权重结构是 `价格位置重权重 + K线轻确认 + 风险不进主分数`
  - `R=0` 排名前列, 说明 ATR/卖压暂时更适合做后置过滤或仓位控制, 不适合直接混入主排序分数

### [AMV] Bull 宽池组合 Top5 排序首轮完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_ranker_lab.py`
  - 新增 Canvas: `amv-bull-pool-combo-ranker.canvas.tsx`
- **实验目的**:
  - 在单因子验证后, 尝试 `价格位置主信号 + Alpha158 K线确认 + 风险/卖压过滤`
  - 组合分数使用每日截面分位加权, 避免不同因子量纲直接相加
- **新增组合**:
  - `高位+K线确认`: `price_pos_20d↑ + close_to_high_20d↓ + KLEN↓ + KMID2↑`
  - `新高+缩振+低风险`: `close_to_high_20d↓ + KLEN↓ + atr_14_pct↓ + panic_vol_ratio_20d↓`
  - `高位+实体强+低风险`: `price_pos_20d↑ + KMID2↑ + atr_14_pct↓ + panic_vol_ratio_20d↓`
  - 另含 `新高+实体强+短上影 / 高位+收盘强+低波 / 新高+缩振+短下影 / 反转+收低+低风险`
- **artifact**:
  - `artifacts/amv_bull_pool_ranker_lab/20260503_165924/summary.json`
- **核心结果**:
  - 20d 新冠军: `高位+K线确认`, 单笔均值 `+2.296%`, 相对随机 `+0.948pp`, rolling-sleeve `NAV +81.64%`, `MaxDD -2.42%`
  - 对比原单因子冠军 `20日高位强势`: 单笔均值 `+2.242%`, 相对随机 `+0.894pp`, `NAV +74.42%`, `MaxDD -5.76%`
  - `新高+缩振+低风险`: 20d 单笔均值 `+2.223%`, 相对随机 `+0.875pp`, `MaxDD -2.80%`
  - `高位+实体强+低风险`: 20d 单笔均值 `+2.217%`, 相对随机 `+0.869pp`, `MaxDD -3.81%`
  - 10d 上组合也开始优于大多数单因子: `高位+K线确认` 单笔均值 `+1.551%`, 相对随机 `+0.705pp`, `MaxDD -4.57%`
- **结论**:
  - 组合路线有效, 不是只靠单因子偶然命中
  - 当前最值得推进的候选是 `高位+K线确认`
  - 风险过滤会降低 `hit15`, 但显著压低回撤并提高 rolling-sleeve 净值, 更接近可交易口径

### [AMV] Bull 宽池补测 Alpha158 K线形态因子

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `scripts/amv_bull_pool_ranker_lab.py`
- **实验目的**:
  - 回答“前期 `core12 + alpha158(kbar_shape)` 里的 K 线形态因子, 是否已在 AMV bull 宽池排序里尝试”
  - 将 Alpha158 `kbar_shape` 9 因子加入 `mechanical AMV bull + LF2 宽池` 的每日 top5 单因子排序评估
- **新增因子**:
  - `KMID / KLEN / KMID2 / KUP / KUP2 / KLOW / KLOW2 / KSFT / KSFT2`
  - 每个因子测试正反两个方向, 共新增 `18` 个 ranker
- **artifact**:
  - `artifacts/amv_bull_pool_ranker_lab/20260503_165512/summary.json`
- **核心结果**:
  - 5d: `K线振幅收缩(KLEN_asc)` 单笔均值 `+1.200%`, 相对随机 `+0.802pp`, `MaxDD -9.57%`
  - 10d: `实体占比偏强(KMID2_desc)` 单笔均值 `+1.379%`, 相对随机 `+0.532pp`
  - 20d: `实体占比偏强(KMID2_desc)` 单笔均值 `+1.969%`, 相对随机 `+0.620pp`
  - 20d: `下影线短(KLOW_asc / KLOW2_asc)` 单笔均值 `+1.787%`, 相对随机 `+0.438pp`, `MaxDD -10.43%`
- **结论**:
  - AMV bull 宽池里, Alpha158 K线形态因子不是空的, 其中 `KLEN_asc / KMID2_desc / KSFT2_desc / KLOW*_asc` 有明确正增益
  - 但第一名仍是价格位置类信号: `接近20日新高` 在 5d/10d 继续最强, `20日高位强势` 在 20d 继续最强
  - 下一步组合应优先考虑: `价格位置主信号 + K线形态确认 + 风险过滤`

### [AMV] Bull 宽池单因子 Top5 排序首轮完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_ranker_lab.py`
  - 新增 Canvas: `amv-bull-pool-ranker-lab.canvas.tsx`
- **实验目的**:
  - 在 `mechanical AMV bull + LF2 宽池` 内, 每天按单因子排序选 top5
  - 对照基准为同一天宽池随机 5 只的日期等权期望
- **第一批排序因子**:
  - 动量/反转: `ret_5d / ret_20d` 正反向
  - 价格位置: `ma_bias_20 / price_pos_20d / close_to_high_20d`
  - 量能/量价: `turnover_ma_ratio / abnormal_vol / turnover_accel / vol_price_corr_20d`
  - 风险/波动: `vol_20d / vol_compress / atr_14_pct / panic_vol_ratio_20d`
  - 日内/行为: `intraday_pos / disp_bias_20`
- **artifact**:
  - `artifacts/amv_bull_pool_ranker_lab/20260503_130223/summary.json`
- **核心结果**:
  - 5d 最强: `接近20日新高`, 单笔均值 `+1.427%`, 相对随机 `+1.029pp`, 但回撤较深 `-38.06%`
  - 10d 最强: `接近20日新高`, 单笔均值 `+1.665%`, 相对随机 `+0.819pp`
  - 20d 最强: `20日高位强势`, 单笔均值 `+2.242%`, 相对随机 `+0.894pp`, rolling-sleeve `NAV +74.42%`, `MaxDD -5.76%`
  - 20d 次强: `接近20日新高`, 单笔均值 `+2.091%`, 相对随机 `+0.742pp`
  - 稳定进入前列: `收盘靠低`, `20日反转`, `成本线下回归`
- **结论**:
  - 单因子排序已能打败随机 5 只, `AMV Bull Pool Ranking` 路线成立
  - 第一批最值得继续挖的是价格位置类信号, 尤其 `price_pos_20d` 与 `close_to_high_20d`
  - 下一步应做二阶组合: 价格位置主信号 + 风险过滤, 目标是在保留 20d 收益的同时进一步压低回撤

### [AMV] Bull 宽池随机 5 只基准成立

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `scripts/amv_bull_pool_random_baseline.py`
  - 新增 Canvas: `amv-bull-pool-random-baseline.canvas.tsx`
- **实验目的**:
  - 不训练模型, 不使用 B1 教科书形态规则
  - 只验证 `mechanical AMV bull + LF2 流动性池 + 每日随机 5 只` 是否具备诚实下限
- **实验口径**:
  - Universe: `market_cap_100m >= 100`, `amount_ma20 >= 5000万`
  - 机械 AMV regime: `bull=max(ret_1d, ret_2d) >= 4.0%`, `bear=ret_1d <= -2.3%`, `effective_lag_days=1`
  - Monte Carlo: 每个 eligible day 随机 5 只, `1000` 次, `seed=42`
  - Horizon: `5d / 10d / 20d`
  - 对照池: `LF2 all days` / `LF2 AMV bull` / `LF2 non-bull`
- **artifact**:
  - `artifacts/amv_bull_pool_random_baseline/20260503_125323/summary.json`
- **核心结果**:
  - `LF2 AMV bull`: `542` 个 eligible days, 每日候选中位数 `1104`
  - 随机 5 只每笔平均收益中位数:
    - `5d +0.399%`
    - `10d +0.842%`
    - `20d +1.337%`
  - `LF2 non-bull` 对照:
    - `5d -0.088%`
    - `10d -0.206%`
    - `20d -0.121%`
  - 20d rolling-sleeve 组合:
    - AMV bull: `NAV +38.27%`, `MaxDD -19.08%`
    - non-bull: `NAV -10.62%`, `MaxDD -24.15%`
- **结论**:
  - AMV bull 作为市场 gate 的价值进一步确认
  - 宽池随机 5 只已经明显打败 all-days 与 non-bull, 说明后续方向不应再纠结 B1 教科书规则
  - 下一步研究重点应转为 `AMV Bull Pool Ranking`: 在 AMV bull 宽池内寻找能稳定打败随机 5 只的排序信号

### [Rotation] AMV bull-only 训练实验完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `notebooks/cross_section_rotation.py`
  - `utils/baostock_utils.py`
- **实验口径**:
  - `TRAIN_REGIME_MODE = "amv_bull_only"`
  - 训练样本只保留机械 AMV bull 生效日 (`effective_lag_days=1`)
  - 机械 AMV regime: `bull=max(ret_1d, ret_2d) >= 4.0%`, `bear=ret_1d <= -2.3%`
  - 导出端仍使用 `BULL_REGIME_SOURCE = "mechanical_amv"` + `EXPORT_EMA_ALPHA = 0.3`
- **artifact**:
  - Train: `artifacts/rotation/rot_fwd_ret_1d_lightgbm_20260503_123830_703384b9`
  - Signal: `signals/20260503_123831_276`
  - 本轮使用 ST 过期缓存兜底后, universe 与前一轮对齐: `2616` 只股票, `1,056,712` 条 score
- **同口径对比** (`2022-09-01 -> 2026-04-08`, `hold_buffer=20`, `max_hold_days=10`, `min_score=0.002`, 成本不变):
  - 旧 all-train + AMV hard gate: `Total Return +9.16%`, `Max Drawdown 7.66%`, `Trades 1307`
  - AMV-weighted + AMV hard gate: `Total Return +0.55%`, `Max Drawdown 8.79%`, `Trades 1498`
  - AMV bull-only + AMV hard gate: `Total Return -24.41%`, `Max Drawdown 29.05%`, `Trades 2917`
  - AMV bull-only 不开 gate: `Total Return -35.97%`, `Max Drawdown 37.36%`, `Trades 5843`
- **结论**:
  - `amv_bull_only` 明显失败, 说明非 bull 样本并不是简单污染源; 直接删掉非 bull 样本会破坏模型排序能力
  - 训练侧 AMV 加权 / 过滤都不应作为 Rotation 主线默认方案
  - 当前更合理的方向是保留 all-train, 继续把 AMV 放在交易层做 hard gate 或 regime-aware 参数
- **基础设施修正**:
  - `get_st_blacklist_pl()` 在 AKShare / Baostock 都失败时, 现在允许使用过期本地缓存
  - 目的: 网络不可用时避免 ST 黑名单变成空表, 导致回测 universe 漂移

## 2026-05-02

### [Rotation] AMV bull 样本加权训练实验完成

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `notebooks/cross_section_rotation.py`
  - `utils/signal_export.py`
- **实验口径**:
  - `TRAIN_REGIME_MODE = "amv_weighted"`
  - `AMV_BULL_SAMPLE_WEIGHT = 2.0`
  - 机械 AMV regime: `bull=max(ret_1d, ret_2d) >= 4.0%`, `bear=ret_1d <= -2.3%`, `effective_lag_days=1`
  - 导出端仍使用 `BULL_REGIME_SOURCE = "mechanical_amv"` + `EXPORT_EMA_ALPHA = 0.3`
- **artifact**:
  - Train: `artifacts/rotation/rot_fwd_ret_1d_lightgbm_20260502_230927_703384b9`
  - Signal: `signals/20260502_230928_106`
  - `train.meta.json` 已记录 `train_regime_mode / amv_bull_sample_weight / AMV trigger / lag`
  - `signal.meta.json` 已记录 `bull_regime_source / bull_regime_rows_pct / AMV trigger / lag`
- **同口径对比** (`2022-09-01 -> 2026-04-08`, `hold_buffer=20`, `max_hold_days=10`, `min_score=0.002`, 成本不变):
  - 旧 all-train + AMV hard gate: `Total Return +9.16%`, `Max Drawdown 7.66%`, `Trades 1307`
  - 新 AMV-weighted + AMV hard gate: `Total Return +0.55%`, `Max Drawdown 8.79%`, `Trades 1498`
  - 新 AMV-weighted 不开 gate: `Total Return -9.39%`, `Max Drawdown 17.75%`, `Trades 3544`
- **结论**:
  - AMV hard gate 仍然显著降低交易频率和回撤, 但 `bull_weight=2.0` 的训练侧加权没有改善 Rotation
  - 当前不应把 B 作为主线默认方案; 下一步更值得做的是 `bull_only` 作为分布污染验证, 或对 AMV bull/bear 分桶分别调 `min_score / max_positions`

## 2026-05-01

### [Rotation] 接入机械 AMV Regime 作为开仓 gate 候选

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `utils/active_market_value_regime.py`
  - 更新 `notebooks/cross_section_rotation.py`
- **方向修正**:
  - AMV 机械 regime 不再以贴近 `LOOSE_PERIODS` 为目标
  - `LOOSE_PERIODS` 只作为 hindsight 上界 / sanity check
  - 下一步优先在 Rotation 上验证机械 AMV gate 是否改善未来收益与回撤
- **实现**:
  - 新增 `build_active_market_value_regime_frame()`
  - 默认机械口径: `bull_trigger=max(ret_1d, ret_2d, ret_3d) >= 4.5%`, `bear_trigger=ret_1d <= -2.3%`
  - `cross_section_rotation.py` 导出阶段新增 `BULL_REGIME_SOURCE = "mechanical_amv"` / `"manual"`
  - 导出 `signal.parquet` 的 `is_bull_regime` 可直接供 Rust `entry.require_bull_regime` 使用
- **验证方式**:
  - 在 Rotation notebook 里导出 `mechanical_amv` signal
  - 同一份 signal 分别跑:
    - `scripts/rotation_backtest.py <signal> --no-require-bull-regime`
    - `scripts/rotation_backtest.py <signal> --require-bull-regime`
  - 再把 `BULL_REGIME_SOURCE` 改为 `manual` 导出一次, 跑 `--require-bull-regime`, 作为 hindsight 上界对照

### [研究] 活跃市值 Regime Lab notebook 新增

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - 新增 `notebooks/active_market_value_regime_lab.py`
- **目标**:
  - 基于独立库 `../QuantData/Ashare/active_market_value.duckdb` 重新机械化定义 bull / bear regime
  - 用 Plotly Candlestick 把活跃市值按 K 线形态画出来, 对齐指南针客户端里的视觉形态
  - 将机械 regime 与旧手工 `LOOSE_PERIODS` 做交叉对账
- **Notebook 首版内容**:
  - 读取 `active_market_value` 主表并计算 `ret_1d / ret_2d / ret_3d / ret_5d`
  - 可调参数: 1/2/3 日累计 bull trigger 阈值, 1/2/3 日累计 bear trigger 阈值
  - 机械状态机: `neutral -> bull / bear`, bull trigger 优先切多, bear trigger 切空
  - K 线图叠加手工 bull 区间、机械 bull 区间、bull/bear trigger marker
  - 输出 overlap 摘要、每个手工区间的机械解释度、最近触发点
- **验证**:
  - `uv run python -m py_compile notebooks/active_market_value_regime_lab.py` 通过
  - `uv run marimo check notebooks/active_market_value_regime_lab.py` 通过
  - 核心数据逻辑冒烟: `manual_days=728`, 默认阈值下 `mechanical_bull_days=703`, `both_days=520`
- **口径修正 (晚间)**:
  - Bear trigger 改为只看单日跌幅 `ret_1d <= threshold`, 不再使用 2/3 日累计跌幅
  - 修正后默认 `bull=4.0 / bear_1d=-2.3`: `mechanical_bull_days=925`, `both_days=650`, `recall=89.3%`, `precision=70.3%`, `F1=78.6%`
  - 当前网格最优: `bull=4.5 / bear_1d=-2.3`, `mechanical_bull_days=860`, `both_days=646`, `recall=88.7%`, `precision=75.1%`, `F1=81.4%`
  - 结论: 活跃市值机械 regime 初步成立, 下一步应优先验证 `bull=4.5 / bear_1d=-2.3` 在 B1 / Rotation 中的可执行效果

### [基础设施] 活跃市值 DuckDB 从 QMT 库拆出

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `rpa_parse/README.md`
  - `rpa_parse/ingest_active_market_value.py`
- **决策**:
  - 活跃市值是指南针客户端衍生的市场级指标, 不是 QMT 数据源
  - 默认 DuckDB 从 `../QuantData/Ashare/qmt_data.duckdb` 改为 `../QuantData/Ashare/active_market_value.duckdb`
  - QMT 行情库继续只承载 QMT 同步链路, 后续研究联表时用 DuckDB `ATTACH`
- **收益**:
  - 避免污染 `qmt_data.duckdb`
  - 避免 QMT 数据库文件锁影响活跃市值日更
  - 活跃市值可作为独立数据产品备份、重建和验收
- **验证**:
  - 已用默认路径执行 `rpa_parse/ingest_active_market_value.py --mode upsert`
  - 写入库: `../QuantData/Ashare/active_market_value.duckdb`
  - 结果: `1776` 行, `2019-01-02 -> 2026-04-30`, `flagged_rows=66`, `qc_errors=0`

### [基础设施] 活跃市值日更 5 张增量校验通过

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
- **本次增量**:
  - `active_market_value.csv / parquet` 已从 `1771` 行更新到 `1776` 行
  - 最新日期从 `2026-04-23` 延伸到 `2026-04-30`
  - 新增区间: `seq_01772.png -> seq_01776.png`
- **质量校验**:
  - 新增 5 行 `review_reason` 均为空
  - 全量 `active_market_value_review.csv` 仍为 `66` 行, 未新增 review 项
  - 复算 `chg_pct` / `amplitude` 的一致性校验: `bad rows = 0`
- **DuckDB 入库状态**:
  - `rpa_parse/ingest_active_market_value.py --mode upsert` 已尝试执行
  - 目标库 `../QuantData/Ashare/qmt_data.duckdb` 当前被另一个 Python 进程持锁 (`PID 10743`), 因此写入待锁释放后重跑
  - 该失败属于 DuckDB 文件锁冲突, 不是 OCR / Parse 数据质量问题
  - 后续已决定改用独立 `../QuantData/Ashare/active_market_value.duckdb`, 并已完成首次 upsert

## 2026-04-30

### [基础设施] 活跃市值 Parse 阶段放弃 PaddleOCR, 切到 macOS Vision

- 已更新:
  - 本文件 (本条)
  - `rpa_parse/parse_active_market_value.py`
  - `rpa_parse/README.md`
  - `project-status.md`
  - `experiments/active-market-value-automation.md`
- **背景**: PaddleOCR / PaddlePaddle 在 macOS + Python 3.13 + Apple Silicon 下虽然官方声明可用, 但依赖重、初始化慢、API 版本差异明显, 对当前固定 readout 小图属于过度方案.
- **决策**: `rpa_parse` 默认 OCR 后端改为 macOS 原生 **Vision Framework**.
  - 不再 import / 调用 `paddleocr`
  - 通过 PyObjC 调用 `VNRecognizeTextRequest`
  - 默认语言 `zh-Hans + en-US`
  - 关闭 language correction, 避免短字段和数字被错误纠正
  - 增加 `customWords`: `开/高/低/收/幅/量/额/盘/率/振/亿/周一~周五`
- **接口保持**:
  - `parse_lines()` / 字段正则 / review CSV / DuckDB 写入逻辑不变
  - CLI 仍是 `python rpa_parse/parse_active_market_value.py --input ... --output ...`
  - 保留 `--limit` / `--progress-every` 便于小样本冒烟
- **下一步**:
  - 同步 PyObjC 依赖: `uv sync`
  - 先跑 `--limit 1`, 验证 Vision 对 11 个字段识别是否完整
  - 若字段名识别不稳, 优先在 `normalize_text()` 增加 OCR 纠错映射, 不再回退 PaddleOCR

### [基础设施] 活跃市值 DuckDB ingest 脚本落地

- 已更新:
  - 新增 `rpa_parse/ingest_active_market_value.py`
  - 更新 `rpa_parse/README.md`
  - 移除 `parse_active_market_value.py` 内旧的简易 DuckDB 写入入口, 避免旧 schema 误用
- **表结构**:
  - 主表: `active_market_value`
  - 主键: `trade_date DATE PRIMARY KEY`
  - OHLC 字段统一用 `amv_open / amv_high / amv_low / amv_close`
  - `chg_pct` 入库改名为 `chg_abs_pct`, 明确截图里的"幅"是绝对涨跌幅
  - `volume / amount / position` 入库改名为 `volume_100m / amount_100m / position_100m`, 明确单位为"亿"
  - 保留 `source_seq / source_filename / raw_ocr_text / quality_flags / ocr_min_confidence` 用于追溯
- **写入语义**:
  - `--mode upsert`: 使用 DuckDB `INSERT ... ON CONFLICT (trade_date) DO UPDATE`, 按日期覆盖
  - `--mode replace`: 清空表后重写
  - 自动创建 `active_market_value_qc` view, 用前一日 `amv_close` 复算 `chg_abs_pct / amplitude_pct`
- **测试**:
  - 已用临时库 `data/active_market_value/test_ingest.duckdb` 跑通 `replace` 与 `upsert`
  - 结果: `1771` 行, `2019-01-02 -> 2026-04-23`, `flagged_rows=66`, `qc_errors=0`

### [基础设施] 活跃市值日更增量链路补齐

- 已更新:
  - `rpa_capture/run_capture.py`
  - `rpa_capture/README.md`
  - `rpa_parse/parse_active_market_value.py`
  - `rpa_parse/README.md`
- **Capture 端**:
  - 既有 `--start-seq` 已确认可用于自定义起始序号
  - 新增 `--overwrite` 开关; 默认禁止覆盖已存在 `seq_*.png`
  - 启动时打印计划截图范围, 例如 `seq_01772.png -> seq_01776.png`
  - 结束提示改为使用实际 `start_seq`, 避免一直提示 `seq_00000`
- **Parse 端**:
  - 新增 `--incremental`
  - 若 `active_market_value.parquet` 已存在, 按 `seq` 跳过已解析图片, 只 OCR 新增截图
  - 合并后仍输出完整 `active_market_value.parquet / csv / review.csv`
  - 保留已有人工修正值, 避免日更时重跑全量 OCR 覆盖历史修正
- **测试**:
  - 当前目录已有 `1771` 行, 无新增图片时运行 `--incremental`:
    - `existing rows=1771`
    - `new images=0`
    - `new rows=0`
    - `review rows=66`

## 2026-04-25

### [基础设施] 活跃市值截图 Parse 阶段首版脚本落地

- 已更新:
  - 本文件 (本条)
  - `project-status.md` (RPA 章节追加 `rpa_parse/` 状态)
  - 新增 `rpa_parse/README.md`
  - 新增 `rpa_parse/parse_active_market_value.py`
- **背景**: 用户已拿到接近 1800 张 `0AMV 活跃市值` readout 截图, 样式固定、黑底高对比、11 个字段稳定, 适合用 OCR + 字段正则解析成结构化数据.
- **技术路线**:
  - 保持 `rpa_capture/` 只负责 Windows 端截图
  - 新增 `rpa_parse/` 作为 Mac/Windows 解析端
  - 用 PaddleOCR 读取 `seq_*.png`, 抽取文本行
  - 用字段名 (`开/高/低/收/幅/量/额/盘/率/振`) + 数字正则解析
  - 输出 `active_market_value.parquet / csv / review.csv`
  - 可选写入 DuckDB `active_market_value`
- **依赖决策**:
  - 暂不把 PaddleOCR / PaddlePaddle 加进主 `pyproject.toml`
  - 推荐用 `uv run --python 3.11 --with paddleocr --with paddlepaddle ...` 临时环境执行, 避免污染主项目 Python 3.13 环境与 `uv.lock`
- **校验能力**:
  - 缺字段进入 `active_market_value_review.csv`
  - `low <= open/close <= high` 异常进入 review
  - OCR 低置信度进入 review
  - 保留 `ocr_text` 与可选 `raw_ocr/*.json`, 便于人工复核和规则迭代
- **下一步**:
  - 用用户当前截图目录跑首轮 1800 张 OCR
  - 查看 `active_market_value_review.csv`, 统计需人工修正比例
  - 若 OCR 错误集中在固定字段, 在 `parse_active_market_value.py` 增加字段级纠错规则
  - 通过日期连续性与 OHLC 派生校验后, 写入 DuckDB 并开始对账旧 `LOOSE_PERIODS`

## 2026-04-21 (晚)

### [文档] 活跃市值自动化路线图正式立项

- 已更新:
  - 本文件 (本条)
  - `project-status.md` (RPA 章节追加路线图索引)
  - **新增 `experiments/active-market-value-automation.md`** (6 阶段路线图)
- **背景**: 今天 RPA Capture PoC 通过 (见下条), 这是项目级长期主线 (单笔 ROI 最高的工程投入), 应该有专项规划文档, 跟 `b1-next-phase.md` / `rotation-next-phase.md` / `target-strategy-evolution.md` 平级
- **新文档结构** (11 节):
  - 一、为什么做 (4 个前期发现总结: alpha 来源 / timing 本质 / 单点依赖 / 必须自建)
  - 二、架构设计 (Capture / Parse / 消费侧三层 + ASCII 拓扑图 + 解耦原则)
  - 三、Phase 1 Capture (✅ 已完成验收清单 + 5 个关键技术决策表)
  - 四、Phase 2 Parse (🟡 待实现, 11 字段 schema + DuckDB 表设计 + 校验规则 + 验收门槛)
  - 五、Phase 3 规则引擎 + 历史复算 (🔵 R0~R4 候选规则 + 75 组阈值网格)
  - 六、Phase 4 集成到 B1 / Rotation (🔵 验收标准 + 多策略框架雏形)
  - 七、Phase 5 日更 + 运维 (🔵 计划任务 + 监控 + 灾备)
  - 八、Phase 6 扩展到其他指南针指标 (🔵 D股价活跃度 / 资金主力 等)
  - 九、风险与不确定性 (5 类风险 + 缓解方案, 含法律风险声明)
  - 十、当前 TODO 清单 (10 项, 已勾 1)
  - 十一、相关文档索引
- **战略价值**: 这份文档是项目下半场的"主航线图". 完成后我们终于能回答的几个核心问题:
  - 手工 LOOSE_PERIODS 能否被机械规则替代 (Phase 3)
  - 如果能, 当前所有 B1 / Rotation 的 hindsight 上界还能不能保住 (Phase 4)
  - 如果不能, alpha 是否本身就含主观成分而非数据成分

## 2026-04-21

### [基础设施] 活跃市值 RPA 数据管道启动: capture 阶段 PoC 通过

- 已更新:
  - 本文件 (本条)
  - `project-status.md` (调整"活跃市值自动化"状态)
  - 新增 `rpa_capture/` 模块 (Windows 端截图脚本)
- **战略背景**: 活跃市值是指南针客户端的专利指标, 当前 B1 / Rotation 的 timing alpha 均依赖手工标记的 25 个 `LOOSE_PERIODS`, 单点依赖风险高且无法日更. 决定通过 RPA 自动化抓取整个指标数据源 (1993 起可回溯).
- **架构拆分** (Mac 主力 + Windows VM 形态):
  - **Capture 阶段** (Windows 端): 只做截图 + 文件落地, 不做 OCR / 解析 / 入库
  - **Parse 阶段** (Mac 端): OCR + 校验 + 入 DuckDB (待实现)
  - 两阶段通过 PNG + manifest.jsonl 解耦, OCR 方案升级不需要重抓
- **依赖纪律**: capture 阶段只 `pywinauto + mss` 2 个包, 不挑环境
- **新增文件** (`rpa_capture/`):
  - `run_capture.py`: 主入口, 支持全屏 / 区域截图 / 续抓 / `--no-focus` 等参数
  - `calibrate_region.py`: 交互式 readout 区域标定 (tkinter, stdlib 无新依赖, 内置 DPI 感知)
  - `requirements.txt`: 极简依赖
  - `README.md`: 完整使用说明 + cursor 漂移问题 troubleshooting
- **关键技术决策**:
  - 用纯 Win32 `SetForegroundWindow` 拉前台, 弃用 `pywinauto.Application` (后者会触发"合成点击"导致图表 cursor 漂移)
  - 截图前把鼠标停到 `(2, 2)`, 避免在主图区域 hover 干扰 cursor
  - 配置 region 后, 单图从 ~3 MB 降到 ~9 KB (省 99.7% 体积, 1700 张总量从 5GB 降到 50MB)
  - 时间方向: `seq=0` 是最早起始日, `seq=N` 是最新, 入库后无需 reverse
- **PoC 验收 (2026-04-21 全过)**:
  - 起点准确: 手动选 2019-01-02, 截图显示 `20190102 周三`, 不漂
  - 方向正确: 按 → 后日期严格 +1 交易日 (周末自动跳过, 指南针自己处理)
  - 10 张图日期连续: `20190102 → 20190103 → 20190104 → 20190107(周一) → ... → 20190115`
  - 性能优于预期: 10 张耗时 2.46 秒, 平均 **246 ms/张**, 推算 1700 天 ≈ 7 分钟
  - readout 截图清晰: 11 个字段 (date / 开高低收 / 幅 / 量 / 额 / 盘 / 率 / 振) 全部可读, OCR 难度极低
- **下一步** (用户在 PD VM 跑完整流程):
  - 历史回填: 1700 天全量截图 (在 PD VM 内执行)
  - 实现 `rpa_parse/` (Mac 端): PaddleOCR + polars + DuckDB
  - 跟手工的 25 个 `LOOSE_PERIODS` 交叉对账验收
  - Windows 计划任务实现日更
- **闲鱼对照**: 闲鱼有人卖 "0AMV 全历史 (1993~2026) excel 200 元", 90% 概率也是 RPA 抓的, 但缺乏可追溯性 / 日更服务 / 数据质量报告. 自建管道的核心价值是可持续 + 可校准 + 可扩展 (能顺便抓 D股价活跃度等其他指南针指标)

## 2026-04-19 (晚)

### [B1] 三个 stage-0 notebook 整合到 `b1_alpha_proof.py`, simple_b1_lab 实证验证, alpha 来源精修

- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `experiments/b1-next-phase.md`
- 项目空间清理 (用户确认):
  - **删除** `notebooks/b1_stage0_alpha_proof.py` (已整合)
  - **删除** `notebooks/b1_stage0_J_interaction.py` (已整合)
  - **删除** `notebooks/b1_stage0_textbook_v2.py` (已整合)
- B1 notebook 当前阵容 (10 → 7):
  - **主线**: `b1_seed_ml_baseline.py` (待重构), `perfect_top10b1_analyze.py`, `b1_case_expansion_mining.py`, `simple_b1_lab.py`
  - **stage-0 整合版** (新): `b1_alpha_proof.py` (Q1~Q8, 一气呵成证明 alpha 在哪)
  - **textbook 旁路**: `textbook_case_classifier.py` (LightGBM OOF 证伪记录)

#### [新增 notebook] `b1_alpha_proof.py` — Q1~Q8 一气呵成 + 大白话
- 整合自三个已删除的 stage-0 notebook (alpha_proof / J_interaction / textbook_v2)
- 全部用 LF2 严格池 (mv≥100亿+amt20≥5000万) 做基准
- 全部用大白话, 不再有 R0/L2/T3/CI 之类术语
- 8 个 cell 顺次回答:
  - **Q1**: 3 档池子各自的全市场基准 (L0 +1.10% → LF2 +0.24%, 缩水 78%)
  - **Q2**: 仅多头区间 ret_lift **+1.46pp** ✓ 显著, 100% ex-ante
  - **Q3**: 仅 (白>黄+收>黄) ret_lift **-0.62pp** ⚠ 显著为负 (反直觉)
  - **Q4**: 仅 J<14 ret_lift **+0.07pp** ✓ 但极弱
  - **Q5**: 多头叠加: 多头 +1.46pp → 加白>黄+收>黄 +1.27pp (-0.19pp 拖累) → 加 J<14 +1.60pp (+0.14pp)
  - **Q6**: 教科书完整 5 条规则在多头内累积, 边际贡献为负, 但 hit_15pct 显著抬高 (有止盈才有意义)
  - **Q7**: **月间稳定性 + 反直觉发现** (本次 stage-0 最重要的新洞察, 见下)
  - **Q8**: 终极汇总表

#### [Q7 关键新洞察] alpha 真正来源是"非多头时段空仓"择时
- **Q7-1 月度 t 检验** (41 个月):
  - 多头日开仓的 fwd_ret_20d 月均 **+1.67%**, 月度标准差 5.10%
  - t = **+2.10** (>2 显著), 60.98% 月份正
  - 与 Q2 的 +1.46pp 一致, **月间稳定**
- **Q7-2 月内对比** (反直觉发现):
  - 月均 (多头日开仓 - 当月每天买) = **-0.21pp**, t = -0.636 不显著
  - 多头日胜出当月每天买的月份: 7/41 = 17.07%
  - 解释: fwd_ret_20d 跨 regime, 多头末期开仓会撞上 regime switch 回吐
- **真正的 alpha 描述**: "**只在多头时段开仓, 避开非多头时段不开仓**" 这个择时动作本身值钱
  - 不是"多头时段票更好" (cross-section selection alpha)
  - 而是"非多头时段空仓" (timing alpha)
- 战略含义:
  - **路线 B (池内 top5 排序)** 价值下调: cross-section 信号在多头内可能没多少 alpha
  - **路线 C (优化择时本身)** 价值上调: 多头切换 / 持仓窗口 / 行业差异才是真正未挖矿
  - 仍可先做路线 A 跑实证下限

#### [实证支撑] `simple_b1_lab.py` 6 年回测复核
- 用户当晚跑了 simple_b1_lab.py (LOOSE_PERIODS + rule_wmacd B1 + T+1 + 3% 止损 + top200, 6314 个信号)
- 总体 持仓20天 +2.60% / 死拿20天 +2.35% — **与 b1_alpha_proof Q5 在 L0 池的 +2.54% 几乎完全一致 (差 0.06pp)**
- 年度衰减曲线证实 Q7-2:
  - 2025 年死拿 30 天 +8.66% > 死拿 20 天 +6.64% > 持仓 20 天 +5.97% — 牛市里止损是浪费机会, 应延长持仓
  - 2019 年 持仓 5/10/15 天 = +5.16/+4.89/+6.21%, 持仓 20/25/30 天 = +3.33/+1.01/+0.91% — 末期撞 regime switch 大幅回吐 (实证 Q7-2)
- 两份证据互相印证, 没有矛盾

#### [战略路线优先级修正]
- 之前 (2026-04-19 上) 建议优先做 B (池内 top5 排序), 现根据 Q7-2 修正:
  - **A (极简化)**: 仍推荐先做 (1 天落地诚实下限)
  - **B (池内排序)**: **降级**, 价值不确定, 放在 A 之后再决定要不要做
  - **C (优化择时)**: **升级为主线候选**, simple_b1_lab 已展示持仓窗口曲线初步轮廓

## 2026-04-19

### [B1] Stage-0 价值检验全跑通: 教科书形态被证伪, 多头区间是唯一金矿
- 已更新:
  - 本文件 (本条)
  - `project-status.md`
  - `experiments/b1-next-phase.md`
- 项目空间清理 (用户确认):
  - 删除 10 个老旧 / 已被替代的 B1 notebook (共 ~232 KB):
    - `b1_condition_mining.py` (被 `b1_case_expansion_mining.py` 取代)
    - `b1_ml_dedicated.py`, `b1_ml_explore.py` (被 stage0 系列 + classifier 取代)
    - `chronos_b1_base.py`, `sequence_b1_base.py`, `sequence_b1_opt.py`
    - `smart_b1_base.py`, `smart_b1_opt.py`
    - `simple_b1.py`, `simple_b1_opt.py`
- B1 notebook 当前阵容:
  - **主线**: `b1_seed_ml_baseline.py` (回测出货), `perfect_top10b1_analyze.py` (探索), `b1_case_expansion_mining.py` (案例挖掘), `simple_b1_lab.py` (轻量验证)
  - **本轮 stage-0**: `b1_stage0_textbook_v2.py` (T1~T6, 主战场), `b1_stage0_alpha_proof.py` (4 大统计检验), `b1_stage0_J_interaction.py` (J × 量价交互), `textbook_case_classifier.py` (LightGBM OOF 证伪)

#### [新增 notebook] `b1_stage0_textbook_v2.py` — 6 步累积验证 stage-0 alpha
- T1: 教科书规则**累积过滤** (L0 全市场 → L2 白>黄+收>黄 → +前期放量 → +今日企稳 → +振幅≤7% → +J<14 → +极致缩量), 在 2021-2025 全样本上 ret_lift 从 +0pp 一路降到 **-1.16pp**
- T2: J × peak_vol_shrink_60d 二维交互 pivot (在 L2_surge 子集), 显示 J<14 + 适度缩量 cell 是 mfe 局部高点
- T3: 关键累积层 Bootstrap 95% CI, 教科书全 5 条置信区间为 [-1.45, -0.86]pp, **统计显著为负**
- T4: 把 T1 的累积过滤限制到 `is_manual_bull == True` 子集, ret_lift 全部翻正:
  - R0 manual_bull baseline: **+1.36pp** ✓
  - R+L2+surge+企稳+振幅+J<14: +1.23pp ✓
  - R+L2+教科书全 5 条 (含极致缩量): +0.97pp ✓
  - 结论: regime 是核心, 教科书规则在 bull 内仍 < regime baseline, 边际贡献为负
- T5: **流动性过滤** (隐患 2 排除), 在 3 档流动性 (L0 / LF1 mv≥50亿+amt≥3000万 / LF2 mv≥100亿+amt≥5000万) 内对比 R0 vs 形态规则:
  - 本档全市场 baseline ret: L0 +1.10% → LF1 +0.48% → **LF2 +0.24%** (缩水 78%)
  - 证明 T4 的 +2.46% manual_bull baseline 含微盘脉冲虚高
  - 在 LF2 严档下, R+L2+J<14 ret_lift **+1.60pp** > R0 ret_lift +1.46pp (反超 +0.14pp)
  - hit_15pct: R+L2 26.72% vs R0 23.40% (形态规则提升 +3.3pp 的大涨概率)
- T6: **3 个候选信号独立拆解** (在 LF2 池子里, 单独抽出与池内 baseline 比):
  - 仅 J<14: ret_lift **+0.07pp** (CI [+0.03, ?]), ✓ 但极弱, 几乎等于瞎选
  - 仅 (白>黄 且 收>黄): ret_lift **-0.62pp** (CI [-0.66, ?]), **⚠ 负 alpha**
  - 仅 多头区间: ret_lift **+1.46pp** (CI [+1.43, +1.49]), **✓ 唯一硬 alpha**
  - 多头 + 白>黄 + 收>黄: +1.27pp (反而比"仅多头" -0.19pp, **被形态拖累**)
  - 多头 + 白>黄 + 收>黄 + J<14: +1.60pp (比"仅多头" +0.14pp)
  - **辛普森悖论实锤**: "白>黄+收>黄" 全市场看是负 alpha, 但在多头子集里加它是正 alpha; 反过来在多头子集里加它又比裸多头略弱

#### [新增 notebook] `b1_stage0_alpha_proof.py` — 4 大统计检验
- 步骤 H (stage-0 表): H1 全市场 / H2 J<=20 / H3 WL>YL & close>YL 各自的 fwd_mfe_risk_adj_20d / fwd_ret_20d / hit15
- 步骤 I (4 个统计检验):
  - I1 截面分位 alpha + t-test: WL>YL & close>YL 显著正, J<=20 显著负
  - I2 hit_15pct Binomial Z 检验: 同向
  - I3 月度时序 alpha t 检验: WL>YL & close>YL 月均 alpha 显著, J<=20 不显著
  - I4 Bootstrap 95% CI: 与 I1 一致

#### [新增 notebook] `b1_stage0_J_interaction.py` — J × 量价二维交互 (前置版)
- 在 L2 (WL>YL & close>YL) 子集内, J×vol_shrink_40, J×red_green_ratio_20 二维 pivot
- 反直觉发现: 低 vol_shrink_40 (传统认为"健康") 反而是负 alpha, 高 vol_shrink_40 是强正 alpha
- 这一结果驱动了 v2 的"量价健康"序列化重定义 (`prior_volume_surge_60d` / `peak_vol_shrink_60d` / `pullback_vol_shrink_5_20`)

#### [新增 notebook] `textbook_case_classifier.py` — LightGBM 排序模型证伪
- Option C: 用 LightGBM 在 textbook 标签上做二分类
- In-sample enrichment 5.48x (典型 leakage)
- 5-fold StratifiedKFold OOF enrichment **0.94x**, 等同瞎猜 → 模型无独立排序能力
- 直接证伪 "复杂特征 + 模型打分" 路线

#### [核心结论摘要 — 2026-04-19]
- **真正的 alpha 99% 来自一条**: 用户 RPA 抓的活跃市值多头区间 (LOOSE_PERIODS, T+1 ex-ante 无 look-ahead)
- **教科书 B1 形态几乎无独立 alpha**: 白>黄+收>黄 / J<14 / 量价健康 / N 字结构, 单独拿出来跟全市场比都不显著, 甚至负
- **它们只在 "多头区间" 子集里组合时贡献边际 +0.14pp**, 但代价是从 60w 样本缩到 2.8w
- **LightGBM + textbook 标签 + alpha158 特征链路全无效**, OOF 0.94 倍验证
- **过去回测能赚的真实功劳分配** (估计):
  - 70% 来自池子里隐含的 regime 偏 + 形态偏
  - 20% 来自宽松期 beta
  - 10% 来自模型偶尔猜对 (运气, 非 alpha)

#### [当前规则下的"价值池"定义 (基于 2026-04-19 证据)]
- 池子 = `manual_bull` AND `WL > YL` AND `close > YL` AND `J < 14` AND `mv ≥ 100亿` AND `amt_ma20 ≥ 5000万`
- 一年大概 1~2 万次信号, 一天 50~100 只候选
- **池子内 top5 排序问题尚未解决** (目前只有规则筛选, 没有验证过的排序信号)

## 2026-04-18

### [B1] 10d/20d 双 horizon + Cohen's d 实测: R1 证伪, H2 强成立, 国轩高科冤案平反
- 已更新:
  - 本文件 (本条)
  - 后续待更新: `project-status.md`, `experiments/b1-next-phase.md`
- 已跑实验:
  - `notebooks/perfect_top10b1_analyze.py` 在 `ACTIVE_HORIZON = 10` 与 `ACTIVE_HORIZON = 20` 各跑一遍
  - 重点观测: Step A (case 自身 multi-horizon) / Step B (B1 占比 enrichment) / Step D (v2 max-archetype) / Step E (Cohen's d)

#### [发现 1] case 端在长 horizon 下显著放大, 国轩高科应保留
- Step A1 case/seed mfe ratio 演化: 5d=5.00x, 10d=5.84x, **20d=6.56x (峰值)**, 30d=6.17x, 40d=6.39x
- Step A3 spotlight `sz.002074` 国轩高科(趋势):
  - 5d/10d 几乎不动 (`fwd_mfe = 0.03 / 0.06`)
  - 15d 抬到 `0.15`, 20d 突跳 `0.38`, 30d 稳定 `0.71`
  - `fwd_mae = -0.012` 全程不变 → 完全无回撤, 教科书趋势型
- 不止国轩高科被 10d 标签截断, 至少 4 个 case 同病:
  - 华纳药厂 10d=0.22 → 15d=0.90 (4x)
  - 方正科技 10d=0.20 → 40d=1.37 (7x)
  - 澄天伟业 10d=0.23 → 15d=0.62 (3x)
  - 国轩高科 10d=0.06 → 30d=0.71 (12x)
- Step A4 各 horizon case_below_top10: `5/11 (10d) → 2/11 (15d) → 0/11 (30d)`
- 结论: case 数据本身没问题, 之前 (2026-04-17) 提议从 textbook cases 移除国轩高科的判断 **撤销**

#### [发现 2] R1 (拉长标签 horizon) 救不了反向富集 — R1 证伪
- Step B (`ACTIVE_HORIZON = 20`):
  - `is_textbook_b1=True` `mean_risk_adj_20d = 0.097`, **仍 < baseline 0.1088** (非B1 是 0.1109)
  - 6 档分箱 (textbook_b1_score 高 → 低): `0.20 → 0.14 → 0.12 → 0.11 → 0.10 → 0.10`, **仍严格单调递减**
  - Top 10% risk_adj_20d enrichment = **0.77x** (10d 时是 0.74x, 几乎没动)
- Step D v2 max-archetype (`ACTIVE_HORIZON = 20`):
  - v2 通过 86,120 / 179,963 (threshold_v2 = 0.7881)
  - `is_textbook_b1_v2=True` `mean_risk_adj_20d = 0.097`, 与 v1 完全相同
  - 6 档分箱: `0.18 → 0.15 → 0.12 → 0.10 → 0.09`, **仍单调递减**
  - Top 10% v2 enrichment = **0.79x**, 反向富集没改善
- 关键认识修正:
  - 案例端在 20d 翻盘 (Step A) ≠ "像案例的群体" 在 20d 翻盘 (Step B)
  - 这是两个独立现象, 之前 R1 期望被推翻
- R1 (改训练标签到 20d) 不再作为首选解药, 但仍可作为 case 自身评估口径

#### [发现 3] Cohen's d 锁死 H2 真凶: textbook14 选错了特征
- Step E 在 59 个数值特征上算 case vs seed_mid baseline 的 Cohen's d:
  - **|d| Top 5 全部不在 textbook14 内** (除 lower_shadow_pct):
    - `turnover_rate`           |d|=0.86 大效应  (rotation_core12, **不在 textbook14**)
    - `lower_shadow_pct`        |d|=0.65        (in textbook14)
    - `KLOW2`                   |d|=0.65        (alpha158_kbar, 与 lower_shadow_pct 数学等价)
    - `plry_cluster_recent_10`  |d|=0.62        (trigger_context, **不在 textbook14**)
    - `close_pos_in_bar`        |d|=0.60        (price_structure, **不在 textbook14**)
  - **case 上方差为 0 的硬约束 (textbook14 漏当 hard rule)**:
    - `bad_k_count == 0`         case_std=0, |d|=0.57
    - `trigger_recent_10 == 1`   case_std=0, |d|=0.45 (虽在 textbook14, 但当软相似度用了)
- textbook14 自身 Cohen's d 排序 (Step E2): 最大 0.65 (lower_shadow_pct), **没有任何特征 |d| ≥ 0.7**
- group 聚合 (Step E3) `mean_abs_d` 排序:
  - `trigger_context`  0.37  (textbook14 占 3/7)
  - `alpha158_kbar`    0.35  (textbook14 占 0/9)
  - `price_structure`  0.33  (textbook14 占 4/8)
  - `trend_strength`   0.29
  - `rotation_core12`  0.27  (textbook14 占 0/12)
  - `weekly_momentum`  0.16  (textbook14 占 2/5)
  - `volume_structure` 0.09  (textbook14 占 2/7) ← 最弱却占了 textbook14 一半权重
- 结论:
  - **H2 强成立** — textbook14 选了一个判别力较弱的特征子集, 真正区分 case 的因子在 rotation_core12 / alpha158_kbar / trigger_context 里, 且部分是 hard rule 不是 similarity
  - 之前 (2026-04-17) "如果连 Top 20 都没有强信号则收手" 的预案 **不触发** — Top 5 都有 ≥ 0.6 的有效信号

#### [发现 4] 数据卫生小 bug (vol_shrink_40 outlier)
- Step E2 `vol_shrink_40` 行 `seed_mean = 9443.99, seed_std = 78150` (case_mean = 0.22 合理)
- 推测: `utils/b1_feature_pool.py` 中 `_vol_max_40` 在 IPO 早期/长停复牌等边界股票退化为接近 0, 导致 `volume / _vol_max_40` 爆炸
- 影响: 不动 Cohen's d 的排序结论, 但训练前应 winsorize 或加下限
- 待修: 在 `utils/b1_feature_pool.py` 给 `_vol_max_40` 加 `max_horizontal(_, 100.0)` 之类下限, 或在 research_frame 出口处 `clip(0, 5)`

#### [当前下一步]
- (待评估) 草拟 `B1_TEXTBOOK_SCORE_FEATURE_COLS_V3`:
  - 加入 `turnover_rate` (单因子最强)
  - 加入 `plry_cluster_recent_10`, `close_pos_in_bar` (或等价 KSFT2/intraday_pos)
  - 砍掉 `vol_shrink_40`, `red_green_ratio_20` (volume_structure 整组 mean |d| 仅 0.09)
  - 把 `bad_k_count == 0`, `trigger_recent_10 == 1` 改成 hard rule (在 `_apply_textbook_structure_labels` 顶层 AND 进 is_textbook_b1)
- (待评估) 跑 v3 重做 Step B/D, 看反向富集是否翻正
- 修 `vol_shrink_40` outlier (纯卫生)
- `progress.md` 之前对国轩高科的"建议移除"判断作废, 不需要改 `manifests/b1_textbook_cases.py`

## 2026-04-17 (晚)

### [B1] 探索性诊断从 mining notebook 抽离到独立 perfect_top10b1_analyze
- 已更新:
  - `notebooks/perfect_top10b1_analyze.py` (整体重写, 1121 行, 承接所有探索)
  - `notebooks/b1_case_expansion_mining.py` (1732 → 989 行, 仅保留 mining 主流程)
- 当前动机:
  - `b1_case_expansion_mining.py` 在 Step 2b/2c/2d/2e 加入后越来越臃肿, 拖慢主流程跑动
  - `perfect_top10b1_analyze.py` 旧内容 (PERFECT_CASES_CONFIG + weekly MACD) 已过时, 推翻重做
- 当前 `perfect_top10b1_analyze.py` 结构:
  - `Step 0`  数据加载 + `build_b1_research_frame` 出 `df_seed / df_cases`
  - `Step 0b` **multi-horizon 标签** (5/10/15/20/30/40d 的 `fwd_mfe / fwd_mae / fwd_ret / fwd_mfe_risk_adj`), notebook 内基于 `q_full` 直接构建, 不动 `utils`
  - `Step A`  完美案例 vs seed_mid baseline 在各 horizon 的均值对比 (含 spotlight `sz.002074` 国轩高科的 horizon 演化, 回答"10d 是否太短")
  - `Step A4` 案例均值 vs Top10% seed 的"是否同档"判读 (各 horizon, 列出在该 horizon 仍跑不过 Top10% 的案例明细)
  - `Step B`  B1 形态占比与平均表现 (按 `ACTIVE_HORIZON` 切换, 默认 10d) — 迁移自旧 Step 2b
  - `Step C`  Textbook centroid 自洽性诊断 (H1) — 迁移自旧 Step 2c
  - `Step D`  v2 max-archetype 模拟 (H1 修复实验, 跟随 `ACTIVE_HORIZON`) — 迁移自旧 Step 2d
  - `Step E`  **Cohen's d 特征效应量排序 (Step 2f)** — 新增, 在全特征池上比较 `case` 与 `seed_mid` 均值差, 输出:
    - `[E1]` |Cohen's d| Top 30 (区分 textbook14 与非 textbook14)
    - `[E2]` 14 个 textbook 特征自身的 Cohen's d (按 |d| 降序)
    - `[E3]` 按 `B1_FEATURE_TO_GROUP` 分组聚合的 mean/max |d|
    - `[E4]` 自动判读: textbook14 在 Top30 命中数 → H2 (强成立 / 部分成立 / 不成立)
- 当前 `b1_case_expansion_mining.py` 现状:
  - 保留: 配置 / 数据加载 / Step 1 样本池 / Step 2 结果强样本过滤 / Step 3 案例相似度扩容 / artifact 导出
  - 删除: Step 2b/2c/2d/2e (全部迁到 perfect_top10b1_analyze)
  - 跑一次主流程不再被 740 行诊断输出干扰
- 当前下一步:
  - 用 `ACTIVE_HORIZON = 20` 重跑 perfect_top10b1_analyze, 看反向富集是否在 20d 标签下消失
  - 看 spotlight 表确认国轩高科 (`sz.002074`) 在 20d/30d 下 fwd_mfe 是否抬到合理强度, 决定是否仍移除
  - 跑 Step E (Cohen's d) 锁死 H2: 哪些非 textbook 特征才是真正区分案例的因子

### [B1] Case Expansion notebook 新增 H1/H2 假设检验链路 (Step 2c/2d/2e)
- 已更新:
  - `notebooks/b1_case_expansion_mining.py`
- 当前新增三步诊断 (顺序执行, 自动判决):
  - `Step 2c` Textbook centroid 自洽性诊断 (H1 验证)
    - 输出 11 个基础案例自身得分 / 14 维特征向量 / 每维 case 内分布 / 11x11 两两相似度矩阵
    - 实测结果: 11 个案例 5/11 自身被判 `is_textbook_b1=False`, `pairwise mean = 0.6889`, std = 0.0961
    - H1 (median centroid 把多 archetype 拍扁) 强成立, 但是否唯一根因待 Step 2d 确认
  - `Step 2d` v2 max-archetype 模拟 (H1 修复实验)
    - 公式 `textbook_b1_score_v2 = max_k mean_f clip(1 - |x[f] - case_k[f]| / scale[f], 0, 1)`
    - 阈值 LOO q20, scale 与 v1 完全一致 (隔离唯一变量)
    - 输出 4 张表 (mean perf / 6 档分箱 / enrichment / Top10 archetype 分布), 每张表自动判决
    - 实测结果: v2 enrichment Top10% = 0.75x (v1 是 0.74x), 6 档仍严格单调递减
    - **H1 已被证伪**: 改聚合方式 enrichment 一动没动, 反向富集与聚合方式无关
  - `Step 2e` 完美案例自身前瞻收益现实检验 (基础假设验证)
    - 输出 11 个案例的 fwd_ret_1d/2d/3d/5d/10d / fwd_mfe_10d / fwd_mae_10d / fwd_mfe_risk_adj_10d
    - 三组均值对比: cases / seed_mid baseline / seed_mid Top10%
    - 实测结果: case mean fwd_mfe_10d = **0.4595 (45.95%)**, baseline = 0.0787, Top10% = 0.2884
    - 案例 / baseline = **5.84x**, 案例 / Top10% = **1.59x**
    - 案例本身真实强 (10日窗口完整装下), 标签和案例对齐良好
- 当前判定结论:
  - 案例真实 ✓ + 标签正确 ✓ + 14 个 textbook 特征不能识别案例的强 ✗
  - **根因被锁死: 14 个特征本身没抓到使案例爆发的因子, 是特征语义错 (H2)**
  - 旁证: 11 个案例内部, `textbook_b1_score` 和 `fwd_mfe_10d` 基本不相关甚至负相关
    - 方正科技(蓄势) textbook 0.9104 → mfe 0.20
    - 昂利康(压轴) textbook 0.6306 → mfe 0.80
    - 新瀚新材(激进) textbook 0.7957 → mfe 0.99
- 当前数据卫生发现:
  - **国轩高科(趋势) `fwd_mfe_10d = 6.36%`, 比 baseline 7.87% 还低**, 不应作为完美案例
  - 后续 case set 清洗时建议移除或重新定位日期
- 当前下一步建议 (未做):
  - `Step 2f` 全特征池 |Cohen's d| 排序, 找出真正让 11 个案例与众不同的特征
  - 判定: |d| ≥ 0.8 大效应, 0.5~0.8 中效应, < 0.5 没有判别力
  - 如果 Top 20 里 14 个 textbook 特征一个都没进 → 特征选择从一开始就错
  - 如果连 Top 20 都没有强信号 → 案例的强属于不可建模偶然 (板块/消息), 收手

## 2026-04-16

### [B1] Case Expansion notebook 新增 B1 形态占比与平均表现概览
- 已更新:
  - `notebooks/b1_case_expansion_mining.py`
- 当前新增 `Step 2b`:
  - `[1]` seed_mid 全体 vs B1 形态 (is_textbook_b1) 的平均前瞻表现对比 (mfe_10d / mae_10d / risk_adj_10d / hit_10pct / hit_15pct)
  - `[2]` textbook_b1_score 分段平均表现 (6 档分箱)
  - `[3]` 强表现样本中 B1 形态占比 (enrichment 分析):
    - seed_mid 全体 baseline
    - 高于中位数 risk_adj
    - df_candidates (结果强样本)
    - Top 10% risk_adj
  - 自动判断 B1 形态在强样本中是否显著富集
- 当前目的:
  - 回答"形态像 B1 的样本是否真的比普通 seed_mid 表现更好"
  - 回答"在表现强的样本中，B1 形态是否显著富集"
  - 为后续教科书案例扩容和 Stage 1 结构标签的价值判断提供定量基线

### [B1] 双阶段 tail15 当前已收敛到 `payoff_score` 排序
- 已更新:
  - `progress.md`
  - `project-status.md`
- 当前结论:
  - `Step 4` 三路诊断已确认:
    - `payoff_score` 是强正排序信号
    - `structure_score` 对 `fwd_mfe_risk_adj_10d` 是负排序信号
    - `final_score = structure_score * payoff_score` 会把原本有效的 `payoff_score` 排序拉坏
  - `tail15 classifier` + `sort_field = "payoff_score"` 的最新六窗回测已全部转正
  - 但长窗最大回撤仍在 `30%~43%`, 暂不升格为稳定主线
- 当前收口口径:
  - 训练层保留双阶段:
    - `Stage 1`: 继续学 `is_textbook_b1`
    - `Stage 2`: 继续学 `P(fwd_mfe_risk_adj_10d >= 0.15)`
  - 组合层当前不再把 `structure_score * payoff_score` 视为默认排序主线
  - 当前研究默认排序字段改为 `payoff_score`
  - `structure_score` 暂时只保留为结构诊断与后续软过滤候选，不再作为收益排序乘子
- 当前建议:
  - 如果后续恢复探索，优先验证“`payoff_score` 直排 + 结构软过滤/弱加权”而不是继续调 `final_score` 绝对阈值

### [B1] Step 4 已支持 final / payoff / structure 三路分数并排诊断
- 已更新:
  - `notebooks/b1_seed_ml_baseline.py`
- 当前实现:
  - `Step 4` 不再只评估 `score`
  - 当前会同时输出:
    - `final_score` (`score`)
    - `payoff_score`
    - `structure_score`
  - `分层结果 / score 分位数 / threshold 表 / top-k 表` 都会附带 `score_source`
  - 当前目的:
    - 直接判断问题出在 `payoff_score`
    - 还是出在 `final_score = structure_score * payoff_score` 这个组合方式
- 当前验证:
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py` 通过
  - `uv run python -m py_compile notebooks/b1_seed_ml_baseline.py` 通过

### [B1] Stage2 已从连续回归切到 strong-tail classifier
- 已更新:
  - `notebooks/b1_seed_ml_baseline.py`
- 当前实现:
  - `Stage 2` 当前默认不再回归连续 `fwd_mfe_risk_adj_10d`
  - 改为二分类尾部目标:
    - `STAGE2_MODEL_MODE = "tail_classifier"`
    - `STAGE2_TAIL_THRESHOLD = 0.15`
  - 当前最终分数口径变为:
    - `final_score = P(is_textbook_b1) * P(fwd_mfe_risk_adj_10d >= 0.15)`
  - 保留现有 `score / structure_score / payoff_score` 导出字段，避免影响 Rust 导出与回测链路
  - 训练 metadata 当前已新增:
    - `stage2_model_mode`
    - `stage2_tail_threshold`
  - `run_label` 当前会显式带上 `tail15`
- 当前目的:
  - 不再让 Stage 2 追求“平均收益回归”
  - 直接把训练目标收口到 `15%+` 强尾部样本识别
- 当前验证:
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py` 通过
  - `uv run python -m py_compile notebooks/b1_seed_ml_baseline.py` 通过

### [B1] Stage2 训练门槛与 Step 4 尾部命中诊断已收紧
- 已更新:
  - `notebooks/b1_seed_ml_baseline.py`
- 当前实现:
  - `Stage 2` 当前新增训练门槛: `fwd_mfe_risk_adj_10d >= 0.10`
  - `Step 4` 不再只看单一 `hit_7%`, 当前同时输出:
    - `hit_7pct`
    - `hit_10pct`
    - `hit_15pct`
    - `hit_18pct`
  - `score threshold` 与 `top-k` 表都会同步展示上述 4 档命中率
  - 训练 metadata 当前已新增:
    - `stage2_min_label_for_train`
    - `eval_hit_thresholds`
- 当前目的:
  - 减少 `Stage 2` 被中等强度样本稀释
  - 直接观察模型是否真的把 `10%~18%` 的强尾部样本压到更前排
- 当前验证:
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py` 通过
  - `uv run python -m py_compile notebooks/b1_seed_ml_baseline.py` 通过

### [B1] 双阶段训练已接入 textbook / recent / tail sample_weight
- 已更新:
  - `notebooks/b1_seed_ml_baseline.py`
- 当前实现:
  - 训练入口新增可配置样本加权口径:
    - `base textbook case = 3.0x`
    - `expanded textbook case = 2.0x`
    - `recent sample = 1.5x @ 2022-01-01+`
    - `stage2 tail label = 2.0x @ >=15%` (`tail_classifier` 默认口径)
  - `Stage 1 (is_textbook_b1)` 当前使用 `base / expanded / recent` 权重
  - `Stage 2 (fwd_mfe_risk_adj_10d)` 当前在 `base / expanded / recent` 之外，再额外加 `tail` 权重
  - 每次 walk-forward 重训日志会输出 `stage1_w_mean / stage2_w_mean / stage2_tail_cut`
  - artifact `train metadata` 已落盘 `sample_weighting` 配置，后续回看训练产物时可直接追溯
- 当前目的:
  - 不再让模型平均学习全部 seed 样本
  - 明确把训练重心压向“更像 textbook 且 recent 更有效”的 tail filter
- 当前验证:
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py` 通过
  - `uv run python -m py_compile notebooks/b1_seed_ml_baseline.py` 通过

## 2026-04-14

### [B1] Rotation Core12 + KBAR 已接入 B1 Lab 实验特征集
- 已更新:
  - `utils/b1_feature_pool.py`
  - `notebooks/b1_condition_mining.py`
- 当前实现:
  - `build_b1_research_frame()` 会在原有 `B1` 研究底表上额外计算 `Rotation core_12` 与 `Alpha158 kbar_shape` 9 因子
  - `B1_FEATURE_SET_REGISTRY` 已新增 `rotation_core12_kbar`
  - `B1_FEATURE_GROUPS / B1_FEATURE_GROUP_LABELS` 已补 `Rotation Core12` 与 `Alpha158 KBAR` 分组
  - `b1_condition_mining.py` 当前默认 `ANALYSIS_FEATURE_SET_NAME = "rotation_core12_kbar"`
  - `TRAIN_FEATURE_SET_NAME = "selected"` 保持不变，不影响 `b1_seed_ml_baseline.py` 训练默认
- 当前目的:
  - 只在 `Lab` 里验证“Rotation 强因子”迁移到 `B1` 后是否提供增量信息
  - 避免把探索性结论误升格为 `ML B1` 的默认训练主线

### [B1] Factor Lab 已兼容 analysis/train 分离展示
- 已更新:
  - `notebooks/b1_condition_mining.py`
  - `utils/ic_analysis.py`
  - `utils/factor_analysis.py`
- 当前修复:
  - 当 `ANALYSIS_FEATURE_SET_NAME` 与 `TRAIN_FEATURE_SET_NAME` 不同，`Step 5 / 9 / 10` 不再把训练冻结集误当成 `ICIR=0`
  - 分组汇总改为只展示当前 analysis 特征集实际命中的 group，避免大量 `0 因子` 空组
  - 冻结训练集画像改为基于 `analysis + train` 并集补算，替换建议会用真实训练特征强弱做比较
  - 多周期衰减里的常量输入 `Spearman` 警告已跳过，不再额外刷 warning

### [B1] 已新增 `selected_rotation_hybrid_v1` 实验冻结集
- 已更新:
  - `utils/b1_feature_pool.py`
  - `utils/__init__.py`
- 当前实现:
  - 新增实验特征集 `selected_rotation_hybrid_v1`
  - 由 `12` 个特征组成:
    - `Bias_WL_YL`
    - `Bias_C_YL`
    - `rw_dif_pct`
    - `rw_hist`
    - `rm_hist`
    - `body_pct`
    - `vol_shrink_40`
    - `atr_14_pct`
    - `KLEN`
    - `vol_60d`
    - `turnover_rate`
    - `KUP`
- 当前目的:
  - 将 `B1` 骨架特征与本轮 `Rotation/KBAR` 最强迁移因子做一轮受控融合
  - 保持默认 `selected` 不变，方便在 `b1_seed_ml_baseline.py` 中做 A/B 训练与 Rust 回测对照

### [B1] 训练 notebook 已自动识别 Rotation/KBAR 特征依赖
- 已更新:
  - `utils/b1_feature_pool.py`
  - `utils/__init__.py`
  - `notebooks/b1_seed_ml_baseline.py`
- 当前修复:
  - 新增 `b1_feature_set_requires_rotation_kbar()`
  - `b1_seed_ml_baseline.py` 会根据 `FEATURE_SET_NAME` 自动判断是否需要额外计算 `Rotation core_12 + KBAR`
  - 当使用 `selected_rotation_hybrid_v1` 或其他含迁移因子的特征集时，不再出现 `ret_max_5d / atr_14_pct / KLEN` 等列缺失报错

### [Docs] B1 主文档已继续收口“默认主线”措辞
- 已进一步更新:
  - `experiments/b1-next-phase.md`
  - `project-status.md`
- 当前新增统一口径:
  - `selected` / `seed_strict` / `fwd_mfe_10d` 仍可作为研究默认配置
  - 但不再等价于默认实盘主线
  - `ML B1` 当前保留为研究链，规则链仍是最稳的可执行链

### [Docs] B1 口径已重写为“hindsight 上界 vs 机械基线”
- 已更新:
  - `experiments/b1-next-phase.md`
  - `project-status.md`
- 当前统一新口径:
  - 旧手工 `LOOSE_PERIODS` 高收益结果只作为 **artificial hindsight upper bound / oracle regime benchmark**
  - 新的完全机械 `活跃市值` 规则结果才是 **客观可执行基线**
- 当前结论:
  - `ML B1` 的旧高收益不再直接视为可落地收益率
  - 当前系统级问题更像是 `活跃市值` 作为硬开仓开关不够稳，而不是冻结特征集的尾部细节

### [B1] 手工活跃市值 regime 日期已在 Lab / Train 间对齐
- 已将 `notebooks/b1_condition_mining.py` 里的 `LOOSE_PERIODS` 同步到与 `notebooks/b1_seed_ml_baseline.py` 一致的手工区间列表。
- 当前目的:
  - 避免 `Lab` 与 `Train` 使用两套不同的宽松期口径，导致因子分析、训练结果和回测结论彼此不可比。
  - 后续所有关于 `活跃市值` 的讨论先基于同一份日期事实，再判断这套机械择时本身是否有效。

## 2026-04-12

### [B1] baseline notebook 已补最新交易日打分与 threshold/top-k 诊断
- `notebooks/b1_seed_ml_baseline.py` 当前已将 `Walk-Forward` 的训练样本与打分样本拆开:
  - 训练仍只使用 `LABEL_COL` 非空的历史样本
  - 打分则覆盖全部特征齐全的 `seed` 候选日期, 包括最后一个尚无 `fwd_mfe_10d` 标签的交易日
- 当前目的:
  - 保持原有 `20` 个交易日重训节奏不变
  - 修复“行情更新到最新收盘后, 最后一天却没有 score, 无法生成次日候选”的致命问题
- `Step 4. 纯模型基线评估` 当前已新增:
  - `score` 分位数表
  - `score threshold` 表现表
  - `top-k` 表现表
- 目的:
  - 直接支持“要不要设开仓阈值”和“冷启动是否买满 `5` 只”的定量判断

### [Docs] B1 主文档已补充实盘前待办与运行口径
- 已在 `experiments/b1-next-phase.md` 补充:
  - `ML B1` 用 `10万元` 作为实验仓试运行的前提
  - 当前模型仍按约 `20` 个交易日重训一次，而非日更重训
  - 当前日常运行链“必须从头重跑整套流程才能刷新次日候选”的不灵活点
  - 周一前优先待办: `2026` 近端利润回吐复盘、`score threshold` 分析、`selected` 一轮受控迭代、冷启动分批建仓
- 已同步更新 `project-status.md` 的 `B1 实盘前待办` 摘要

### [B1] `SELL / CLOSE` 日志已拆分清仓收益与整笔收益
- `backtest-engine/crates/b1/src/systems.rs` 与 `backtest-engine/crates/b1/src/main.rs` 当前已将最终平仓日志改为同时显示:
  - `ExitPnL`: 本次清仓这最后一腿的收益
  - `TradePnL`: 该标的从建仓到结束的累计总收益
- 当前日志也把原来的裸 `TP1 / TP2` 标记改成了 `Stage: None/TP1/TP2`
- 目的:
  - 避免把最终 `[SELL]` / `[CLOSE]` 行里的收益误读为“只代表最后一次卖出”
  - 让分批止盈后的最终清仓日志更直观可读

### [Docs] B1 六窗对比已切换为修复后 ML 结果
- 已基于修复后的 `notebooks/b1_seed_ml_baseline.py` 导出结果，回填最新六窗 `ML` 回测结论到:
  - `experiments/b1-next-phase.md`
  - `project-status.md`
- 当前文档口径:
  - 规则版列沿用上一轮已确认结果
  - `ML` 列改为修复“持仓后续行情丢失”后的新结果
  - 旧文档中 `ML` 的 `24%~29%` 大回撤数字不再引用
- 当前结论更新为:
  - `2022~2025` 的 `ML MDD` 已收敛至 `9%~10%`
  - `2021` 全窗 `ML MDD` 为 `19.32%`
  - `2026` 近端窗口为 `+0.57% / 8.60%`，且最大回撤尚未恢复

### [B1] ML 导出底座已修复“持仓后续行情丢失”问题
- 问题现象:
  - `notebooks/b1_seed_ml_baseline.py` 之前直接用 `df_all` 导出 `signal.parquet`
  - 而 `df_all` 会受 `MV_MIN / MV_MAX / MIN_LIST_DAYS` 研究过滤影响
  - 导致已买入股票在后续日期若跌出研究 universe，会从导出 parquet 中直接消失
- 当前修复:
  - 导出底座改为 `calc_b1_factors_wmacd(q_full, {"MV_THRESHOLD": MV_MIN})`
  - 再把 `df_all` 里的 `seed_mid/seed_strict` 信号与 `df_scores_raw.score` 左连接回完整行情底座
  - 即:
    - 行情与持仓估值使用完整后续价格
    - 候选池限制仍然只由研究过滤后的 signal/score 决定
- 当前目的:
  - 防止已持仓股票因研究过滤而在 Rust 回测中丢失估值
  - 避免 `Total Asset / update_stats / Max Drawdown` 被错误低估

### [B1] `TP1 / TP2` 日志口径已改为显示剩余仓位
- `backtest-engine/crates/b1/src/systems.rs` 当前已将分批止盈日志从 `Sold x/y shares` 调整为:
  - `Sold N shares | Remaining M/Initial`
- 目的:
  - 明确 `TP1 / TP2` 是按初始仓位的固定三分之一分批卖出
  - 避免把第二次 `TP2` 的 `1300/3900` 误读成“当前剩余仓位仍是 3900”

### [B1] 交易日志已补充 `Total Asset` 与统一收益口径
- 当前 `BUY / TP1 / TP2 / SELL / CLOSE` 日志都会输出成交后的 `Total Asset`
- 当前全平仓类日志已统一显示:
  - 绝对收益 `PnL`
  - 百分比收益 `PnL%`
- 当前 `EndOfBacktest` 强平行也会显式打印原因，避免把最后一条 `CLOSE` 误解成“尚未结算”

### [B1] 回测报告已补充最大回撤的峰值/谷底/恢复日期
- 已在 `backtest-engine/crates/core/src/lib.rs` 扩展 `BacktestStats::update_drawdown()`:
  - 保持原有 `max_drawdown` 数值口径不变
  - 新增记录 `max_drawdown` 对应的 `peak / trough / recovery` 日期
- 当前 `bt-b1 / bt-rotation / bt-renko` 的 `update_stats()` 已统一透传 `current_date`
- 当前输出已补充到:
  - `report.txt`
  - `report.json.metrics`
- 当前默认仍**不**写入 `backtest.jsonl` registry，先保持 registry 结构稳定
- 当前已验证:
  - `cargo test --manifest-path /Users/zhangyubo/Projects/QuantLab/backtest-engine/Cargo.toml -p bt-core`
  - `cargo check --manifest-path /Users/zhangyubo/Projects/QuantLab/backtest-engine/Cargo.toml -p bt-b1 -p bt-rotation -p bt-renko`

### [B1] 六组规则版 vs ML 回测对比已收口为统一结论
- 已将 `2021-06-21 / 2022-01-01 / 2023-01-01 / 2024-01-01 / 2025-01-01 / 2026-01-01` 六组起始日期的 `B1` 回测结果统一整理到:
  - `experiments/b1-next-phase.md`
  - `project-status.md`
- 当前统一结论:
  - `seed_mid + ML score` 在 `2021~2025` 五组窗口里均显著跑赢规则版，当前仍是 `B1` 主线
  - 规则版更适合作为低回撤基线与 regime sanity check
  - `2026-01-01` 超短窗口里规则版阶段性反超，说明 `ML` 当前更需要解决的是近端 regime 适应与回撤控制
- 当前下一步优先级:
  - 优先压 `ML` 回撤
  - 观察近端 regime 失配
  - 不再重复做“ML 是否优于规则版”的主结论验证

## 2026-04-10

### [B1] 配置文件已收口为 rule / ml 两套
- `backtest-engine/crates/b1` 当前仅保留:
  - `config.toml`: 规则版默认配置，`sort_field = "rw_dif_pct"`
  - `config_ml.toml`: ML 版默认配置，`sort_field = "score"`
- 已清理旧命名:
  - `config_wmacd.toml`
  - `config_wmacd_ml.toml`
- `scripts/b1_backtest.py` 当前已支持:
  - 不传 `--config` 时，按 signal metadata 自动选择 `config.toml / config_ml.toml`
  - 使用 `--start-date YYYY-MM-DD` 覆盖回测起始时间
  - 使用 `--min-score 0.0641` 覆盖 `backtest.min_score`

### [B1] `bt-b1` 已修复同日买入又同日卖出的 T+1 bug
- 问题现象:
  - `process_buy_signals()` 在开盘建仓后
  - `check_sell_conditions()` 又在同日收盘对新仓执行 `TP1 / TP2 / 止损 / Weak / Trailing`
  - 导致规则版与 ML 版都会出现“同日买入、同日卖出”的错误成交
- 当前修复:
  - `backtest-engine/crates/b1/src/systems.rs` 已增加 `hold_days <= 0` 直接跳过卖出逻辑
  - 即当日新仓只允许留到下一交易日后，才进入任何卖出判断
- 当前已补回归测试:
  - `does_not_sell_on_entry_day_even_if_take_profit_hits`
  - `can_take_profit_after_entry_day`
- 当前已验证:
  - `cargo test --manifest-path /Users/zhangyubo/Projects/QuantLab/backtest-engine/Cargo.toml -p bt-b1`

### [B1] 规则版导出已接入 artifact，并与 ML 版显式区分
- `notebooks/simple_b1_lab.py` 当前已改为通过 `artifact_metadata` 导出规则版 `B1`
- 当前规则版导出约定:
  - 不再额外写 `data/signals/...` alias parquet，`artifacts/b1/.../signals/.../signal.parquet` 为唯一真源
  - `train_run_id / signal_source / model_name / feature_mode` 均显式标记为规则链
  - 回测结果会继续落回对应的 `artifacts/b1/<train_run_id>/signals/<signal_id>/backtests/...`
- `scripts/b1_backtest.py` 的交互选择列表当前已改为直接显示:
  - `source`
  - `model`
  - `label`
  - `feature_set`
- 当前目标:
  - 让规则版 `B1` 与 `seed + ML score` 版共用同一套回测入口
  - 但在导出、选择、回测登记三个层面都能一眼区分

### [B1] Artifact 导出与选择式回测已接入第一版
- `utils/signal_export.py::export_for_rust()` 当前已支持 `artifact_metadata`:
  - 训练后可直接落盘到 `artifacts/b1/<train_run_id>/...`
  - 生成 `train.meta.json` / `signal.meta.json` / `signals.jsonl` / `backtest.jsonl`
  - 可选同时写出 `data/signals/...` alias parquet
- `backtest-engine/crates/b1` 当前已接入 `bt-core` 的 signal meta / report bundle / registry append
- 已新增 `scripts/b1_backtest.py` 与 `backtest-engine/run_b1.bat`:
  - 支持 `--pick` 交互式选择 B1 signal
  - 自动创建 `backtests/<timestamp>/effective.config.toml`
  - 回测结果自动写回对应 signal 目录
- 当前已验证:
  - `python -m py_compile utils/signal_export.py`
  - `python -m py_compile notebooks/b1_seed_ml_baseline.py`
  - `python -m py_compile scripts/b1_backtest.py`
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py`
  - `cargo check -p bt-b1`

### [B1] Seed 纯模型导出已切到“seed 信号 + score 排序”口径
- 已更新 `notebooks/b1_seed_ml_baseline.py` 的 `Step 5. 导出到 Rust`
- 当前导出逻辑改为:
  - 使用 `build_b1_research_frame()` 产出的研究底表作为导出底座
  - 直接将 `SEED_COL` 映射为 parquet 里的 `b1_signal`
  - `score` 继续作为额外排序列导出给 Rust
- 当前目标:
  - 保持 `bt-b1` 的时钟流、买卖规则与成本模型不变
  - 仅替换“谁是候选”与“候选内排序”两件事
  - 让规则版 `B1` 与 `seed + ML score` 版只通过不同 parquet 做可比回测

### [B1] Seed 训练入口已补齐 walk-forward 进度打印
- `notebooks/b1_seed_ml_baseline.py` 的 `Step 3. LightGBM Walk-Forward` 当前已新增:
  - 训练窗口 / 重训频率 / 标签 / 特征集打印
  - 每次重训时打印当前日期、进度百分比与训练样本量
  - 训练结束后打印最终打分条数
- 当前效果对齐 `notebooks/cross_section_rotation.py` 的训练入口风格，便于长时间运行时观察进度

### [B1] `selected` 冻结特征集已做第一轮保守升级
- 基于 `notebooks/b1_condition_mining.py` 在 `seed_mid + bull_only + fwd_mfe_10d` 下的新一轮 lab 结果，已更新 `utils/b1_feature_pool.py` 中的 `B1_SELECTED_FEATURE_COLS`
- 本轮调整:
  - 移除较弱尾部: `Bias_C_WL` / `red_green_ratio_20` / `days_since_key_k` / `bias_wl_yl_delta_5`
  - 新增更强候选: `body_pct` / `vol_shrink_40` / `rw_hist_delta_5` / `rm_hist_delta_5`

### [B1] Factor Lab 的 freeze 建议逻辑已改为“弱尾替换”
- `notebooks/b1_condition_mining.py` 的 `Step 10. Lab 结论` 不再只做“旧冻结集 + group top 后截断”
- 当前改为:
  - 先保留相关性诊断后未被剪掉的冻结列
  - 再按 `abs_ICIR` 用更强的 watchlist / group top 候选替换最弱尾部
  - 额外打印 `replacement_candidates / recommended_drops / recommended_adds / suggested_next_freeze`
- 当前已验证:
  - `python -m py_compile utils/b1_feature_pool.py`
  - `python -m py_compile notebooks/b1_condition_mining.py`
  - `uv run marimo check notebooks/b1_condition_mining.py`

## 2026-04-09

### [B1] Lab / Train 拆分已落地主入口
- 已将 `notebooks/b1_condition_mining.py` 重构为纯 `B1 factor lab`:
  - 主线改为 `seed overview -> IC -> group summary -> bin scoreboard -> decay -> corr diagnostics`
  - 不再把浅树规则、手工规则验证和条件收敛写在主线里
  - 训练入口需要的冻结特征观察也改为 lab 内单独输出
- 当前已验证:
  - `python -m py_compile notebooks/b1_condition_mining.py`

### [B1] 冻结特征集已接入训练入口
- `utils/b1_feature_pool.py` 当前已统一维护:
  - `core`
  - `candidate`
  - `selected`
- `notebooks/b1_seed_ml_baseline.py` 已改为默认消费 `FEATURE_SET_NAME="selected"`
- 当前训练 notebook 已进一步收敛为纯入口:
  - 读底表
  - 训练 `LightGBM walk-forward`
  - 输出 `IC / ICIR / q4-q0`
  - 导出 Rust parquet
- 当前导出文件名已带上 `feature_set`
- 当前已验证:
  - `python -m py_compile notebooks/b1_seed_ml_baseline.py`

### [Docs] B1 叙事已切换为 `Lab -> Train -> Export`
- 已重写 `experiments/b1-next-phase.md`
- 已更新 `project-status.md`
- 当前统一口径:
  - 规则链独立运行
  - ML 链严格按 `Lab -> Train -> Export` 拆分
  - lab 先证明因子，再手工更新 `selected`，最后由训练 notebook 稳定消费

### [B1] 双轨研究路线第一轮已落地到代码
- 已新增共享特征工具 `utils/b1_feature_pool.py`:
  - 统一输出 `B1` 条件挖掘与 seed 纯模型共用的研究底表
  - 保留原第一批连续特征
  - 新增第二批 `17` 个 B1 专属连续特征，覆盖:
    - 触发上下文 (`trigger_recent_10` / `key_k_recent_20` / `plry_cluster_recent_10` / `days_since_key_k_inv`)
    - 形态结构 (`range_pct` / `body_to_range` / `close_pos_in_bar` / `gap_from_prev_close_pct`)
    - 量价结构 (`vol_to_prev_vol` / `vol_to_avg40` / `vol_shrink_20_delta_5` / `red_green_ratio_delta_5`)
    - regime / 周月动能 (`rw_hist_delta_5` / `rm_hist_delta_5` / `bias_wl_yl_delta_5` / `close_above_yl_pct_5` / `close_above_wl_pct_5`)
- 已更新 `utils/__init__.py` 导出新的 B1 特征池与底表构造函数

### [B1] `b1_condition_mining` 已进入“候选条件收敛”阶段
- `notebooks/b1_condition_mining.py` 已改为直接复用 `build_b1_research_frame()`
- 当前 notebook 已从“第一批最小特征集”升级为“第一批 + 第二批”统一特征池
- 已新增 `Step 8. 候选条件收敛`:
  - 自动汇总 `Step 7` 浅树候选
  - 自动汇总 `Step 7b` 手工规则验证
  - 自动收敛出 `2~5` 条可解释候选条件，供下一轮规则增强或纯模型对照直接复用
- 当前已验证:
  - `python -m py_compile notebooks/b1_condition_mining.py`
  - `uv run marimo check notebooks/b1_condition_mining.py`

### [B1] seed 纯模型对照线已打通第一版
- 已新增 `notebooks/b1_seed_ml_baseline.py`
- 当前支持:
  - 直接切换 `SEED_COL="seed_mid"` 或 `SEED_COL="seed_strict"`
  - 在 seed 内做 `LightGBM walk-forward`
  - 输出 `IC / ICIR / t-stat / q4-q0`
  - 导出 `score` 到 Rust B1 回测 parquet
- 当前已验证:
  - `python -m py_compile notebooks/b1_seed_ml_baseline.py`
  - `uv run marimo check notebooks/b1_seed_ml_baseline.py`

### [Docs] B1 路线文档已收敛为单一入口
- 已将原先分散的“条件挖掘计划文档”和“四路线对比板”合并回 `experiments/b1-next-phase.md`
- 当前 `b1-next-phase.md` 已统一承载:
  - 双轨研究框架
  - 最新条件挖掘结论
  - 最小四路线对照集
  - 当前推荐执行顺序

### [Deps] `uv` 默认源已收敛为项目级配置
- 已在 `pyproject.toml` 中新增 `[[tool.uv.index]]` 默认源:
  - `https://mirrors.aliyun.com/pypi/simple/`
- 当前目标:
  - 固定项目级默认 index
  - 避免 macOS / Windows 因各自终端里的 `UV_INDEX_URL` 导致 `uv.lock` 来回漂移
- 当前保留:
  - `torch / torchvision` 在 Windows 下继续显式走 `pytorch-cu130`

## 2026-04-08

### [B1] `b1_condition_mining` 第一版 notebook 已开工落地
- 已新增 `notebooks/b1_condition_mining.py` 第一版骨架，当前直接复用 `calc_b1_factors_wmacd()` 作为底层主链，不再另写一套 `B1` 计算逻辑
- 当前 notebook 已包含最小研究闭环:
  - 原始数据加载 + `ST` 过滤
  - `B1` 主链因子计算
  - 三档 `seed pool` (`seed_loose / seed_mid / seed_strict`)
  - 第一批连续特征列
  - `manual bull regime` 标注
  - 单变量分箱得分榜
  - 指定特征深挖
  - 浅树候选规则提取
  - `Step 7b` 手工候选规则验证
- 已进一步收敛输出风格:
  - 改回 `print-first`，更接近 `cross_section_rotation.py`
  - 将关键结论和关键表格直接打印在各 step 内
  - 方便直接复制终端输出到对话框继续分析
- 当前已验证:
  - `python -m py_compile notebooks/b1_condition_mining.py`
  - `uv run marimo check notebooks/b1_condition_mining.py`

### [B1] 条件挖掘计划已收敛为第一阶段最小实施版
- 已收敛到统一 B1 路线文档，当前第一阶段只保留:
  - 三档 `seed pool` (`loose / mid / strict`)
  - 第一批最小连续特征集 (`12~18` 个)
  - 单变量 / 双变量稳定性分析 + 浅树规则提取
  - 三条最小对照基线
- 当前明确延后:
  - `manifest` 注册
  - 直接回写 `utils/b1_factors_opt.py`
  - 大规模 rule sweep
- 当前目标:
  - 先产出 `2~5` 组可解释候选条件
  - 再决定哪些值得接入全市场 ML 精排主线

### [Docs] 新增 B1 下一阶段路线文档
- 已新增 `experiments/b1-next-phase.md`，集中记录:
  - `B1` 当前可执行主链
  - `活跃市值` 继续手工判断、不做自动化的约定
  - `规则召回 / 全市场 ML 精排 / Agent 辅助` 三条可能路线
  - 重新切回 `B1` 主线前需要确认的准备事项
- 当前文档口径已明确:
  - `B1` 若重新升回主线，优先考虑“规则候选 + 全市场 ML 排序”的分层方案
  - `B1` 专属 ML 暂不作为默认主线

## 2026-04-07

### [Rotation] Top-20 优化阶段先验证出 `hold_buffer=20` 的主效应
- 基于 `Cell 8` 的 `Top-20 Tail Diagnostics`，当前信号已确认呈现明显前排集中:
  - `Top1-2` 最强
  - `Top16-20` 已明显弱于前排
  - 当前问题首先更像“旧仓卖得太慢”，而不只是“新仓怎么进”
- 回测层新加的两条能力:
  - `max_daily_buys`
  - `entry_rank_limit`
  已完成落地，但本轮实验尚未证明它们本身就是收益提升主因
- 本轮关键对比结论:
  - 当 `max_daily_buys=20` 且 `entry_rank_limit=20` 时，新机制已退化回旧买入行为
  - 在这一前提下，仅将 `hold_buffer` 从 `50` 收紧到 `20`，回测表现明显改善
  - 当前更应优先把 `hold_buffer=20` 视作新的强候选基线，再评估更复杂的入场控制
- 当前阶段判断:
  - `hold_buffer=20` 是 Top-20 优化的第一有效改动
  - `max_daily_buys / entry_rank_limit` 目前保留为后续实验工具，不作为本轮主结论

### [Rotation] 回测层已支持渐进建仓与前排准入
- `backtest-engine/crates/rotation` 已新增两条独立控制:
  - `max_daily_buys`: 每日最多新开仓数量, 用于控制建仓节奏
  - `entry_rank_limit`: 仅允许新开仓来自前排 rank, 用于控制准入质量
- 当前规则语义已明确拆开:
  - `max_positions` = 组合容量上限
  - `max_daily_buys` = 每日新增节奏
  - `entry_rank_limit` = 新仓质量门槛
  - `hold_buffer` = 已持仓保留阈值
- 兼容性处理:
  - 旧配置文件即使未声明新字段也可继续运行
  - `max_daily_buys` 默认回落到 `max_positions`
  - `entry_rank_limit` 默认回落到 `top_n`
- 已同步更新:
  - `backtest-engine/crates/rotation/config.toml`
  - 回测控制台配置输出
  - report bundle / registry 元数据
- 已验证:
  - `cargo check -p bt-rotation`
  - 仅有 rotation crate 既有 dead_code warning, 本次未新增编译错误

### [Rotation] 训练 notebook 已新增 Top-20 专用诊断面板
- `notebooks/cross_section_rotation.py` 已在 `Cell 7` 后新增 `Top-20 Tail Diagnostics`:
  - `Top-20` 日均收益 / 中位数 / 单票胜率
  - `Rank 1-20` 的逐名次日均收益
  - `Rank 1-5 / 6-10 / 11-15 / 16-20` 分桶收益
  - `Top-20` vs `Rank 21-40` 的日均收益与累计收益
  - `Rank20-21` 的 score gap
  - `Top-20` 最差 `1` 只 / 最差 `3` 只的日均拖累
- 当前设计选择:
  - 与 `Cell 7` 统一口径, 继续基于 `df_scores_raw` 做诊断侧 EMA
  - 固定按 `fwd_ret_1d` 做经济评估, 不改训练和导出主流程
- 已验证:
  - `python -m py_compile notebooks/cross_section_rotation.py`
  - `uv run marimo check notebooks/cross_section_rotation.py`

### [Rotation] 自定义因子组合训练入口落地
- `manifests/rotation_feature_sets.py` 已新增运行时 `custom feature set` 解析能力:
  - 支持在训练 notebook 中直接传入任意 `Rotation / Alpha158` 因子组合
  - 自动推导:
    - `feature_mode`
    - `alpha158_group_mode`
    - 未知因子校验
- `notebooks/cross_section_rotation.py` 当前已支持:
  - `FEATURE_SET = "custom"`
  - `CUSTOM_FEATURE_SET_NAME`
  - `CUSTOM_FEATURE_COLS`
- 当前设计口径:
  - 训练层仍优先消费 manifest 中冻结的稳定特征集
  - 但已不再限制只能跑 registry 里预先写死的组合
  - 对跨 Alpha158 分组的自定义组合, 训练层会自动推导所需分组并准备依赖因子

### [Rotation] `rotation_factor_lab.py` marimo 兼容性修复
- 已用 `uv run marimo check` 复现并修复 Factor Lab 中的典型 marimo 问题:
  - 跨 cell 重复定义局部变量
  - cell 分支中的早退 `return`
  - 空 cell 告警
- 当前结果:
  - `uv run marimo check notebooks/rotation_factor_lab.py` 已通过
  - `uv run marimo check notebooks/cross_section_rotation.py` 已通过
- 顺手做的结构清理:
  - Factor Lab 中各 cell 的临时变量统一改为私有命名
  - 训练入口说明补充 `selected_feature_set.note`
  - `rotation_train_meta` 新增 `feature_set_name`

## 2026-04-05

### [Docs] 跨设备统一口径固化
- 为避免不同设备、不同 agent 把 `Rotation` 与博主完整体系混为一谈, 已将统一口径显式写入:
  - `experiments/rotation-next-phase.md`
  - `experiments/target-strategy-evolution.md`
  - `project-status.md`
- 当前统一约定:
  - `Rotation = 候选子策略`
  - “对标”默认指向博主早期公开的日频截面基线
  - 博主当前多策略 `rule-based` 体系属于系统级长期目标

### [Docs] Rotation 主线/锚点/废弃路线再收口
- `experiments/rotation-next-phase.md` 已进一步整理为 agent 入口页风格:
  - 新增 `当前活跃路线`
  - 新增 `已收口 / 后置路线`
  - 新增 `历史归档说明`
  - 历史结论统一降级为按日期归档, 避免新 agent 把旧实验误判成当前主线
- `project-status.md` 已同步补充:
  - 当前活跃路线
  - 已收口路线

### [Rotation] 三层解耦第一阶段落地
- `Rotation` notebook 已按“分析层 / 训练层 / 清单层”开始拆分:
  - `notebooks/rotation_factor_lab.py`: 独立分析 notebook, 专门负责
    - 因子 IC
    - 分组汇总
    - Alpha decay
    - Alpha158 top1 / 强子集筛选
  - `manifests/rotation_feature_sets.py`: Python manifest, 作为训练层唯一稳定特征集来源
  - `notebooks/cross_section_rotation.py`: 收敛为训练入口 notebook
- `cross_section_rotation.py` 当前改动:
  - 不再依赖 `Cell 3` 现场产出的 `alpha158_top1_factor_cols / core_factors / factors_keep`
  - 训练入口改为直接读取 `FEATURE_SET`
  - 当前默认主线改为 `core_plus_alpha158_kbar_shape`
- manifest 当前已显式区分:
  - `active`: `core_12`, `core_plus_alpha158_kbar_shape`
  - `archived / experimental`: `all_rotation`, `alpha158_kbar_shape`, `all_plus_alpha158_kbar_shape` 等
  - `analysis-only`: `core_plus_alpha158_top1`, `pruned_rotation`
- 元数据兼容约束保持不变:
  - `rotation_train_meta` 结构未改键名
  - `Cell 6b` 仍通过 `export_meta = {**rotation_train_meta, ...}` 导出
  - `utils/signal_export.py` 未改 artifact metadata 消费契约

### [Benchmark] 目标策略认知修正
- 重新审视小红书博主公开帖、评论区与私信截图后确认:
  - 她并非一直在做截面多因子排序
  - 早期存在明确的 `128` 日日 K 截面多因子阶段
  - 后期已转向 `rule-based / trigger-based` 的多策略组合
  - 当前公开可见体系更接近 `12` 个子策略 + 市场状态切换 + 1 分钟级 T+0
- 额外确认:
  - 纯量价策略大约只有 `3` 个
  - 其他策略混合基本面 / 另类数据
  - “不是不做动量, 是不做截面排序”, 仍有策略保留动量内核, 只是 `trigger` 改变
- 新建 `experiments/target-strategy-evolution.md`:
  - 保留旧 benchmark 的全部关键信息
  - 按“早期日频基线 / 后期多策略体系”重新整理
  - 明确 `Rotation` 现在只应对标其一个子策略层面的能力
- 原 `experiments/rotation-benchmark.md` 废弃:
  - 原问题不是数据失真, 而是把不同阶段的信息混写成了一个静态 benchmark
  - 后续统一以 `target-strategy-evolution.md` 为准

## 2026-04-03

### [Rotation] Alpha158 各组 `top1` 组合验证失败, `kbar_shape` 继续保留主线地位
- 验证组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = core_plus_alpha158_top1`
  - `ALPHA158_ANALYSIS_GROUP_MODE = all`
  - 总特征数 = `21` (`core_12 + 9` 个 Alpha158 各组 `top1`)
- `Cell 7` 诊断结果:
  - `Target IC Mean = +0.0322`
  - `ICIR = +0.2987`
  - `L/S Sharpe = 1.36`
  - `Top-20` 日均双边换手约 `121.4%`
- Rust 回测结果:
  - `Gross Return = +10.40%`
  - `Total Return = -49.33%`
  - `Max Drawdown = 50.38%`
  - `Avg Trades/Day = 7.0`
  - 总成本 `298,619`
- 阶段结论:
  - “每组保留 1 个”不适合作为主训练入口的默认规则
  - 弱组 `top1` 会稀释 `kbar_shape` 这类强增量组, 同时放大高换手噪声
  - `core_plus_alpha158_top1` 暂时退出主线, 当前 Alpha158 主线仍是 `core_plus_alpha158(kbar_shape)`

### [Rotation] Cell 6 已接入 `core_12 + Alpha158各组top1` 训练入口
- `notebooks/cross_section_rotation.py` 新增 `FEATURE_MODE = "core_plus_alpha158_top1"`:
  - 训练特征 = 冻结 `core_12` + `Cell 3` 产出的 `alpha158_top1_factor_cols`
  - 不再手工拷贝各组 `top1` 因子名到训练面板
- 新模式设计为显式依赖 `Cell 3`:
  - 若 `alpha158_top1_factor_cols` 为空, `Cell 6` 会直接报错并提示先运行 `Cell 3`
  - `marimo` 现在会自动保证 `Alpha158 top1` 结果先于训练入口准备完成
- 训练元数据已补充:
  - `alpha158_analysis_group_mode`
  - `alpha158_top1_factors`

### [Rotation] Alpha158 分组 top1 + Cell 3 面板重构启动
- 新增 `utils/factor_analysis.py`，下沉 notebook 原先散落的公共分析逻辑:
  - `IC summary` 汇总表
  - 分组汇总表
  - `Alpha158` 每组 `top1` 因子提取
  - 通用 `Alpha Decay` 计算
- `notebooks/cross_section_rotation.py` 的 `Cell 3` 系列已开始按“thin notebook”方向重构:
  - `Cell 3` 现在统一产出:
    - `rotation_ic_summary / rotation_ic_results`
    - `alpha158_ic_summary / alpha158_ic_results`
    - `df_alpha158_group_summary`
    - `df_alpha158_top1`
    - `alpha158_top1_factor_cols`
  - 原 `Cell 3aa` 的 Alpha158 分组汇总已并回主分析面板, 不再单独散落
  - `Cell 3b` 不再写死依赖 Rotation `Top-15`, 现支持:
    - `rotation`
    - `alpha158_top1`
    - `custom_list`
- 新增配置口径:
  - `ALPHA158_ANALYSIS_GROUP_MODE`
  - `ALPHA_DECAY_SOURCE`
  - `ALPHA_DECAY_CUSTOM_FACTORS`
- 训练与分析已开始解耦:
  - `ALPHA158_GROUP_MODE` 继续服务训练特征选择
  - `ALPHA158_ANALYSIS_GROUP_MODE` 可单独控制分析侧要覆盖哪些分组
- `Cell 3c / 3d` 已降级:
  - `3c` 变为可选诊断工具, 默认不跑
  - `3d` 退出主流程, 默认直接回落到冻结 `core_12`
- 当前冻结 `core_12` 已显式写入 notebook 配置, 不再依赖每次运行时现场重筛

### [Rotation] Alpha158 全量 Polars 复刻落地
- 新增 `utils/alpha158_factors.py`:
  - 按 `Qlib Alpha158` 默认配置复刻全部 `158` 个因子
  - 默认包含:
    - `kbar` 9 因子
    - `OPEN/HIGH/LOW/VWAP` 的 `window=0` 价格因子
    - `29` 类 rolling 算子 × `5/10/20/30/60` 窗口
- 当前实现口径:
  - 基于本地 `open_adj/high_adj/low_adj/close_adj/vwap_adj/volume`
  - 价格类统一使用复权价
  - 成交量类沿用本地 `volume` 序列
- 已完成冒烟验证:
  - 因子总数确认 = `158`
  - 可与 `Rotation` 原有 `46` 因子合并后一起走 `cross_section_normalize`
- 后续性能优化已完成:
  - `RANK / BETA / RSQR / RESI / IMAX / IMIN / IMXD` 已全部从 `rolling_map` 改为 Polars 原生表达式
  - 现改用:
    - `rolling_rank`
    - `rolling_cov / rolling_var`
    - `shift + max_horizontal`
- 数值校验:
  - 新旧实现对比后, 上述重因子最大误差仅为浮点噪声级别
  - 不再依赖 Python `numpy` 回调逐窗口执行

### [Rotation] Notebook 已接入 Alpha158 特征模式
- `notebooks/cross_section_rotation.py` 现已在 Cell 2 同时计算:
  - 原 `Rotation` 因子
  - `Alpha158` 因子
- Cell 6 新增 `FEATURE_MODE` 选项:
  - `alpha158`
  - `core_plus_alpha158`
  - `all_plus_alpha158`
- 当前设计选择:
  - 因子分析 Cell 3/3d 仍以原 `Rotation` 因子为主
  - `Alpha158` 先作为训练特征集接入, 先验证是否提升最终回测

### [Rotation] Feature Mode 懒计算落地
- `FEATURE_MODE` 已前移到 Cell 1, 数据构建阶段即可感知当前实验模式
- Cell 2 现按 `FEATURE_MODE` 懒计算因子:
  - `alpha158` 模式: 跳过全部 `Rotation` 因子计算
  - `all / pruned / core` 模式: 跳过 `Alpha158` 因子计算
  - `core_plus_alpha158 / all_plus_alpha158` 模式: 同时计算两套因子
- Cell 2 最终仅 `collect` 当前模式实际需要的特征列, 不再无差别落盘全部因子
- `Cell 3/3a/3b/3c/3d` 在 `alpha158` 模式下会自动跳过 `Rotation` 因子分析, 避免无意义计算

### [Rotation] Alpha158 `kbar_shape` 首轮对照出现正增益
- 首轮实验组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = core_plus_alpha158`
  - `Alpha158 分组 = kbar_shape`
  - 总特征数 = `21` (`core_12 + 9` 个 `kbar_shape` 因子)
  - `NORMALIZE_MODE = zscore`
  - `EXPORT_EMA_ALPHA = 0.30`
- `Cell 7` 信号质量结果:
  - `Target IC Mean = +0.0441`
  - `ICIR = +0.3827`
  - `t-stat = +10.89`
  - 经济分层 `L/S Sharpe = 1.47`
  - `Top-20` 日均双边换手约 `106.6%`
- Rust 组合回测结果:
  - `Gross Return = +59.79%`
  - `Total Return = +18.61%`
  - `Max Drawdown = 14.45%`
  - `Avg Trades/Day = 3.0`
  - 总成本 `205,925`
- 相比当前冻结研究基线 (`core_12 + fwd_ret_1d + EMA=0.30`), 本轮 `gross / net` 均继续抬升, 且回撤进一步收敛
- 阶段结论:
  - `Alpha158` 不应按“全部 158 因子一次性灌入”推进
  - `kbar_shape` 这条小而强的增量分组已值得进入主线候选
  - 下一步优先隔离验证:
    - `alpha158(kbar_shape)` 单跑
    - `core_12` vs `core_12 + kbar_shape` 的组合层成本兑现差异

### [Rotation] Alpha158 `kbar_shape` 单跑验证完成
- 单跑实验组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = alpha158`
  - `Alpha158 分组 = kbar_shape`
  - 总特征数 = `9`
  - `NORMALIZE_MODE = zscore`
  - `EXPORT_EMA_ALPHA = 0.30`
- `Cell 7` 结果:
  - `Target IC Mean = +0.0438`
  - `ICIR = +0.4250`
  - `t-stat = +12.10`
  - `L/S Sharpe = 1.45`
  - `Top-20` 日均双边换手约 `124.0%`
- Rust 组合回测结果:
  - `Gross Return = +7.65%`
  - `Total Return = -21.70%`
  - `Max Drawdown = 28.68%`
  - `Avg Trades/Day = 2.8`
  - 总成本 `146,736`
- 结论更新:
  - `kbar_shape` **本身不是可直接单跑的强组合 alpha**
  - 但它与 `core_12` 组合时能显著抬升 `gross / net` 并改善回撤
  - 因此当前更合理的判断是:
    - `kbar_shape` 主要是高价值的**交互增强器**
    - 而不是应替代 `core_12` 的独立主特征集
- 对研究优先级的影响:
  - 暂不再推进 `alpha158(kbar_shape)` 单跑方向
  - 下一步主线转为围绕 `core_12 + kbar_shape` 做组合层兑现优化

## 2026-04-02

### [Rotation] 训练目标复盘 + LABEL 过滤 bug 修复
- 复盘 `notebooks/cross_section_rotation.py` 后确认:
  - 当前 `LABEL = fwd_ret_1d` + `LGBMRegressor` **没有根本性错误**
  - 但训练目标是“收益幅度回归”, 实际使用是“当日截面 Top-N 排名”, 存在目标错配
- 当前共识更新:
  - `fwd_ret_1d` 继续保留为真实基线
  - 不回到 `fwd_ret_1d_excess` 主线 (此前已验证失败)
  - 下一步优先做“排序化标签”实验, 再决定是否值得切到 `LGBMRanker`
- 修复一处明显 bug:
  - Cell 6 训练样本过滤原先写死为 `fwd_ret_1d`
  - 现改为跟随 `LABEL` 动态过滤, 避免切换标签时样本集与训练目标不一致

### [Rotation] Artifact Traceability 落地
- `utils/signal_export.py` 为 `Rotation` 导出新增 artifact 追踪能力:
  - 训练阶段固定 `train_run_id`
  - 保存 `artifacts/rotation/<train_run_id>/train.meta.json`
  - 保存 `artifacts/rotation/<train_run_id>/raw_scores.parquet`
  - 保存 `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/signal.parquet`
  - 保存对应 `signal.meta.json`
  - 在 `artifacts/rotation/<train_run_id>/signals.jsonl` 记录该 train run 派生过哪些 signal
- sidecar 元数据已记录:
  - `LABEL`
  - `feature_mode`
  - `feature_hash`
  - 完整因子列表
  - `LightGBM` 参数
  - `EXPORT_EMA_ALPHA`
  - `git_commit`

### [Rotation] Rust 回测报告追踪增强
- `bt-rotation` 现在会自动读取信号文件旁的 `signal.meta.json`
- 每次回测输出固定文件:
  - `report.txt`
  - `report.json`
- 输出目录改为 signal 目录下:
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/backtests/<backtest_timestamp_ms>/`
- 回测记录统一写入:
  - `artifacts/rotation/<train_run_id>/backtest.jsonl`
- 报告中区分:
  - `Input Signal` (Rust 实际读取路径)
  - `Canonical Signal` (artifact 真正归档路径)

### [Rotation] Signal Source 改为 artifacts 唯一真源
- 默认不再导出 `data/signals/rotation_scores.parquet`
- `artifacts/rotation/.../signal.parquet` 成为唯一真实 signal 文件
- `signals.jsonl` 改为下沉到每个 `train_run_id` 目录
- `backtest-engine/run_rotation.bat` 现在是轻量包装器:
  - 不带参数时, 进入交互式选择
  - 传入 `signal.parquet / signal.meta.json / signal目录` 时直接回测
- 新增 `scripts/rotation_backtest.py`:
  - 交互式选择 `train run -> signal`
  - 自动创建 `backtests/<backtest_timestamp_ms>/`
  - 调用 `bt-rotation` 并把结果写回对应 signal 目录
- 这样可以安全支持:
  - 历史 signal 回测
  - 批量导出多个 signal
  - 后续参数扫描

### [Rotation] 回测参数覆盖入口落地
- `scripts/rotation_backtest.py` 新增 CLI 覆盖参数:
  - `--hold-buffer`
  - `--min-score`
  - `--max-hold-days`
  - `--top-n`
- 每次回测都会在对应 `backtests/<backtest_timestamp_ms>/` 下生成:
  - `effective.config.toml`
- 设计约定:
  - Git 追踪的 `backtest-engine/crates/rotation/config.toml` 继续作为稳定基线
  - 临时实验参数不再要求手改基线 config
  - 实际生效参数通过 `effective.config.toml` 与 `backtest.jsonl / report.json` 一起追踪

### [Rotation] 导出 EMA 扫描完成
- 基于 `core_12 + fwd_ret_1d` 对 `EXPORT_EMA_ALPHA` 做两轮扫描:
  - 粗扫: `1.0 / 0.4 / 0.3 / 0.2 / 0.15 / 0.1 / 0.05`
  - 细扫: `0.25 / 0.28 / 0.30 / 0.32 / 0.35`
- 结论:
  - `EXPORT_EMA_ALPHA = 0.30` 给出当前最佳净收益 (`Gross +51.19% / Net +16.16%`)
  - `0.28` 为峰值附近次优平衡点
  - `1.0 / 0.4` 平滑不足, 交易成本显著失控
  - `0.1 / 0.05` 平滑过强, 会压缩真实 alpha
- 已将 `notebooks/cross_section_rotation.py` 的导出默认值更新为新的临时研究基线:
  - `EXPORT_EMA_ALPHA = 0.30`

### [Rotation] LABEL 跟随 bug 修复补全
- `notebooks/cross_section_rotation.py` 的 Cell 6 训练样本过滤已改为使用 `LABEL`
- 进一步补齐 Cell 7 信号质量分析:
  - join 的标签列不再写死 `fwd_ret_1d`
  - 过滤条件不再写死 `fwd_ret_1d`
  - IC / Quintile / Turnover 分析读取的收益列改为跟随 `LABEL`
- 这样后续切换到 `fwd_ret_2d / fwd_ret_5d / excess / 排序化标签` 时, 训练层与分析层不会再出现静默口径错位

### [Rotation] 组合参数阶段性收敛
- 在 `core_12 + fwd_ret_1d + EXPORT_EMA_ALPHA=0.30` 下完成两轮组合参数扫描:
  - `hold_buffer = 35 / 50 / 70 / 90 / 120`
  - `max_hold_days = 5 / 7 / 10 / 15`
- 结论:
  - `hold_buffer = 50` 仍是当前最优退出阈值
  - `max_hold_days = 15` 给出当前最高 `net return` (`+17.00%`)
  - 但为了保持博主早期公开日频基线“平均持仓约 2.8 天”的节奏特征, 当前**不**将 `15` 升格为早期日频对标锚点
  - 当前冻结的早期日频对标锚点为:
    - `Feature Set = core_12`
    - `LABEL = fwd_ret_1d`
    - `EXPORT_EMA_ALPHA = 0.30`
    - `hold_buffer = 50`
    - `max_hold_days = 10`
- 下一步转向:
  - 先验证“排序化标签”是否能更贴近最终 `Top-N` 排名目标
  - 暂缓继续深挖 `top_n / min_score`，待训练目标方向确认后再回头微调

### [Rotation] 排序化标签最小实验入口落地
- `notebooks/cross_section_rotation.py` 现已新增一组最小排序化标签:
  - `fwd_ret_1d_rank_pct`
  - `fwd_ret_2d_rank_pct`
  - `fwd_ret_3d_rank_pct`
  - `fwd_ret_5d_rank_pct`
- 定义方式:
  - 在每日截面内, 将未来收益映射为 `[0, 1]` 分位数标签
  - 保留现有 `LGBMRegressor` 训练链路不变, 先验证“标签语义更贴近排序”本身是否有效
- 设计目的:
  - 避免一开始就同时改 `LABEL + 模型类型 + Ranker`，把实验变量拆开
  - 若 `rank_pct` 标签已能显著改善 `IC / Quintile / Rust Top-N`，再决定是否值得进入 `LGBMRanker`

### [Rotation] Cell 7 双口径评估落地
- `notebooks/cross_section_rotation.py` 的 Cell 7 现已拆成两套口径:
  - `7a Target IC`: 跟随 `LABEL`，诊断模型是否学到训练目标
  - `7b / 7d Economic Evaluation`: 固定使用 `fwd_ret_1d`，统一比较不同训练目标的真实经济效果
- 这样切到 `fwd_ret_1d_rank_pct / fwd_ret_2d / excess` 时:
  - 不会再把标签值本身误读成“真实收益”
  - Quintile / L-S / 分年统计可以横向可比
- 首轮 `fwd_ret_1d_rank_pct` 手动复核结论:
  - `Target IC` 仍然较高, 这本身不构成优势证明
  - 因为 `rank_pct` 与原始收益在单日截面上是单调映射, `Spearman IC` 天然可能接近
  - 更关键的 `fwd_ret_1d` 经济分层与回测暂未显示优于当前 `fwd_ret_1d` 基线

### [Rotation] 截面归一化模式开关落地
- `utils/rotation_factors.py` 的 `cross_section_normalize()` 现已支持:
  - `zscore`
  - `rank_pct`
  - `rank_gauss`
- `notebooks/cross_section_rotation.py` 新增 `NORMALIZE_MODE` 配置项:
  - 目前可直接切换因子输入的截面归一化方式, 无需改训练主链路
- 设计目的:
  - 快速验证“特征截面 z-score”是否值得替换为更偏排序化的输入表达
  - 保持 `LABEL / LightGBM / Rust 回测` 链路不变, 隔离单一实验变量
- 训练元数据现已额外记录:
  - `normalize_mode`
- `utils/signal_export.py` 现已同步把 `normalize_mode` 写入:
  - `train.meta.json`
  - `signal.meta.json`
  - `signals.jsonl`
- 首轮实验结论:
  - `NORMALIZE_MODE = rank_pct` 明显失败:
    - `Gross +8.66% / Net -12.76% / Avg Trades 2.0`
  - `NORMALIZE_MODE = rank_gauss` 同样失败, 且未优于 `rank_pct`:
    - `Gross +9.13% / Net -14.26% / Avg Trades 2.1`
  - 两者都显著弱于当前 `zscore` 基线:
    - `Gross +51.19% / Net +16.16% / Avg Trades 2.6`
- 当前结论:
  - 对 `core_12 + LightGBM + rotation` 主线, 因子输入不能简单替换为纯 rank 系截面归一化
  - `rank_pct / rank_gauss` 让交易频率略降, 但 gross alpha 基本被打穿
  - 因此“特征归一化改成 rank 系”这条线暂时收口, 主线继续保留 `zscore`

### [Rotation] VWAP 数据链路补齐
- `utils/duckdb_utils.py` 的日线 / 60 分钟加载现已新增:
  - `vwap_raw`
  - `vwap_adj`
- 当前约定:
  - `stock_daily.volume` 单位确认为“手”，不是“股”
  - `vwap_raw = amount / (volume * 100)`
  - `vwap_adj = vwap_raw * adj_ratio`
- `notebooks/cross_section_rotation.py` 的 `df_all` 现已保留 `vwap_adj / vwap_raw`
- `notebooks/cross_section_rotation.py` 新增 `Cell 2b`:
  - 直接比较 `amount / volume` 与 `amount / (volume * 100)` 哪个更贴近 `close_raw`
  - 若 `vwap_raw` 与 `close_raw` 数量级失配会直接报错
  - 固定写明:
    - `vwap_raw = amount / (volume * 100)`
    - `turnover_rate(%) = volume * 100 / circulating_capital * 100`
- 同步修正:
  - `utils/rotation_factors.py` 的 `turnover_rate`
  - `utils/b1_factors_opt.py` 的 `turnover_rate`
- 目的:
  - 为后续接入 `Qlib Alpha158` 准备价格字段
  - 避免未来再回头改一次底层数据链路

### [Backtest Core] Artifact I/O 安全下沉
- 将与策略无关的 artifact 追踪 I/O 从 `bt-rotation` 抽到 `bt-core`:
  - `SignalArtifactMeta`
  - `load_signal_meta()`
  - `build_report_stem()`
  - `write_report_bundle()`
  - `resolve_registry_path()`
  - `append_jsonl_record()`
- `Rotation` 仅保留策略专属部分:
  - 配置序列化
  - 额外统计 (`limit_up_blocked`)
  - registry 记录字段选择
- 这次重构目标是为后续 `B1 / Renko / 更多策略` 复用同一套 artifact 追踪能力, **不强行统一各策略的个性化逻辑**

## 2026-04-01

### [Rotation] 下一阶段执行清单落地
- 新增 `experiments/rotation-next-phase.md`
- 将下一阶段目标明确为:
  - 导出侧独立 `EXPORT_EMA_ALPHA`
  - 因子治理与核心因子收敛
  - `LightGBM` 之外的模型基线对照
  - 固定研究基线后的组合参数收敛
- 明确修正共识: `Rotation` 当前标的池已经是 **80~500 亿**, 不再作为下一阶段主任务

### [Rotation] 导出侧独立 EMA 落地
- `notebooks/cross_section_rotation.py` 的训练 Cell 现在只输出 `df_scores_raw`
- 新增独立导出 Cell, 本地控制 `EXPORT_EMA_ALPHA`
- 修改导出平滑参数时, 只需重跑导出 Cell, 无需重新训练 `LightGBM`
- `Rotation` 与 `Renko` 现已统一为“raw score → export EMA”的导出模式

### [Rotation] 因子分组基础设施落地
- `utils/rotation_factors.py` 新增:
  - `FACTOR_GROUPS`
  - `FACTOR_GROUP_LABELS`
  - `FACTOR_TO_GROUP`
- 分组覆盖当前全部 `Rotation` 因子, 并在模块加载时校验完整性
- `notebooks/cross_section_rotation.py` 新增分组概览 Cell, 可直接查看每组因子数量、平均 `|ICIR|` 与组内最佳因子

### [Rotation] 核心因子筛查入口落地
- `notebooks/cross_section_rotation.py` 新增 `Cell 3d`
- 基于:
  - 因子分组
  - 单因子 `|ICIR|`
  - 全局相关性剪枝结果 (`factors_keep`)
- 自动给出一版建议 `core feature set`
- `FEATURE_MODE` 现支持 `"core"`, 且参数位于 `Cell 6` 本地, 可直接训练核心因子版本与 `"all"` / `"pruned"` 对照

### [Renko] 研究链路时钟统一重构

#### 核心决策
- `notebooks/renko_ml_explore.py` 统一为: **T 日收盘确认信号 / 计算特征 → T+1 日开盘买入**
- 本次只改研究 notebook 的时间线, **暂不修改 Rust 导出 / 回测格式**

#### 主要修改
- Renko 专属 `rk_*` 因子不再混用 `T-1` 数据:
  - 删除 `_c1`, `_o1`, `_v1` 临时 shift 列
  - `rk_bias_wl`, `rk_wl_yl_spread`, `rk_shape`, `rk_rw_dif_pct`, `rk_vol_shrink` 全部改为 **T 日收盘可得**
- 标签改为以 `T+1 open` 为基准:
  - 新增 `buy_open_t1 = open_adj.shift(-1)`
  - `fwd_mfe_5d = max(high[T+1:T+5]) / buy_open_t1 - 1`
  - `fwd_ret_1d = close[T+1] / buy_open_t1 - 1`
- 保留 `renko_signal[T]` 作为信号确认时点, 不再与 `T-1` 特征混搭

#### 修复的问题
- 旧版 notebook 内部同时混用了:
  - `rotation` 通用因子: T 日
  - 部分 `rk_*` 因子: T-1
  - 标签分母: `close[T]`
  - 单笔交易分析: `open[T+1]`
- 现已统一为单一时间线, 便于后续重新评估 Renko ML 是否真实有效

#### 新增: Renko 可切换标签入口
- Cell 1 新增 `LABEL` 配置, 当前默认 `fwd_ret_open_2d`
- Cell 2 预先计算以下标签, 后续只改一行即可重训:
  - `fwd_ret_open_2d = open[T+2] / open[T+1] - 1`
  - `fwd_ret_close_2d = close[T+2] / open[T+1] - 1`
  - `fwd_ret_close_3d = close[T+3] / open[T+1] - 1`
- Cell 3 / 4 / 5 全部改为自动引用 `LABEL`

#### 新增: Renko 分析实验面板
- Cell 5b 改为专门验证高换手短脉冲问题, 提供三组 notebook 内实验:
  - **EMA 平滑实验**: `ANALYSIS_EMA_ALPHAS = [1.0, 0.2, 0.1, 0.05]`
  - **Top-N 扩大实验**: `ANALYSIS_TOP_NS = [20, 50, 100]`
  - **高分阈值过滤实验**: `ANALYSIS_SCORE_QUANTILES = [0.99, 0.97, 0.95, 0.90]`
- 设计目标: 先在 notebook 内验证 `open_2d / close_3d` 的 alpha 是否能通过平滑、扩容或高分过滤保留下来, 再决定是否值得继续改回测引擎

## 2026-03-31

### [Rotation] 涨跌停幻觉修复 + 训练管线重构

#### 发现: 涨停幻觉 (幻觉 1)
- 之前 +400% 收益中, 大量 alpha 来自"买入涨停股" — 实操中根本买不进
- 统计: Top-20 中日均 4.8 只是涨停股 (88% 的交易日有过滤), 占候选的 ~24%
- 过滤涨停后, 旧模型 Gross 从 +586% 暴跌至 +5.7%, 证实 alpha 几乎全是幻觉

#### Rust 回测引擎: 涨跌停过滤 (bt-core 抽象)
- `bt-core/src/lib.rs`: 新增 `price_limit_pct()`, `is_limit_up()`, `is_limit_down()` 共享函数
  - 主板 (60/00) → ±10%, 创业板 (300/301) / 科创板 (688/689) → ±20%
  - 容差 0.1% (覆盖复权价四舍五入精度)
- `bt-rotation/main.rs`: 候选股过滤 — 先选 Top-N 再剔除涨停, 并统计被过滤数量
- `bt-rotation/systems.rs`: 跌停锁仓 — 持仓跌停时跳过卖出, 打印 [LOCKED] 日志

#### Python 训练管线: 排除涨停样本
- `utils/duckdb_utils.py`:
  - `load_daily_data_full()` 新增 `pre_close_adj` 输出列
  - 新增 `add_price_limit_cols()` 共享函数 (与 Rust 判定逻辑一致)
- `utils/__init__.py`: 导出 `add_price_limit_cols`
- `notebooks/cross_section_rotation.py`:
  - Cell 2: 调用 `add_price_limit_cols()` 打标记, 统计涨跌停样本数
  - Cell 6: 训练时 `valid = np.isfinite(y_tr) & ~is_limit_up_np[ts:te]`
  - 打分: 全量打分 (不过滤), Rust 兜底

#### 修复后真实 alpha (fwd_ret_1d, 0.1% 滑点)
| 指标 | 修复前 (含涨停幻觉) | 修复后 (排除涨停) |
|---|---|---|
| Gross Return | +586% (幻觉) | **+48%** |
| Total Return | +64% | **+10.5%** |
| Win Rate | 41.9% | **45.6%** |
| Max Drawdown | 27.5% | **21.3%** |
| 涨停过滤 (日均) | 4.8 只 | **2.0 只** |

### [Rotation] LABEL 参数化 + 超额收益标签实验

#### LABEL 参数化
- Cell 1 配置区新增 `LABEL` 参数, Cell 3 (IC) 和 Cell 6 (训练) 自动引用
- 可选: `fwd_ret_{1/2/3/5}d` 或 `fwd_ret_{1/2/3/5}d_excess`

#### 超额收益标签 (excess) 实验 — 失败
- 标签: `fwd_ret_1d - mean(fwd_ret_1d).over("date")`, 截面去均值
- 结果: 五分位单调性崩塌 (Q4 ≈ Q1), Top-20 选股退化为随机, Gross ≈ 0%
- 原因: 去均值后上半区信号区分度丢失

#### fwd_ret_2d 实验
- Gross +54% (高于 1d 的 +48%), 但换手反而增加 (3406 vs 2202 笔)
- 额外成本吞掉了额外 alpha, 净效果不如 fwd_ret_1d

### [Rotation] 因子时序对齐 (Route B: 滑点吸收法)
- `utils/rotation_factors.py` 重写: 去掉 `shift(1)`, 所有因子直接用 T 日 OHLCV
- `notebooks/b1_ml_explore.py`, `b1_ml_dedicated.py`, `renko_ml_explore.py`: 自行计算 `_c1` 等 shift 列
- Rust config.toml: slippage 调整为 0.3% (含 14:45~15:00 快照误差)

## 2026-03-25

### [B1] ML 排序替代手搓因子探索

#### 全市场模型 (`b1_ml_explore.py`)
- 新建 notebook, 56 特征 (42 rotation + 14 B1 专属), 标签 MFE-10
- 全市场训练 LightGBM, 对 B1 候选排序
- 信号质量: IC +0.137, L/S +3.95%, t-stat +32.38 — 极显著
- Rust 回测: 近期 +36.63% (手搓 +30.49%), 长周期 +78.36% (手搓 +81.05%)
- **近期跑赢手搓 +6pp, 长周期持平, 但回撤更大**

#### B1 专属模型 (`b1_ml_dedicated.py`)
- 仅用 B1 信号日样本 (14,242 条) 训练, 38 特征 (IC 筛选)
- 发现: B1 子集 IC 排序与全市场完全不同 — `amihud_illiq_20d` 全市场 #1 但 B1 无效, `vol_60d` 在 B1 中最强
- 信号质量: IC +0.009, t-stat +0.54 (不显著) — 每天仅 10~17 只 B1 候选, 截面太窄
- Rust 回测: 近期 +21.38%, 长周期 +43.83% — **全面跑输**
- **结论: B1 专属模型不可行, 最佳方案仍是全市场 ML 排序**

#### Bug 修复
- 修复 `b1_ml_explore.py` Quintile 标签方向 (Q1/Q5 含义反了)
- 修复 Top-N Overlap 计算 (对比数组索引改为对比股票代码)
- 添加 LightGBM 训练进度打印 (对齐 rotation notebook)

### 文档结构优化
- 新建 `experiments/` 目录, 每个实验独立 markdown
- `project-status.md` 按策略分节 (Rotation / B1 / 共享基础设施)
- 迁移旧 `experiments.md` 内容到 `experiments/rotation-benchmark.md` + `rotation-factors.md`
  - 注: `rotation-benchmark.md` 后续已被 `target-strategy-evolution.md` 取代

## 2026-03-24

### 因子实验: 128 天长周期因子 (失败, 已回退)
- 尝试从 42 因子扩展至 55 因子, 新增 128 天动量/波动率/回撤/均线偏离/价格位置等
- 结果: IC 下降, 五分位单调性破坏, Rust 回测净收益从 +82% 降至 +30%
- 尝试根据单因子 IC 剪枝 (55→50), 反而更差 — 树模型的非线性交互使单因子 IC 不适合做删减依据
- **结论**: 128 天方向无效, 回退至 42 因子基线 (40 通用 + 2 处置效应)

### EMA 平滑参数探索
- 发现 α=1.0 (无平滑) 导致 Rust 回测灾难性结果: 日均 14.1 笔交易, 成本 55 万 > 本金 50 万, 净收益 -45%
- α=0.1 (旧默认): IC +0.0227, L/S t-stat 1.78 (不显著)
- **α=0.2 (新最优)**: IC +0.0234, L/S t-stat **2.08 (首次统计显著)**, Sharpe 1.14, 五分位完美单调
- 信号平滑是必需的, hold_buffer 单独无法控制换手

### Notebook 架构优化: Cell 6/7 解耦
- **问题**: Cell 6 (训练导出) 和 Cell 7 (信号分析) 各自做一次 EMA, 导致双重平滑
- **修复**: Cell 6 新增 `df_scores_raw` 输出 (EMA 前的原始分数)
- Cell 7 改为依赖 `df_scores_raw`, 独立控制 EMA_ALPHA
- **效果**: 调整分析侧 α 只需重跑 Cell 7, 无需重新训练模型

### Renko 导出侧独立 EMA
- `notebooks/renko_ml_explore.py` 的 Cell 6 新增 `EXPORT_EMA_ALPHA`
- 导出 parquet 时改为基于 `df_scores_raw` 现场做 EMA, 不再依赖训练 Cell 内部平滑
- `EXPORT_EMA_ALPHA` 已下沉到 Cell 6 本地配置, 避免 marimo 修改 Cell 1 时触发上游重跑
- **效果**: 切换导出用 α 只需重跑 Cell 6, 无需重新训练 LightGBM
- **复用价值**: 这套“训练输出 raw score, 导出侧单独做平滑”的模式后续可迁移到 `cross_section_rotation.py`, 避免每次只改导出 EMA 也要重训模型

### Renko 回测结论: 暂停继续深入
- 基于 `fwd_ret_open_2d` 做了 Rust 组合回测, 发现结果对导出侧 `EXPORT_EMA_ALPHA` 高度敏感
- `EMA=1.0 / 0.1 / 0.2` 下净值表现都很差, 且 gross alpha 不稳定
- `EMA=0.05` 虽然能把 Gross Return 提升到正值, 但净收益依旧明显为负, 成本无法覆盖
- 结论: 当前 Renko ML 信号在 notebook 统计上有一定信息量, 但**组合层可兑现性不足**, 暂不作为优先探索方向

### Rust 回测引擎: 报告自动保存
- `bt_core` 新增 `format_results()` + `write_report()` 共享函数
- rotation 和 b1 两个 crate 均支持 `--output-dir` 参数, 自动保存带时间戳的回测报告到 `results/`
- 报告包含完整配置参数 + 回测结果, 便于跨实验对比

### 当前最优回测结果 (α=0.2, 42 因子)
| 指标 | 值 |
|---|---|
| 净收益 | +82.57% |
| 毛收益 | +164.83% |
| 最大回撤 | 27.35% |
| 胜率 | 42.1% |
| 总交易 | 3,617 笔 (803 天) |
| 日均交易 | 4.5 笔 |
| 总成本 | 411,298 (Gross PnL 的 50%) |

## 2026-03-23

### 架构重构: Python 打分 + Rust 回测分离
- **核心决策**: Python 模型只负责截面打分 (1d/1d), 回测/风控/持仓管理全部交给 Rust ECS 引擎
- 依据:
  - LightGBM 1d/1d 模型偏度已为正 (+0.28), 天然适合日频信号
  - Python 固定持仓 N 天 (3d/3d, 1d/3d) 效果均一般, 无法灵活止损止盈
  - Rust 回测框架已支持 B1 策略的 Parquet 导入, 可复用架构

#### Python 端代码变更
- **`utils/signal_export.py`**: 新增 `export_rotation_scores()` 函数
  - 输入: `df_scores` (date, code, score + OHLCV + market_cap)
  - 输出: Parquet 含 score, rank, is_top_n, pre_close_adj 等列
  - Rust 端可直接读取, 每日选 Top-N 候选, 自行决策买卖
- **`notebooks/cross_section_rotation.py`**:
  - Cell 6: 从"LightGBM Walk-Forward 回测"重构为"打分 → Parquet 导出"
    - 移除: 所有 Python 侧回测逻辑 (HOLD_BUFFER/COST/HOLD_DAYS/portfolio 模拟/净值曲线/年度拆解)
    - 保留: Walk-Forward 训练循环、特征重要性输出
    - 新增: 每日全 universe 打分 → join 价格 → export_rotation_scores
  - Cell 4/5: 清空 (线性排名回测 + 旧可视化, 已被 LightGBM + Rust 替代)

#### Rust 引擎重构: 单 crate → Cargo workspace
- **改造原因**: 原引擎为 B1 单策略设计, PriceBar/信号/退出逻辑全部硬编码 B1 语义, 无法复用于轮动策略
- **新结构**: `backtest-engine/` 改为 Cargo workspace, 三个 crate:
  - `bt-core`: 共享类型 — Portfolio, BacktestStats, CostModel, 工具函数
  - `bt-b1`: B1 超跌反转策略 (从旧代码迁移, 功能不变)
  - `bt-rotation`: 截面轮动策略 (新建)
- **轮动策略回测逻辑**:
  - 读取 `rotation_scores.parquet` (Python LightGBM 打分结果)
  - 每日系统: check_exit_conditions → fill_positions → update_stats
  - 退出条件: 排名跌出 hold_buffer / 固定止损 / 移动止损 / 最大持仓天数
  - 入场条件: Top-N 买入 (尾盘收盘价), 等权仓位
  - TOML 配置: top_n, hold_buffer, stop_loss, trailing_stop, costs
- **运行方式**: `cargo run -p bt-rotation --release` (从 backtest-engine/ 目录)

## 2026-03-22

### 因子扩展: 处置效应因子 (Disposition Effect)
- 新增 2 个行为金融因子至 `utils/rotation_factors.py`，因子库扩展为 7 类 42 个
  - `disp_bias_20`: 20日 EWM 估算持仓成本偏离度 (短期处置效应)
  - `disp_bias_60`: 60日 EWM 估算持仓成本偏离度 (中期处置效应)
- 底层算法: EHC = EWM(TypicalPrice × Volume) / EWM(Volume)，用 EWM 指数衰减近似换手率驱动的筹码替换
- 无需额外数据源，仅依赖现有 OHLCV + 换手率
- 完全嵌入现有 Polars lazy chain，零额外 collect 开销

#### A/B 对比结果
- **毛收益年化**: 41.0% → 50.3% (+9.3pp), Sharpe 1.32 → 1.52
- **净收益年化**: 1.2% → 8.2% (+7.0pp), Sharpe 0.19 → 0.41
- **最大回撤**: 51.1% → 44.2% (-6.9pp)
- **偏度**: -0.01 → +0.28 (从负偏转正偏)
- 2025 年超额(净)从 -9.6% 翻正至 +19.8%
- 因子本身未进 Top 15 特征重要性，通过 GBM 交互效应提升整体表达力
- 详见 `results/disposition_effect_ab_test.md`

## 2026-02-24

### 截面轮动策略 Phase 1-2 完整研究

#### Phase 1: 因子工程 + IC 分析
- 实现 `utils/rotation_factors.py`，共 40 个日线截面因子（6 大类）
- Universe: 流通市值 80-500 亿，非 ST，上市 > 60 天，2020-09 起
- IC 分析结论: 25 个因子 |ICIR| > 0.1，12 个 > 0.35
- Top IC 因子: vol_std_20d (-0.58), ret_max_5d (-0.53), turnover_rate (-0.52)
- 新增 A 股 T+1 专用因子效果显著: high_open_pct (ICIR -0.46), amihud_illiq_20d (+0.40)
- 线性排名 Top-20 回测失败 (-46% 年化)，证实简单排名无法利用弱 IC

#### Phase 2: LightGBM Walk-Forward
- 480 天训练窗口，每 20 天重训，200 棵树，depth=6
- **毛收益**: 43% 年化, Sharpe 1.34, 正偏度 — alpha 确认存在
- Hold buffer 优化路径:
  - 无缓冲: 换手 87.8%, 净 -9.9%
  - Top-50: 换手 79.8%, 净 -1.7%
  - Top-150: 换手 68.2%, 净 +2.8%, Sharpe +0.24
- 特征重要性 Top 5: vol_ratio, vol_compress, turnover_ma_ratio, turnover_accel, abnormal_vol
- 核心瓶颈: 日均毛收益 0.166% vs 日均成本 0.136%，净利润空间薄

#### 原策略更多信息 (2026-01 原帖截图)
- **并非单模型, 而是 12 个子策略组合** — 灰线是 12 条子策略净值, 蓝线是组合
- **平均持仓 2.8 天**, 非严格 T+1 → 日换手率 ~31% (6.2 笔/天 ÷ 20 只)
- 7475 笔交易 / 5 年, 平均仓位 ~50%, 最大同时持仓 20 只
- 年化 50.42%, 最大回撤 9.13%, 胜率 54.01%, 盈亏比 1.5:1
- Alpha 0.60, Beta 1.78 (R²=0.28), 日收益偏度 0.90
- **日内还有独立的 1 分钟 T+0 择时系统**, 实盘比回测多 10-20% 年化
- 对我们的启示:
  1. 多策略架构 > 单一模型 (分散化 + 降回撤)
  2. 持仓 2.8 天 → 真实成本仅 0.062%/日, 是我们假设的一半
  3. 50% 仓位是为日内 T+0 留空间, 非风控约束

#### 技术修复
- 修复 Polars panic: `is_in(st_blacklist)` 改为 `anti-join`
- 修复嵌套 `.over("code")` 问题: 拆分为多步 `.with_columns()` 物化中间列
- 修复 marimo 变量重定义: Cell 内逻辑包裹在函数中
- 修复 numpy datetime64 → Polars Date 类型转换
- 移除重复因子 `gap`（与 `overnight_ret` 公式完全相同）

## 2026-04-14

### B1 教科书双阶段目标首版落地

- 教科书案例清单已从 `utils/` 拆到 manifest 层:
  - `manifests/b1_textbook_cases.py`
  - `manifests/b1_expanded_textbook_cases.py`
- `utils/b1_feature_pool.py` 现在只消费 `B1_TEXTBOOK_CASES` manifest，不再自己维护案例常量
- `notebooks/b1_condition_mining.py` 已同步显示 `base / expanded / total` 案例规模，避免分析层与训练层脱节
- `notebooks/b1_seed_ml_baseline.py` 已同步显示并记录 `textbook base / expanded / total` 规模与 `textbook_cases_version`，训练元数据会随 artifact 一并落下
- `B1_TEXTBOOK_SCORE_FEATURE_COLS`、`B1_TEXTBOOK_RULE_COLS` 仍由 `utils/b1_feature_pool.py` 维护
- 基于教科书案例在研究底表中动态生成:
  - `textbook_similarity_score`
  - `textbook_rule_score`
  - `textbook_b1_score`
  - `textbook_b1_threshold`
  - `is_textbook_b1`
  - `is_textbook_case`
  - `textbook_case_name`
- `Stage 1` 不再只盯周/月动能, 而是综合趋势位置、K 线几何、缩量/触发节奏、关键 K 上下文做案例相似度
- 为了提升可解释性, `textbook_b1_score` 已拆成四个显式子分:
  - `textbook_trend_score`
  - `textbook_kbar_score`
  - `textbook_volume_score`
  - `textbook_trigger_score`
- 当前总分口径改为四段式聚合:
  - `trend 30% + kbar 25% + volume 20% + trigger 25%`
  - 其中 `trigger_score = 0.7 * 触发上下文相似度 + 0.3 * 规则骨架分`
- `notebooks/b1_seed_ml_baseline.py` 新增 `TARGET_MODE`
  - `single_stage_mfe`: 保留原单阶段 `fwd_mfe_10d`
  - `two_stage_textbook`: `LGBMClassifier(is_textbook_b1)` + `LGBMRegressor(fwd_mfe_risk_adj_10d)`
- 双阶段最终分数口径:
  - `structure_score = P(is_textbook_b1)`
  - `payoff_score = E[fwd_mfe_risk_adj_10d | structure qualified]`
  - `final_score = structure_score * payoff_score`
- 导出链路同步保留 `structure_score / payoff_score`, 方便回测和排查“结构对了但收益排序错了”或相反
- `notebooks/b1_condition_mining.py` 支持新标签阈值, 并在 Lab 摘要里额外展示 textbook 结构分数、正类占比、案例覆盖行数
- 当前验证状态:
  - `python -m py_compile notebooks/b1_seed_ml_baseline.py`
  - `python -m py_compile notebooks/b1_condition_mining.py`
  - `python -m py_compile utils/b1_feature_pool.py`
  - `uv run python -c "import utils.b1_feature_pool"` 通过
  - `uv run python -c "import importlib.util; ... b1_seed_ml_baseline.py"` 通过
  - `uv run python -c "import importlib.util; ... b1_condition_mining.py"` 通过
  - `uv run python -X utf8 -c "... build_b1_research_frame(...)"` 已确认 `textbook_*` 子分可正常落表
- 首轮实跑结果:
  - `Stage 1` 确实开始学到更接近教科书案例的量价结构, 人工抽样观察与 `trend / kbar` 子分方向一致
  - 但 `Step 4` 全截面评估仍显示 `daily_ic_mean = -0.0792`, `q4_minus_q0 = -0.0232`
  - 同时 `top-k / threshold` 尾部表现仍有一定改善, 说明当前更像“高分尾部筛选器”, 还不是稳定的全局排序器
  - `2025+` 窗口回测出现一定苗头, 但由于教科书案例全部来自 `2025`, 暂不能排除时间风格锚定
- 当前限制:
  - 当前 Windows 终端若未显式使用 `uv run python -X utf8`，旧日志里的 emoji 可能触发 `UnicodeEncodeError`
  - `textbook_b1_threshold` 目前按案例分数 `q20` 动态推导, 后续仍需结合正类占比和回测结果再校准

### B1 历史案例扩容挖掘 notebook

- 新增 `notebooks/b1_case_expansion_mining.py`
- 当前默认输入池改用 `seed_mid`
- 当前结构职责已收口为:
  - 分析层: `notebooks/b1_case_expansion_mining.py`
  - 清单层: `manifests/b1_textbook_cases.py` / `manifests/b1_expanded_textbook_cases.py`
  - 训练层: `utils/b1_feature_pool.py` / `notebooks/b1_seed_ml_baseline.py`
- 当前 notebook 目标:
  - 从 `2021-01-01 ~ 2025-12-31` 的 `seed_mid` 样本里
  - 先按 `fwd_mfe_10d / fwd_mae_10d / fwd_mfe_risk_adj_10d` 过滤出“结果强”样本
  - 再与 `B1_BASE_TEXTBOOK_CASES` 做两层相似度比较:
    - 结构向量相似度
    - 归一化价格曲线相似度
- 当前实现不依赖 `fastdtw`, 先用固定窗口归一化曲线 + 简化距离做第一版验证
- 当前保留的核心输出:
  - `best_match_case`
  - `Top-k Archetype 覆盖摘要`
  - `Archetype 两两相似度矩阵`
  - `扩容候选分组摘要 / 明细`
  - 最终 `EXPANDED_TEXTBOOK_CASES` manifest 产物
- 最终产物口径:
  - 只收 `top1`
  - 按 `total_similarity / fwd_mfe_risk_adj_10d / textbook_b1_score` 做保守筛选
  - 当前收紧为 `0.84 / 0.18 / 0.75`
  - 每个 archetype 当前上限 `8`
  - 最终 manifest 按 `(source_archetype, code)` 去重，只保留该股票最优日期
  - 默认生成 Python manifest 文本，必要时再显式写入 `manifests/b1_expanded_textbook_cases.py`
- 当前验证状态:
  - `python -m py_compile notebooks/b1_case_expansion_mining.py`
  - `uv run python -c "import importlib.util; ... b1_case_expansion_mining.py"` 通过
  - `uv run marimo check notebooks/b1_case_expansion_mining.py` 通过

### AMV 多子策略: 静态袖子 Rust 回测起点

- 背景:
  - LTR 直接 Top3 与简单状态分类器暂时都未能稳定超过规则基线
  - Oracle Rust 回测显示“选对袖子”的理论空间极大，但 oracle 本身存在严重未来函数
  - 当前阶段改为先把可复现的静态子策略单独接入 Rust，建立每个袖子的真实收益/回撤画像
- 新增/调整:
  - 新增 `scripts/amv_static_sleeve_signal_export.py`
  - 扩展 `scripts/amv_bull_pool_export_signals.py` 的 feature frame，保留 `ret_5d / ret_20d`
  - 导出 5 个静态袖子信号:
    - `ret_5d`
    - `ret_20d`
    - `kmid2`
    - `klen`
    - `manual_p2_k0p5_r0`
  - 每个袖子使用 `bt-amv-topn` 跑 `1td / 2td / 3td / 6td`
  - 其中 `6td` 是与 AMV Bull Pool 因子年度稳定性、既有 AMV TopN 基线可比的主口径；`1td / 2td / 3td` 仅作为持仓周期敏感性扫描
  - 均为 no-stop、T+1 open、Top3、AMV bull regime required
- 本轮数据注意:
  - ST 实时源访问失败，导出时使用过期本地 ST 缓存，共 239 只；结果可用于研究排序，但后续正式批量回测建议在网络恢复后刷新 ST 缓存复跑
- 输出:
  - 信号根目录: `artifacts/amv_static_sleeve_signals/`
  - 汇总表: `artifacts/amv_static_sleeve_signals/static_sleeve_backtest_summary.csv`
  - 年度表: `artifacts/amv_static_sleeve_signals/static_sleeve_annual_returns.csv`
  - Canvas: `amv-static-sleeve-backtest.canvas.tsx`
- `6td` 主口径 Rust 净收益结论:
  - `manual_p2_k0p5_r0 6td`: `+168.01%`, gross `+225.39%`, max DD `14.97%`, trades `273`, win rate `51.65%`
  - `ret_5d 6td`: `+71.48%`, gross `+102.12%`, max DD `70.29%`, trades `269`, win rate `46.10%`
  - `ret_20d 6td`: `+6.46%`, gross `+28.36%`, max DD `84.14%`, trades `268`, win rate `44.40%`
  - `kmid2 6td`: `-48.93%`, gross `-30.01%`, max DD `73.40%`, trades `273`, win rate `45.05%`
  - `klen 6td`: `-72.49%`, gross `-62.39%`, max DD `76.95%`, trades `154`, win rate `36.36%`
- `6td` 年度稳定性:
  - `manual_p2_k0p5_r0 6td`: 2021 `+9.7%`, 2022 `+38.4%`, 2023 `+14.3%`, 2024 `+48.9%`, 2025 `+12.4%`, 2026 YTD `-8.7%`
  - `ret_5d 6td`: 2021 `+3.2%`, 2022 `+25.1%`, 2023 `-47.2%`, 2024 `-1.1%`, 2025 `+88.9%`, 2026 YTD `+37.5%`
  - `ret_20d 6td`: 2021 `-14.3%`, 2022 `-27.6%`, 2023 `-14.7%`, 2024 `+30.4%`, 2025 `+13.8%`, 2026 YTD `+25.8%`
- 短持仓扫描结论:
  - `ret_20d 2td`: `+59.01%`
  - `manual_p2_k0p5_r0 3td`: `+38.11%`
  - 这些结果说明持仓周期敏感，但不应替代 6td 主基线
- 当前判断:
  - 和 6td 基线对齐后，第一批主候选应改为 `manual_p2_k0p5_r0 6td`
  - `ret_5d 6td` 可以作为高波动候选，但最大回撤约 `70%`，不能直接视为成熟主线
  - `ret_20d 6td` 净收益很弱且回撤极大；它在 `2td` 强，说明它更像短周期动量袖子
  - 下一步应围绕 `manual_p2_k0p5_r0 6td`、`ret_5d 6td`、`ret_20d 2td` 做状态画像，分清“主策略”、“高波动增强”和“短周期动量”的适用环境

### AMV Horizon-aware sleeve oracle

- 背景:
  - 旧 oracle 主要是 `1td / 2td / 3td`，不能直接回答 `6td` 主基线下应该怎么做策略/权重切换
  - `manual_p2_k0p5_r0` 和 `manual_p3_k0p5_r0` 本身来自 6d 因子与权重网格实验，因此新的 oracle 必须把 “因子/权重 + 持仓周期” 一起作为 sleeve 定义
- 新增:
  - `scripts/amv_horizon_aware_oracle_lab.py`
  - Canvas: `amv-horizon-aware-oracle.canvas.tsx`
- 默认候选:
  - `manual_p2_k0p5_r0_6td`
  - `manual_p3_k0p5_r0_6td`
  - `ret_5d_6td`
  - `ret_20d_2td`
  - `ret_20d_6td`
  - `klen_6td`
  - `kmid2_6td`
  - `kbar_momentum_6td`
- 口径:
  - 使用 `next_open_to_close` 执行标签，即 `T+1 open -> T+N close`
  - `--exclude-limit-up-entry`
  - 每个候选每天 Top3
  - 只保留全部候选都有结果的 common dates，共 `807` 天
  - 同时输出两个 oracle:
    - `oracle_raw_return_with_cash`: 按候选持有期总收益选，若全部 <= 0 则选 cash
    - `oracle_dailyized_with_cash`: 按候选日化收益选，若全部 <= 0 则选 cash
- 产物:
  - `artifacts/amv_horizon_aware_oracle/20260517_130248/summary.json`
  - `daily_candidate_sleeves.csv`
  - `selected_candidate_signals.csv`
  - `daily_oracle_choices.csv`
  - `candidate_summary.csv`
  - `candidate_year_summary.csv`
  - `oracle_choice_summary.csv`
- 静态候选标签侧均值:
  - `ret_5d_6td`: mean `+0.96%`, dailyized `+0.07%`
  - `kmid2_6td`: mean `+0.74%`, dailyized `+0.08%`
  - `manual_p2_k0p5_r0_6td`: mean `+0.70%`, dailyized `+0.10%`, positive ratio `54.0%`
  - `ret_20d_2td`: mean `+0.70%`, dailyized `+0.27%`
  - `manual_p3_k0p5_r0_6td`: mean `+0.69%`, dailyized `+0.10%`, positive ratio `54.5%`
  - `klen_6td`: mean `-0.30%`
- Oracle 选择分布:
  - 按总收益选:
    - `ret_5d_6td`: `163` 天
    - `ret_20d_6td`: `138` 天
    - `kmid2_6td`: `125` 天
    - `ret_20d_2td`: `120` 天
    - `manual_p2_k0p5_r0_6td`: `68` 天
    - `cash`: `57` 天
  - 按日化收益选:
    - `ret_20d_2td`: `294` 天
    - `ret_5d_6td`: `106` 天
    - `kmid2_6td`: `104` 天
    - `ret_20d_6td`: `75` 天
    - `manual_p2_k0p5_r0_6td`: `61` 天
    - `cash`: `57` 天
- 当前判断:
  - `6td` 不是天然最优，而是当前主基线；模型目标必须显式区分“持有期总收益”与“资金效率/日化收益”
  - 如果目标是单笔总收益，oracle 更偏 `ret_5d_6td / ret_20d_6td`
  - 如果目标是资金效率，oracle 明显偏 `ret_20d_2td`
  - `manual_p2_k0p5_r0_6td` 更像稳定主策略底座，不是 hindsight oracle 最常选择的进攻袖子
  - 下一步训练切换模型前，应先决定目标函数: 固定 6td 收益、混合 horizon 总收益、还是日化/资金效率收益

### AMV Horizon-aware oracle explainability

- 背景:
  - 用户确认目标不是回到固定人工规则，而是让模型学习策略/权重/持仓周期切换
  - 在训练切换模型前，需要先判断 horizon-aware oracle 的选择是否能被交易前状态解释
- 新增:
  - `scripts/amv_horizon_oracle_explainability.py`
  - Canvas: `amv-horizon-oracle-explainability.canvas.tsx`
- 产物:
  - `artifacts/amv_horizon_oracle_explainability/20260517_131130/summary.json`
  - `oracle_choices_with_state.csv`
  - `choice_state_summary.csv`
  - `phase_choice_distribution.csv`
  - `feature_separation.csv`
  - `candidate_feature_diffs.csv`
  - `candidate_top_feature_diffs.csv`
- 方法:
  - 读取 horizon-aware oracle 的 `daily_oracle_choices.csv`
  - 使用 `build_ltr_dataset(..., horizon=6, label_mode=next_open_to_close)` 生成交易前状态特征
  - 按 oracle 选择的 candidate 汇总 AMV 状态、宽池赚钱效应、topN 动量、成交额变化等特征
  - 计算各候选组之间的状态均值差异，单位为全样本标准差
- 核心观察:
  - 当前状态特征对 oracle class 的分离度偏弱，最强组间差异约 `0.59 std`
  - `oracle_dailyized_with_cash` 最强分离特征:
    - `pool_candidate_count_scaled`: `0.59 std`
    - `bull_day_scaled`: `0.58 std`
    - `amv_bull_trigger_ret_scaled`: `0.54 std`
    - `trail_pool_ret_5d`: `0.51 std`
  - `manual_p2_k0p5_r0_6td` 被选时更偏 AMV bull 早期:
    - raw oracle: early `36.8%`, middle `33.8%`, late `29.4%`
    - dailyized oracle: early `36.1%`, middle `32.8%`, late `31.1%`
  - `ret_20d_6td` 更偏宽池上涨比例与 topN 20d 动量较强的状态:
    - dailyized oracle 下 `pool_up_ratio_5d = 60.6%`
    - `pool_topn_ret_20d = 45.5%`
  - `cash` 并不是简单对应“弱市场”，反而在 AMV bull age、AMV trigger、近 5 日宽池收益偏高时出现；这更像过热/后段而非低迷
- 当前判断:
  - 直接训练多分类模型去预测完整 oracle candidate 可能会很噪，当前状态特征不足以清晰分开所有袖子
  - 更合理的第一版模型目标是:
    - 默认使用 `manual_p2_k0p5_r0_6td`
    - 学习何时切到进攻袖子 (`ret_5d_6td / ret_20d_6td / ret_20d_2td`)
    - 学习何时切到 `cash` 或降仓
  - 下一步应做二分类/三分类 gating，而不是直接做 8 类 sleeve classifier:
    - `base_ok`: 是否继续用 `manual_p2_6td`
    - `attack_ok`: 是否允许动量进攻袖子
    - `cash_ok`: 是否需要空仓/降仓

### AMV constrained oracle: base + attack/cash 上限

- 背景:
  - 在使用模型前，先验证“默认主策略 + 少数例外切换”的理论空间
  - 这是比完整 oracle 更接近落地的受约束上限，但仍是使用未来收益的标签侧实验，不是可交易规则
- 新增:
  - `scripts/amv_constrained_oracle_lab.py`
  - Canvas: `amv-constrained-oracle.canvas.tsx`
- 产物:
  - `artifacts/amv_constrained_oracle/20260517_131848/summary.json`
  - `daily_constrained_choices.csv`
  - `strategy_summary.csv`
  - `choice_summary.csv`
  - `year_summary.csv`
- 设定:
  - base: `manual_p2_k0p5_r0_6td`
  - attack candidates:
    - `ret_5d_6td`
    - `ret_20d_6td`
    - `ret_20d_2td`
    - `kmid2_6td`
  - margin: `0% / 1% / 2% / 3%`
  - 两个目标:
    - `top_ret`: 按持有期总收益比较
    - `top_ret_dailyized`: 按日化/资金效率比较
  - 两个规则:
    - `base + attack`: attack 超过 base + margin 才切
    - `base + attack + cash`: 若 base < 0 且 best_attack <= 0 则 cash，否则按 attack 规则
- 关键结果:
  - `top_ret + attack + cash, margin=3%`:
    - mean chosen ret `+9.14%`
    - lift vs base `+8.44pp`
    - attack `518` 天, base `206` 天, cash `83` 天
  - `top_ret_dailyized + attack + cash, margin=3%`:
    - mean chosen ret `+4.57%`
    - lift vs base `+3.87pp`
    - attack `197` 天, base `527` 天, cash `83` 天
  - `top_ret + attack + cash, margin=0%`:
    - mean chosen ret `+9.31%`
    - attack `602` 天, base `122` 天, cash `83` 天
- 当前判断:
  - 受约束 oracle 仍有很大上限，说明“主策略 + 例外切换”方向值得继续
  - 但如果用持有期总收益作为目标，即使 margin 提到 `3%`，仍需要 `518/807` 天切 attack；这不是“少数例外”，而是高频状态切换
  - 如果用日化/资金效率目标，`3% margin` 下 attack 降到 `197/807` 天，形态更接近可学习的例外切换
  - cash 上限稳定为 `83` 天，且带来额外提升，后续应单独建 `cash_ok`/降仓标签
  - 当前理论分析锚点是 Canvas `amv-constrained-oracle.canvas.tsx`（AMV 受约束 Oracle）；后续讨论切换模型时，应优先引用这个结论，避免回到完整 hindsight oracle 或重复 8 类 sleeve selector 分析
  - 目前不再把“完整 oracle 哪天选哪个袖子”当作直接训练目标；它只作为上限诊断，真正建模入口是 `manual_p2_k0p5_r0_6td` 底座上的 `cash_ok` / `attack_ok`
  - 下一步更合理的建模目标:
    - 第一优先: `cash_ok` 二分类，识别是否应该避开主策略亏损日
    - 第二优先: `attack_ok` 二分类，先用 dailyized `3% margin` 作为较保守标签
    - 暂不直接训练完整多分类 sleeve selector

### AMV attack_ok first binary lab

- 背景:
  - 用户要求直接推进 `attack_ok`
  - 理论锚点继续使用 `amv-constrained-oracle.canvas.tsx`（AMV 受约束 Oracle），不回到完整 hindsight oracle
- 新增:
  - `scripts/amv_attack_ok_lab.py`
  - Canvas: `amv-attack-ok-lab.canvas.tsx`
- 运行:
  - `uv run python scripts/amv_attack_ok_lab.py`
  - 产物: `artifacts/amv_attack_ok/20260517_150811/summary.json`
- 标签定义:
  - 从 `artifacts/amv_constrained_oracle/20260517_131848/daily_constrained_choices.csv` 读取
  - 使用 `target_metric = top_ret_dailyized`
  - 使用 `margin = 3%`
  - 使用 `allow_cash = true`
  - `choice_type == attack` 记为 `attack_ok = 1`
  - 全样本 `807` 天: attack `197`, base `527`, cash `83`
- 特征:
  - 使用交易前 `STATE_FEATURES`
  - 优先复用 `artifacts/amv_horizon_oracle_explainability/20260517_131130/oracle_choices_with_state.csv`，避免重复构建 ST/宽池状态
- Walk-forward 测试:
  - test years: `2023 / 2024 / 2025 / 2026`
  - AUC:
    - 2023: `0.489`
    - 2024: `0.497`
    - 2025: `0.549`
    - 2026: `0.619`（样本仅 `32` 天）
  - AP:
    - 2023: `0.270`
    - 2024: `0.273`
    - 2025: `0.247`
    - 2026: `0.625`
- 策略侧诊断（仍是标签侧，不是可交易回测）:
  - `always_base`: avg dailyized `+0.11%`
  - `label_oracle_attack_ok`: avg dailyized `+1.48%`, lift `+1.37pp`, attack `93/369`
  - `model_threshold_0.30`: avg dailyized `+0.24%`, lift `+0.13pp`, precision `0.326`, recall `0.076`, pred attack `24/369`
  - `model_valid_best_f1`: avg dailyized `+1.86%`, lift `+1.75pp`, precision `0.267`, recall `0.841`, pred attack `282/369`
  - `always_attack`: avg dailyized `+2.14%`, lift `+2.03pp`, precision `0.268`, recall `1.000`, pred attack `369/369`
- 当前判断:
  - 第一版 `attack_ok` 没有证明当前状态特征能稳定学习“什么时候进攻”
  - 2023/2024 AUC 接近随机；2025 略高；2026 虽高但样本过少
  - 验证集 F1 阈值在 2024-2026 退化为近似 `always_attack`，说明模型不是在精确识别少数例外，而是在低阈值下高召回
  - 固定 `0.30` 阈值更保守，但只召回 `7/93` 个真实 attack 日，收益提升很小
  - 由于当前 attack 侧仍使用 future-best attack sleeve，`always_attack` 和模型经济指标都只能作为标签侧诊断，不能解释为真实可交易收益
  - 下一步不应直接把当前 `attack_ok` 接交易，而应先收紧目标:
    - 固定单一进攻袖子后重新做二分类
    - 或提高 `attack_ok` 标签门槛，追求更高 precision
    - 或先补 `cash_ok`，减少“该避险却被迫在 base/attack 中选”的噪声
