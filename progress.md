# Progress

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
