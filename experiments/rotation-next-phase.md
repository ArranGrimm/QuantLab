# Rotation 下一阶段可执行清单

> 目标: 在保留当前真实 gross alpha 的前提下, 提升 `Rotation` 的可兑现性与研究效率。
> 当前共识: `Rotation` 已完成涨停幻觉修复, 标的池已限制为 **80~500 亿**, 下一阶段不再把市值池收敛当成主任务。
> 2026-04-03 更新: `core_12 + Alpha158(kbar_shape)` 已跑出当前更优主线候选, 后续不再默认以“`core_12` 必然是最终主线”作为前提。
> 术语约定: 本文中的“对标”若无特别说明, 均指博主**早期公开的日频截面基线**, 不指其当前完整的多策略 `rule-based` 体系。

## 一、跨设备统一口径


| 层级              | 定义               | 当前含义                                               |
| --------------- | ---------------- | -------------------------------------------------- |
| 系统级长期目标         | 博主当前公开可见的完整体系    | 多策略 `rule-based` + 状态切换 + 分钟级 T+0 + 部分混合基本面 / 另类数据 |
| `Rotation` 当前定位 | 我们项目里正在打磨的一条主研究线 | 一个**候选子策略**, 不再承担“单策略复刻完整体系”的目标                    |
| 早期日频对标锚点        | 用来横向比较的稳定参考线     | 博主早期公开的日频截面基线, 以及我们冻结的 `core_12` 基线                |
| 当前主线候选          | 近期应优先推进的方案       | `core_plus_alpha158(kbar_shape)`                   |


### Agent 工作约定

1. 讨论 `Rotation` 时, 默认目标是“把一个子策略做强做稳”, 不是单策略复刻博主完整体系。
2. 讨论“对标”时, 默认是对标博主**早期日频公开基线**, 不是对标她当前 12 策略系统的最终实盘表现。
3. 若需要讨论长期系统方向, 统一引用 `target-strategy-evolution.md`, 不把系统级目标直接压到 `Rotation` 单条线上。

## 二、阶段目标

### 主目标

- 保住当前 `gross alpha` (`Gross Return` 约 +47% ~ +48%)
- 降低交易成本侵蚀, 争取提升 `net return`
- 确认真实 alpha 来自哪些核心因子, 而不是重复表达
- 建立 `LightGBM` 之外的模型基线, 避免单模型路径依赖

### 非目标

- 暂不引入分钟级数据
- 暂不做遗传算法自动挖因子
- 暂不扩展到新的短线策略主线

## 三、执行顺序

### Phase 1: 导出链路解耦

#### 任务 1.1

- 将 `notebooks/cross_section_rotation.py` 的导出逻辑改为:
  - 训练 Cell 只输出 `df_scores_raw`
  - 导出 Cell 单独控制 `EXPORT_EMA_ALPHA`

#### 完成标准

- 修改导出平滑参数时, 只需重跑导出 Cell
- 不需要重新训练 `LightGBM`
- 与当前 `Rotation` 导出 parquet 格式保持兼容

#### 验证

- 对比 `EXPORT_EMA_ALPHA = 1.0 / 0.2 / 0.1 / 0.05`
- 确认导出文件生成正常, Rust 回测可直接复用

### Phase 2: 建立统一实验面板

#### 任务 2.1

- 新增一张统一实验表, 记录每次实验的:
  - `LABEL`
  - `feature_set`
  - `model`
  - `EXPORT_EMA_ALPHA`
  - `top_n`
  - `hold_buffer`
  - `max_hold_days`
  - `min_score`
  - `gross return`
  - `net return`
  - `max drawdown`
  - `avg trades/day`

#### 建议位置

- 先放在 `experiments/rotation-next-phase.md` 末尾
- 如果后续实验很多, 再拆成单独表格文件

#### 面板完成标准

- 后续每一轮回测都能横向比较
- 不再依赖聊天记录回忆参数组合

### Phase 3: 因子治理

#### 任务 3.1 因子分组

- 将当前 `Rotation` 因子按信息源分组:
  - 短期反转
  - K 线结构
  - 波动/风险
  - 成交量/流动性
  - 中期动量
  - 位置/偏离率

#### 任务 3.2 核心因子筛查

- 对每个因子检查:
  - OOS `IC mean`
  - `ICIR`
  - 分年稳定性
  - 与核心因子的相关性
  - 去掉该因子后对 `gross return` 的影响

#### 任务 3.3 收敛核心集

- 目标不是把 40+ 因子硬砍成极少数
- 目标是收敛到 **8~15 个核心信息源**, 其余只保留少量补充项

#### 因子治理完成标准

- 能明确回答“哪些因子是真核心 alpha, 哪些只是重复表达”
- 形成一版 `core feature set`
- 形成一版 `full feature set`

### Phase 4: 模型基线对照

#### 任务 4.1

- 在 `LightGBM` 之外增加:
  - `Ridge` 或 `ElasticNet`
  - `CatBoost` 或 `XGBoost`

#### 任务 4.2

- 对三类模型统一比较:
  - OOS `IC`
  - Quintile 单调性
  - Rust `gross return`
  - Rust `net return`
  - turnover / costs

#### 模型对照完成标准

- 确认 `LightGBM` 是否真的优于简单线性模型
- 避免“复杂模型只是在样本里更会拟合”

### Phase 5: 组合参数收敛

#### 任务 5.1

- 在固定最佳 `feature_set + model + LABEL` 后, 只搜索以下参数:
  - `EXPORT_EMA_ALPHA`
  - `top_n`
  - `hold_buffer`
  - `max_hold_days`
  - `min_score`

#### 任务 5.2

- 每轮只改一类参数, 避免多变量混淆

#### 参数收敛完成标准

- 找到一套可复现的“当前最优组合参数”
- 确认 `gross` 和 `net` 的最优点是否一致

## 四、暂不做的事

### 遗传算法

- 暂不作为下一阶段主线
- 原因:
  - 当前 alpha 偏薄
  - 组合成本敏感
  - 容易把噪音当信号

### 分钟级数据

- 暂不引入
- 只有当日频 `Rotation` 已稳定、且确认瓶颈来自执行误差时再讨论

### 新策略主线

- 暂停 `Renko`
- 下一阶段只聚焦 `Rotation`

## 五、优先级列表

### P0

- `core_plus_alpha158(kbar_shape)` 升格为主线候选并继续补对照
- 补跑 `alpha158(kbar_shape)` 单跑, 隔离验证独立贡献
- 围绕新候选先做组合层压成本 (`hold_buffer / min_score / top_n`)

### P1

- 因子分组 + 核心因子筛查
- `cross_section_rotation.py` 继续保留 `core_12` 基线, 但不再默认优先于 `kbar_shape` 增量线

### P2

- 增加 `Ridge/ElasticNet`
- 增加 `CatBoost/XGBoost` 对照

### P3

- 若以上都稳定, 再讨论遗传算法是否值得作为局部搜索工具

## 六、阶段退出标准

满足以下任意一条, 即可结束本阶段:

1. 找到一套比当前基线更优、且 `gross/net/drawdown/turnover` 更平衡的方案
2. 证明当前因子体系的真实 alpha 已接近日频上限, 再继续堆模型收益不大
3. 证明简单模型与复杂模型差异极小, 则后续重心转向执行与组合层优化

## 七、当前基线

- 标的池: **80~500 亿**
- `LABEL`: `fwd_ret_1d`
- 冻结早期日频对标锚点:
  - `Feature Set = core_12`
  - `EXPORT_EMA_ALPHA = 0.30`
  - `hold_buffer = 50`
  - `max_hold_days = 10`
  - `Gross Return = +51.19%`
  - `Net Return = +16.16%`
  - `Avg Trades/Day = 2.6`
- 当前最新主线候选:
  - `Feature Set = core_plus_alpha158(kbar_shape)`
  - `Feature Count = 21`
  - `EXPORT_EMA_ALPHA = 0.30`
  - `hold_buffer = 50`
  - `max_hold_days = 10`
  - `Gross Return = +59.79%`
  - `Net Return = +18.61%`
  - `Max Drawdown = 14.45%`
  - `Avg Trades/Day = 3.0`

### 基线对照表


| 口径       | Feature Set                      | EXPORT_EMA_ALPHA | Hold Buffer | Max Hold Days | Gross     | Net       | MDD      | Trades/Day | 用途        |
| -------- | -------------------------------- | ---------------- | ----------- | ------------- | --------- | --------- | -------- | ---------- | --------- |
| 早期日频对标锚点 | `core_12`                        | `0.30`           | `50`        | `10`          | `+51.19%` | `+16.16%` | `19.32%` | `2.6`      | 跨设备统一比较锚点 |
| 当前主线候选   | `core_plus_alpha158(kbar_shape)` | `0.30`           | `50`        | `10`          | `+59.79%` | `+18.61%` | `14.45%` | `3.0`      | 当前优先推进方案  |
| 纯净收益最优参考 | `core_12`                        | `0.30`           | `50`        | `15`          | `+50.06%` | `+17.00%` | `19.48%` | `2.5`      | 仅作收益上界参考  |


### 当前活跃路线

1. 主线候选固定为 `core_plus_alpha158(kbar_shape)`
2. 比较锚点固定为 `core_12 + EMA=0.30 + hold_buffer=50 + max_hold_days=10`
3. 下一步优先工作:
  - 压组合层成本: `hold_buffer / min_score / top_n`
  - 讨论“因子分析 / 因子选择 / 主训练”三层解耦
4. 暂不优先:
  - `LGBMRanker`
  - `rank_pct / rank_gauss` 归一化继续扩实验
  - `alpha158(kbar_shape)` 单跑继续深挖
  - `core_plus_alpha158_top1` 继续扩容

### 已收口 / 后置路线


| 路线                                       | 当前结论                          | 状态   |
| ---------------------------------------- | ----------------------------- | ---- |
| `46-all`                                 | 弱于当前主线, 不再主推                  | 收口   |
| `36-pruned`                              | 弱于 `46-all`, 不再主推             | 收口   |
| `fwd_ret_1d_rank_pct`                    | 统计信号未转化为更优净收益                 | 后置观察 |
| `NORMALIZE_MODE = rank_pct / rank_gauss` | gross alpha 明显塌陷, 弱于 `zscore` | 收口   |
| `alpha158(kbar_shape)` 单跑                | 更像交互增强器, 非独立主线                | 收口   |
| `core_plus_alpha158_top1`                | 高换手高成本失控                      | 收口   |
| `max_hold_days = 15`                     | 纯净收益更高, 但只作收益上界参考             | 保留参考 |


### 历史归档说明

- 下方按日期记录的是**历史实验细节**与推理过程。
- 新 agent 若只想快速进入当前主线, 应优先阅读:
  - `跨设备统一口径`
  - `当前基线`
  - `当前活跃路线`
  - `下一步扫描顺序`

### 2026-04-02 历史结论

- `46-all` 不再作为下一阶段主推版本
- `36-pruned` 在当前口径下弱于 `46-all`, 不建议继续作为主线
- `core_12` 仍是当前主线特征集
- 导出侧 `EXPORT_EMA_ALPHA` 扫描后确认:
  - `0.30` 给出当前最高 `net return` (`+16.16%`)
  - `0.28` 是接近峰值的次优平衡点
  - `1.0 / 0.4` 说明平滑不足时换手和成本会显著失控
  - `0.1 / 0.05` 说明平滑过强会压缩真实 alpha
- 组合参数扫描后确认:
  - `hold_buffer = 50` 仍是当前最优退出阈值
  - `max_hold_days = 15` 给出最高 `net return` (`+17.00%`)
  - 但 `max_hold_days = 15` 偏离博主早期日频基线“平均持仓 2.8 天”的节奏特征
  - 因此当前保留两套口径:
    - 纯净收益最优参考: `EMA=0.30 / hold_buffer=50 / max_hold_days=15`
    - 早期日频对标锚点: `EMA=0.30 / hold_buffer=50 / max_hold_days=10`
- 下一阶段临时研究基线:
  - `Feature Set = core_12`
  - `LABEL = fwd_ret_1d`
  - `EXPORT_EMA_ALPHA = 0.30`
  - `hold_buffer = 50`
  - `max_hold_days = 10`

### 2026-04-02 历史结论: 训练目标复盘

- 当前 `cross_section_rotation.py` 的训练目标**没有根本性错误**:
  - 标签是 `fwd_ret_1d`
  - 模型是 `LGBMRegressor`
  - 本质上是在做“次日收益回归”
- 但当前存在明显的**目标错配**:
  - 训练时优化的是收益幅度 (`MSE`)
  - 实际使用时做的是当日截面 `Top-N` 排名
  - 因此当前更像“收益回归打分”, 而不是“直接学排序”
- 已确认:
  - `fwd_ret_1d_excess` 这条路之前已经失败, 暂不回到超额收益标签主线
  - 现在最值得优先验证的不是继续扫组合参数, 而是**先把标签语义做成更贴近排序**
  - 最小落地方案已确定为: 先试 `fwd_ret_1d_rank_pct`，继续使用 `LGBMRegressor`
- 已修复一处 notebook 明显 bug:
  - Cell 6 训练样本过滤原先写死为 `fwd_ret_1d`
  - 现已改为跟随 `LABEL` 动态过滤, 避免后续切换到 `fwd_ret_2d / fwd_ret_5d / rank label` 时样本集错位

### 2026-04-03 历史结论: 排序化标签首轮观察

- Cell 7 现改为双口径:
  - `7a Target IC` 跟随 `LABEL`
  - `7b / 7d Economic Evaluation` 固定按 `fwd_ret_1d`
- 这样做的原因:
  - `rank_pct` 标签本身是单日截面上的单调映射
  - 对 `Spearman IC` 来说, 它可能和原始 `fwd_ret_1d` 非常接近
  - 因此不能只看 `IC Mean / ICIR` 就判断排序化标签成功
- `fwd_ret_1d_rank_pct` 首轮手动复核结果:
  - `Target IC` 依然较高
  - 但固定 `fwd_ret_1d` 的 Quintile / L-S 经济分层不理想
  - Rust 回测端也暂未表现出优于当前 `fwd_ret_1d` 主线的净收益
- 当前解释:
  - 这更像是“标签语义改动本身尚未带来可兑现 alpha”
  - 若后续继续给 `rank_pct` 第二轮机会, 应优先重标定 `min_score / top_n`
  - 在此之前, 不宜仅凭 `IC` 继续推进到 `LGBMRanker`
- 新增一个可立即执行的平行实验入口:
  - 保持 `LABEL=fwd_ret_1d` 或 `fwd_ret_1d_rank_pct` 不变
  - 仅切换特征截面归一化 `NORMALIZE_MODE = zscore / rank_pct / rank_gauss`
  - 用同一套组合参数比较 `Cell 7` 经济分层与 Rust 净收益
- 截面归一化实验结论补充:
  - `NORMALIZE_MODE = rank_pct`:
    - `Gross +8.66% / Net -12.76% / Avg Trades 2.0 / MDD 21.97%`
  - `NORMALIZE_MODE = rank_gauss`:
    - `Gross +9.13% / Net -14.26% / Avg Trades 2.1 / MDD 21.47%`
  - 当前 `zscore` 基线:
    - `Gross +51.19% / Net +16.16% / Avg Trades 2.6 / MDD 19.32%`
- 解释:
  - `rank_pct / rank_gauss` 确实略微压低了交易频率
  - 但并没有改善成本后的净值, 反而让 gross alpha 大幅塌陷
  - 这说明当前 `core_12` 因子集与 `LightGBM` 仍在利用部分幅度信息, 不能简单替换为纯 rank 系输入
- 阶段性结论:
  - “特征归一化改成 rank 系”这条线先收口
  - 当前主线继续保留 `zscore`
  - 后续优先级回到组合层低换手兑现能力, 而不是继续深挖归一化变体

### 2026-04-03 历史结论: Alpha158 `kbar_shape` 首轮结论

- 首轮对照组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = core_plus_alpha158`
  - `Alpha158 分组 = kbar_shape`
  - 总特征数 = `21` (`core_12 + 9` 个 `kbar_shape` 因子)
  - `NORMALIZE_MODE = zscore`
  - `EXPORT_EMA_ALPHA = 0.30`
- `Cell 7` 信号质量:
  - `Target IC Mean = +0.0441`
  - `ICIR = +0.3827`
  - `t-stat = +10.89`
  - 经济分层 `L/S Sharpe = 1.47`
  - `Top-20` 日均双边换手约 `106.6%`
- Rust 回测结果:
  - `Gross Return = +59.79%`
  - `Net Return = +18.61%`
  - `Max Drawdown = 14.45%`
  - `Total Trades = 2457`
  - `Avg Trades/Day = 3.0`
  - `Total Costs = 205,925`
- 与冻结早期日频对标锚点 `core_12 + fwd_ret_1d + EMA=0.30 + hold_buffer=50 + max_hold_days=10` 相比:
  - `gross` 提升
  - `net` 提升
  - `drawdown` 明显改善
- 当前解释:
  - `Alpha158` 不是只能靠“全量 158 因子堆料”才可能有效
  - `kbar_shape` 这组小而强的增量因子已足够进入主线候选
  - 特征重要性显示新增贡献集中在 `KUP2 / KLOW2 / KUP / KLOW / KMID2 / KSFT / KMID / KSFT2 / KLEN`
- 仍需确认的问题:
  - notebook 高换手诊断能否通过组合层参数进一步压缩为更优净值兑现
  - 是否还存在别的 `Alpha158` 小分组, 能像 `kbar_shape` 一样作为增量增强器

### 2026-04-03 历史结论: Alpha158 `kbar_shape` 单跑验证

- 单跑组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = alpha158`
  - `Alpha158 分组 = kbar_shape`
  - 特征数 = `9`
  - `NORMALIZE_MODE = zscore`
  - `EXPORT_EMA_ALPHA = 0.30`
- `Cell 7` 结果:
  - `Target IC Mean = +0.0438`
  - `ICIR = +0.4250`
  - `t-stat = +12.10`
  - `L/S Sharpe = 1.45`
  - `Top-20` 日均双边换手 = `124.0%`
- Rust 回测:
  - `Gross Return = +7.65%`
  - `Net Return = -21.70%`
  - `Max Drawdown = 28.68%`
  - `Avg Trades/Day = 2.8`
- 当前解释:
  - `kbar_shape` 单跑时统计信号不弱, 但组合层几乎不可兑现
  - 这意味着上一轮 `core_12 + kbar_shape` 的 uplift 已基本可判定主要来自**交互增强**, 而不是独立 alpha 替代
  - 因此当前不应把 `alpha158(kbar_shape)` 继续作为独立主线推进
- 研究决策更新:
  - 保留 `core_plus_alpha158(kbar_shape)` 为主线候选
  - 停止继续深挖 `alpha158(kbar_shape)` 单跑
  - 下一步集中到 `core_12 + kbar_shape` 的组合层成本压缩

### 2026-04-03 历史结论: Alpha158 各组 `top1` 组合验证

- 验证组合:
  - `LABEL = fwd_ret_1d`
  - `FEATURE_MODE = core_plus_alpha158_top1`
  - `ALPHA158_ANALYSIS_GROUP_MODE = all`
  - 特征数 = `21` (`core_12 + 9` 个 Alpha158 各组 `top1`)
  - `NORMALIZE_MODE = zscore`
  - `EXPORT_EMA_ALPHA = 0.30`
- `Cell 7` 结果:
  - `Target IC Mean = +0.0322`
  - `ICIR = +0.2987`
  - `t-stat = +8.50`
  - `L/S Sharpe = 1.36`
  - `Top-20` 日均双边换手 = `121.4%`
- Rust 回测:
  - `Gross Return = +10.40%`
  - `Net Return = -49.33%`
  - `Max Drawdown = 50.38%`
  - `Avg Trades/Day = 7.0`
  - `Total Costs = 298,619`
- 当前解释:
  - “每组保留 1 个”对训练入口过于机械, 会把弱组一起并入
  - 模型被高换手因子牵引后, 统计信号难以兑现为净值
  - 这条线不如 `core_plus_alpha158(kbar_shape)` 干净, 且明显弱于当前主线候选
- 研究决策更新:
  - `core_plus_alpha158_top1` 暂时退出主线
  - 若后续继续扩 Alpha158, 优先考虑更强子集, 而不是 `top1_all`
  - 先不要继续往当前主训练面板里叠更多“分析侧筛出来的特征选择逻辑”

### 待讨论待办: 因子挖掘分析 / 因子选择 与主训练流程更彻底解耦

- 背景:
  - 当前 `cross_section_rotation.py` 已同时承载:
    - 因子计算
    - 因子分析
    - 因子选择
    - 模型训练
    - 导出与回测前准备
  - 随着 `Alpha158` 扩容, cell 之间依赖正在变复杂, 后续维护成本会继续上升
- 目标方向:
  - 让“因子挖掘分析 / 筛选规则 / top1 或强子集决策”独立于主训练 notebook
  - 主训练流程只消费一个稳定的 `feature manifest / feature set config`
  - 避免训练 notebook 直接依赖分析面板的中间产物
- 待确认问题:
  - 是拆成独立 notebook, 还是下沉为 `utils/` + 配置文件驱动
  - 特征选择结果如何持久化: `json/yaml/parquet/meta`
  - 主训练入口最终按“预定义特征集名称”还是“显式文件清单”消费
- 状态:
  - 已记录为下一轮架构讨论待办
  - 暂不在本轮直接实施, 先等需求与方案收敛

### 下一步扫描顺序 (2026-04-03 更新)

1. 保留冻结早期日频对标锚点作为比较锚点:
  - `Feature Set = core_12`
  - `LABEL = fwd_ret_1d`
  - `EXPORT_EMA_ALPHA = 0.30`
  - `hold_buffer = 50`
  - `max_hold_days = 10`
2. 将 `core_plus_alpha158(kbar_shape)` 升格为新的主线候选:
  - 先保持 `LABEL / EMA / backtest` 参数不变
  - 暂不急着扩到全量 `158` 因子
3. `alpha158(kbar_shape)` 单跑已验证:
  - 不适合作为独立主线
  - 当前判断以“交互增强器”为主
4. `core_plus_alpha158_top1` 已验证失败:
  - 不再继续把“各组 top1 全量并入训练”作为主线方向
  - 后续若继续扩 Alpha158, 优先看更强子集或人工治理后的集合
5. 先讨论并明确“因子分析 / 因子选择 / 主训练”三层解耦方案:
  - 避免继续把分析逻辑堆回主 notebook
  - 待方案确定后再做下一轮结构重构
6. 围绕新候选先做组合层压成本:
  - `hold_buffer`
  - `min_score`
  - `top_n`
  - 必要时再看 `max_hold_days`
7. `fwd_ret_1d_rank_pct` 保留观察, 但后置:
  - 不再高于 `Alpha158(kbar_shape)` 与组合层兑现优化
  - 只有当经济分层或净收益出现明确改善时, 才继续推进
8. 若排序化标签明显改善 `IC / Quintile / Rust Top-N`, 再尝试 `LGBMRanker`
9. 主线特征与训练目标稳定后, 再进入模型基线对照:
  - `Ridge / ElasticNet`
  - `CatBoost / XGBoost`

原因:

- 当前 `core_12` 已证明因子治理方向有效
- `Alpha158` 已完成本地 Polars 复刻, 且 `kbar_shape` 首轮对照已出现明确正增益
- `alpha158(kbar_shape)` 单跑已证明其独立可兑现性较弱, 主要价值更像对 `core_12` 的交互增强
- `core_plus_alpha158_top1` 已证明“各组 top1 全并入训练”会显著放大换手与成本, 不适合作为默认扩容路径
- 当前早期日频对标锚点已经足够稳定, 现在更值得优先优化新候选的组合层兑现能力
- 直接上 `LGBMRanker` 有可能更好, 但调参复杂度更高, 不应跳过中间的排序化标签验证
- 在标签语义未收敛前, 暂不进入 `Ridge/ElasticNet` 与 `CatBoost/XGBoost` 对照

---

## 八、实验记录表


| 日期         | LABEL        | Feature Set                      | Model      | Export EMA | Top-N | Hold Buffer | Max Hold Days | Min Score | Gross     | Net       | MDD      | Trades/Day | 备注                               |
| ---------- | ------------ | -------------------------------- | ---------- | ---------- | ----- | ----------- | ------------- | --------- | --------- | --------- | -------- | ---------- | -------------------------------- |
| 2026-04-02 | `fwd_ret_1d` | `all`                            | `LightGBM` | `0.15`     | `20`  | `50`        | `10`          | `0.002`   | `+26.84%` | `-8.42%`  | `23.24%` | `2.8`      | 46 因子统计层可用, 组合层退化                |
| 2026-04-02 | `fwd_ret_1d` | `core`                           | `LightGBM` | `0.15`     | `20`  | `50`        | `10`          | `0.002`   | `+28.85%` | `+9.91%`  | `11.92%` | `1.4`      | 旧临时基线, 已被 `0.30` 替代              |
| 2026-04-02 | `fwd_ret_1d` | `pruned`                         | `LightGBM` | `0.15`     | `20`  | `50`        | `10`          | `0.002`   | `+20.59%` | `-10.22%` | `25.90%` | `2.5`      | 全局相关性剪枝不适合作为当前主线                 |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `1.00`     | `20`  | `50`        | `10`          | `0.002`   | `-1.05%`  | `-73.74%` | `74.24%` | `11.2`     | 无平滑, 成本失控                        |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.40`     | `20`  | `50`        | `10`          | `0.002`   | `+29.28%` | `-11.99%` | `25.24%` | `3.8`      | 平滑不足, 净收益转负                      |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.35`     | `20`  | `50`        | `10`          | `0.002`   | `+46.96%` | `+6.72%`  | `20.14%` | `3.2`      | 接近峰值区间上沿, 换手偏高                   |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.32`     | `20`  | `50`        | `10`          | `0.002`   | `+44.20%` | `+7.95%`  | `21.03%` | `2.9`      | 高于 `0.30` 的方向开始退化                |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `50`        | `10`          | `0.002`   | `+51.19%` | `+16.16%` | `19.32%` | `2.6`      | 当前早期日频对标锚点                       |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.28`     | `20`  | `50`        | `10`          | `0.002`   | `+47.81%` | `+15.69%` | `18.85%` | `2.4`      | 峰值附近次优平衡点                        |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.25`     | `20`  | `50`        | `10`          | `0.002`   | `+39.31%` | `+11.07%` | `18.64%` | `2.2`      | 比 `0.30` 更稳但收益偏低                 |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.20`     | `20`  | `50`        | `10`          | `0.002`   | `+28.64%` | `+5.64%`  | `12.81%` | `1.7`      | 平滑加强后 alpha 开始衰减                 |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.10`     | `20`  | `50`        | `10`          | `0.002`   | `+13.36%` | `+0.51%`  | `9.16%`  | `1.0`      | 过度平滑                             |
| 2026-04-02 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.05`     | `20`  | `50`        | `10`          | `0.002`   | `+12.04%` | `+2.23%`  | `8.52%`  | `0.8`      | 过度平滑, gross alpha 明显受损           |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `35`        | `10`          | `0.002`   | `+45.94%` | `+9.71%`  | `19.43%` | `2.8`      | `hold_buffer` 过紧, 净值退化           |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `50`        | `10`          | `0.002`   | `+51.19%` | `+16.16%` | `19.32%` | `2.6`      | `hold_buffer` 当前最优               |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `70`        | `10`          | `0.002`   | `+48.82%` | `+15.44%` | `19.27%` | `2.5`      | 接近峰值, 但未超越 `50`                  |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `90`        | `10`          | `0.002`   | `+41.41%` | `+9.81%`  | `19.46%` | `2.5`      | 退出过慢, 收益回落                       |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `120`       | `10`          | `0.002`   | `+44.39%` | `+12.77%` | `19.38%` | `2.4`      | 拉宽缓冲无额外收益                        |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `50`        | `5`           | `0.002`   | `+35.62%` | `-2.61%`  | `18.70%` | `3.1`      | 持有上限过短, alpha 未兑现                |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `50`        | `7`           | `0.002`   | `+47.31%` | `+10.06%` | `18.37%` | `2.9`      | 更贴近博主风格, 但净收益下降                  |
| 2026-04-03 | `fwd_ret_1d` | `core_12`                        | `LightGBM` | `0.30`     | `20`  | `50`        | `15`          | `0.002`   | `+50.06%` | `+17.00%` | `19.48%` | `2.5`      | 纯净收益最佳, 但偏离早期日频锚点节奏              |
| 2026-04-03 | `fwd_ret_1d` | `alpha158(kbar_shape)`           | `LightGBM` | `0.30`     | `20`  | `50`        | `10`          | `0.002`   | `+7.65%`  | `-21.70%` | `28.68%` | `2.8`      | 单跑统计信号不弱, 但组合层不可兑现               |
| 2026-04-03 | `fwd_ret_1d` | `core_plus_alpha158(kbar_shape)` | `LightGBM` | `0.30`     | `20`  | `50`        | `10`          | `0.002`   | `+59.79%` | `+18.61%` | `14.45%` | `3.0`      | 新主线候选, `gross/net/drawdown` 同时改善 |
| 2026-04-03 | `fwd_ret_1d` | `core_plus_alpha158_top1`        | `LightGBM` | `0.30`     | `20`  | `50`        | `10`          | `0.002`   | `+10.40%` | `-49.33%` | `50.38%` | `7.0`      | 各组 `top1` 全并入训练失败, 高换手高成本失控      |


