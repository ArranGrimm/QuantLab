# Progress

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
