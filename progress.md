# Progress

## 2026-02-24 (续)

### 架构重构: Python 打分 + Rust 回测分离
- **核心决策**: Python 模型只负责截面打分 (1d/1d), 回测/风控/持仓管理全部交给 Rust ECS 引擎
- 依据:
  - LightGBM 1d/1d 模型偏度已为正 (+0.28), 天然适合日频信号
  - Python 固定持仓 N 天 (3d/3d, 1d/3d) 效果均一般, 无法灵活止损止盈
  - Rust 回测框架已支持 B1 策略的 Parquet 导入, 可复用架构

#### 代码变更
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
