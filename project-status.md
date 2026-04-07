# Project Status

> 实验详情见 `experiments/` 目录，本文件仅跟踪当前状态和最优指标。

---

## 一、截面轮动策略

### 统一口径
- `Rotation` 当前定位: **候选子策略**, 不再承担“单策略复刻博主完整体系”的目标
- `Rotation` 默认对标对象: 博主**早期公开的日频截面基线**
- 博主当前完整体系: 见 `experiments/target-strategy-evolution.md`, 应理解为系统级长期目标
- `Rotation` 当前主线候选: `core_plus_alpha158(kbar_shape)`

### 当前活跃路线
- 主线候选: `core_plus_alpha158(kbar_shape)`
- 比较锚点: `core_12 + fwd_ret_1d + EXPORT_EMA_ALPHA=0.30 + hold_buffer=50 + max_hold_days=10`
- 当前优先工作:
  - 固定 `core_plus_alpha158(kbar_shape)`, 暂不继续扩因子
  - 先用训练侧 `Top-20` 专用诊断排查排序尾部与边界问题
  - 回测层当前先验证 `hold_buffer=20` 这条更直接的持仓优化主线
  - `max_daily_buys / entry_rank_limit` 暂作为附加实验工具保留
  - 再考虑 `score_adj` 前的二次整形

### 三层解耦现状
- 分析层: `notebooks/rotation_factor_lab.py`
  - 负责因子 IC、分组汇总、Alpha decay、Alpha158 top1 / 强子集筛选
  - 已修复 marimo 的跨 cell 重名变量与分支 `return` 问题, `uv run marimo check` 通过
- 清单层: `manifests/rotation_feature_sets.py`
  - 负责维护稳定特征集、状态标签 (`active / archived / experimental`) 与说明
  - 现已支持运行时 `custom feature set` 解析, 可承接任意自定义因子组合训练
- 训练层: `notebooks/cross_section_rotation.py`
  - 只消费 manifest 中冻结的特征集
  - 负责训练集构造、Walk-forward 打分、raw score 导出与 artifact 落盘
  - 现已支持 `FEATURE_SET = "custom"` + `CUSTOM_FEATURE_COLS`

### 已收口路线
- `46-all` / `36-pruned`: 不再作为主推版本
- `fwd_ret_1d_rank_pct`: 首轮未证明优于当前主线, 仅后置观察
- `rank_pct / rank_gauss` 截面归一化: 弱于 `zscore`, 当前收口
- `alpha158(kbar_shape)` 单跑: 判定为交互增强器, 不作为独立主线
- `core_plus_alpha158_top1`: 已验证失败, 不继续作为扩容主线

### 已完成
- `utils/rotation_factors.py`: 42 个日线截面因子 (动量/波动/成交量/技术/微观/A股T+1/处置效应)
  - T 日数据版: 所有因子直接用当日 OHLCV, 与实盘 14:45 快照对齐
  - 已新增分组注册表: `FACTOR_GROUPS / FACTOR_GROUP_LABELS / FACTOR_TO_GROUP`
- `utils/duckdb_utils.py`: `add_price_limit_cols()` 涨跌停标记 (与 Rust bt-core 逻辑一致)
- `utils/signal_export.py`: 信号导出 (B1 事件信号 + 截面轮动打分), Parquet 格式供 Rust 消费
  - `Rotation` 已新增 artifact 追踪: `train.meta.json` / `signal.meta.json` / `raw_scores.parquet`
  - `Rotation` 默认不再写 `data/signals/rotation_scores.parquet`, `artifacts/` 为唯一真源
- `bt-core`: 已新增 artifact 追踪公共 I/O
  - `SignalArtifactMeta`
  - `load_signal_meta()`
  - `build_report_stem()`
  - `write_report_bundle()`
  - `resolve_registry_path()`
  - `append_jsonl_record()`
- `notebooks/cross_section_rotation.py`: 训练入口 notebook
  - Cell 1: 读取 `FEATURE_SET` 并从 manifest 加载稳定特征集
  - Cell 1: 现支持 `FEATURE_SET = "custom"` 运行时传入任意自定义因子组合
  - Cell 2: 数据加载 + 因子计算 + 涨跌停标记 + 截面标准化 + 标签构造
  - Cell 2: 仍支持 `zscore / rank_pct / rank_gauss` 与 `fwd_ret_{1/2/3/5}d_rank_pct`
  - Cell 2b: 校验 `amount / volume` 单位口径, 防止 `vwap_raw` 误放大 100 倍
  - Cell 3: 明确提示因子分析已迁移到 `rotation_factor_lab.py`
  - Cell 6: LightGBM Walk-Forward 打分, 输出 `df_scores_raw + rotation_train_meta`
  - Cell 6b: 导出 parquet, 独立控制 `EXPORT_EMA_ALPHA`, 写入 artifact 元数据, 无需重训模型
  - Cell 7: 保留原始分数的 Signal Quality 诊断
  - Cell 8: 新增 `Top-20 Tail Diagnostics`, 专看 rank-by-rank、20/21 边界和尾部拖累
- `backtest-engine/crates/rotation`:
  - 已支持 `max_daily_buys`, 用于限制每日新增仓位数量
  - 已支持 `entry_rank_limit`, 用于限制新开仓只来自更前排的 rank
  - 新旧配置兼容: 未声明新参数时, 自动回落到旧的“尽量补满 + Top-N 候选”行为
  - 当前实验结论: 本轮收益改善首先由 `hold_buffer=50 -> 20` 带来, 入场控制能力尚未单独证明优于简单基线
- `notebooks/rotation_factor_lab.py`: 独立分析 notebook
  - 负责 Rotation / Alpha158 因子 IC、分组汇总、Alpha decay、相关性诊断与 core 候选筛查
  - 已通过 `uv run marimo check`, 可作为稳定分析入口使用
- `manifests/rotation_feature_sets.py`:
  - 当前稳定训练入口: `core_12`, `core_plus_alpha158_kbar_shape`
  - `core_plus_alpha158_top1` 与 `pruned_rotation` 已降级为 `analysis-only`
  - 已新增自定义组合解析能力, 不再要求所有训练组合都预先写死在 registry 中
- 新增 `utils/factor_analysis.py`:
  - `build_ic_summary_frame`
  - `summarize_factor_groups`
  - `extract_group_top_factor_cols`
  - `build_daily_ic_frame`
  - `resolve_decay_factor_cols`
  - `compute_factor_decay`
- `utils/alpha158_factors.py`:
  - 已按 `Qlib Alpha158` 默认模板本地复刻 `158` 因子
  - 已可直接与现有 `Rotation` 因子合并训练
  - `RANK / BETA / RSQR / RESI / IMAX / IMIN / IMXD` 已改为 Polars 原生实现, 不再依赖 `rolling_map`
- Rust 回测引擎:
  - `bt-core`: 涨跌停判定 (`price_limit_pct`, `is_limit_up`, `is_limit_down`)
  - `bt-core`: artifact 追踪公共 I/O (报告 bundle / meta 读取 / registry append)
  - `bt-rotation`: 涨停过滤 (买入) + 跌停锁仓 (卖出) + 过滤统计
  - 报告自动保存到 signal 目录下的 `backtests/<backtest_timestamp_ms>/`
  - 自动读取 `signal.meta.json`, 输出 `report.txt` / `report.json`, 并追加 `artifacts/rotation/<train_run_id>/backtest.jsonl`
  - `run_rotation.bat` 现在是 `python scripts/rotation_backtest.py` 的 Windows 包装器
  - 支持:
    - 交互式选择 `train run -> signal`
    - 或直接传 `signal.parquet / signal.meta.json / signal目录`
    - CLI 覆盖 `top_n / hold_buffer / min_score / max_hold_days`
    - 每次回测自动落盘 `effective.config.toml`, 避免手改 Git 追踪的基线 config

### 我们的核心数据 (2026-03-31 更新, 排除涨停幻觉后的真实 alpha)
| 指标 | 值 | 备注 |
|---|---|---|
| Gross Return (810天) | +76.05% | 当前最佳结果来自 `hold_buffer=20` 版本 |
| Total Return | +24.27% | `core_plus_alpha158(kbar_shape)` + `EXPORT_EMA_ALPHA=0.30` |
| 最大回撤 | 13.31% | 明显优于旧 `hold_buffer=50` 基线 |
| 胜率 | 46.5% | |
| 总交易笔数 | 2,202 | 日均 2.7 笔 |
| 信号 IC Mean | +0.0376 | t-stat = 8.81 ✅ |
| ICIR | +0.3096 | |
| L/S 年化 Sharpe | 1.65 | |
| L/S 胜率 | 54.8% | |
| Top-20 日均换手 | 62.5% | 年化 151x |
| LABEL | fwd_ret_1d | |
| 涨停过滤 (Rust) | 日均 2.0 只 | 模型残余偏好 |

### 当前最新主线候选 (2026-04-03 更新, `core_plus_alpha158 + kbar_shape`)
| 指标 | 值 | 备注 |
|---|---|---|
| Feature Mode | `core_plus_alpha158` | `core_12 + Alpha158(kbar_shape)` |
| Feature Count | 21 | `12 + 9` |
| Normalize | `zscore` | |
| Export EMA | `0.30` | |
| Target IC Mean | +0.0441 | `t-stat = +10.89` |
| ICIR | +0.3827 | |
| L/S Sharpe | 1.47 | 经济评估口径 `fwd_ret_1d` |
| Gross Return | +59.79% | 810 天 |
| Total Return | +18.61% | 当前已知优于旧主线 |
| 最大回撤 | 14.45% | 较旧主线明显改善 |
| 总交易笔数 | 2,457 | 日均 3.0 笔 |
| 总成本 | 205,925 | 其中滑点占比最高 |
| Top-20 日均双边换手 | 106.6% | notebook 诊断口径 |

阶段判断:
- `Alpha158` 首轮对照已出现明确正增益, 且不是只换来更高回撤的“虚胖 alpha”
- 最有效的新增信息当前集中在 `kbar_shape` 组, 与特征重要性结果一致
- 需要继续确认:
  - uplift 是来自 `kbar_shape` 独立 alpha, 还是主要来自其与 `core_12` 的交互
  - notebook 高换手诊断与 Rust 实际成交次数之间的差异, 是否还能通过组合层参数继续压缩成本

### `alpha158(kbar_shape)` 单跑验证 (2026-04-03 更新)
| 指标 | 值 | 备注 |
|---|---|---|
| Feature Mode | `alpha158` | 仅 `kbar_shape` 9 因子 |
| Feature Count | 9 | |
| Export EMA | `0.30` | |
| Target IC Mean | +0.0438 | `t-stat = +12.10` |
| ICIR | +0.4250 | 统计信号并不弱 |
| L/S Sharpe | 1.45 | notebook 经济分层仍显著 |
| Gross Return | +7.65% | 810 天 |
| Total Return | -21.70% | 成本后明显失败 |
| 最大回撤 | 28.68% | 高于当前主线 |
| 总交易笔数 | 2,250 | 日均 2.8 笔 |
| 总成本 | 146,736 | gross alpha 无法覆盖成本 |
| Top-20 日均双边换手 | 124.0% | 高于 `core_plus_alpha158` |

阶段判断更新:
- `kbar_shape` 单跑时只体现出“统计上有信息量”, 但组合层不可兑现
- 因此上一轮 uplift 不能再理解为“发现了可独立替代 `core_12` 的新主线”
- 当前更合理的解释是:
  - `kbar_shape` 对 `core_12` 更像高价值交互增强器
  - 其独立排序能力不足以支撑当前 `Top-20 + hold_buffer=50 + min_score=0.002` 的组合约束
- 后续主线应聚焦:
  - `core_12 + kbar_shape`
  - 以及该组合在组合层的成本压缩与兑现优化

### `core_plus_alpha158_top1` 全组 top1 验证 (2026-04-03 更新)
| 指标 | 值 | 备注 |
|---|---|---|
| Feature Mode | `core_plus_alpha158_top1` | `core_12 + Alpha158(all groups top1)` |
| Feature Count | 21 | `12 + 9` |
| Export EMA | `0.30` | |
| Target IC Mean | +0.0322 | `t-stat = +8.50` |
| ICIR | +0.2987 | 弱于 `kbar_shape` 组合 |
| L/S Sharpe | 1.36 | notebook 经济分层仍为正 |
| Gross Return | +10.40% | 810 天 |
| Total Return | -49.33% | 成本后彻底失败 |
| 最大回撤 | 50.38% | 明显不可接受 |
| 总交易笔数 | 5,631 | 日均 7.0 笔 |
| 总成本 | 298,619 | `slippage + stamp duty` 主导 |
| Top-20 日均双边换手 | 121.4% | 明显高于 `kbar_shape` 组合 |

阶段判断更新:
- `Alpha158` 各组 `top1` 虽然统计上并非完全无效, 但组合层兑现严重失控
- “每组保留 1 个”这个规则对训练入口过于机械, 会把弱组一起带入并稀释强组增量
- 因此 `core_plus_alpha158_top1` 暂时退出主线, 当前 Alpha158 主线仍维持 `core_plus_alpha158(kbar_shape)`

#### 失败实验记录
- **涨停幻觉**: 旧模型 +586% 几乎全是虚假买入涨停股, 过滤后 Gross 仅 +5.7%
- **超额收益标签** (fwd_ret_1d_excess): 五分位单调性崩塌, Top-20 选股退化为随机
- **fwd_ret_2d 标签**: Gross +54% 但换手增加, 净效果不如 fwd_ret_1d
- **128 天长周期因子** (42→55 因子): IC 下降, 回测净收益从 +82% 降至 +30%, 已回退
- **EMA α=1.0** (无平滑): 日均 14.1 笔交易, 成本 > 本金, 净收益 -45%
- **单因子 IC 剪枝** (55→50): 反而更差, 树模型非线性交互使单因子 IC 不适合做删减依据
- **`core_plus_alpha158_top1`**: 虽有 `Gross +10.40%`, 但 `Net -49.33% / MDD 50.38% / 7.0 trades/day`, 属于高换手失控失败

### 原策略关键参数 (2026-01 更新)
| 指标 | 值 | 备注 |
|---|---|---|
| 年化收益率 | 50.42% | 纯日频, 不含日内 T+0 |
| 最大回撤 | 9.13% | |
| 胜率 | 54.01% | |
| 盈亏比 | 3.04 : 2.02 | ≈ 1.5:1 |
| 日收益偏度 | 0.90 | 正偏, 厚尾盈利 |
| 总交易笔数 | 7475 | 5 年, 日均 ~6.2 笔 |
| 最大持仓 | 20 只 | |
| 平均仓位 | ~50% | 留现金给日内做 T |
| **平均持仓天数** | **2.8 天** | **非严格 T+1 轮动** |
| Alpha | 0.60 | |
| Beta | 1.78 (R²=0.28) | 高 beta 暴露 |
| **子策略数量** | **12 个** | **多策略组合, 非单模型** |
| 日内 T+0 系统 | 1 分钟频率 | 实盘额外贡献 +10-20% 年化 |

### 与目标策略的差距

详见 `experiments/target-strategy-evolution.md`。当前应区分“博主早期公开的日频截面基线”与“后期多策略 rule-based 体系”。核心差距已不只是单策略指标, 还包括多策略集成、状态切换、分钟级 T+0 与混合数据源。

### Rotation 后续方向

详见 `experiments/rotation-next-phase.md`。

当前下一阶段优先级:
1. 将 `core_plus_alpha158(kbar_shape)` 作为当前最强主线候选, 与旧 `core_12` 基线并行对照
2. `alpha158(kbar_shape)` 单跑已验证不适合作为独立主线, 暂不继续深挖
3. 先讨论并设计“因子挖掘分析/因子选择”与“主训练/导出/回测”更彻底解耦的架构, 再决定下一轮 notebook 改造
4. 若继续扩 Alpha158, 优先看更强子集而非 `top1_all`, 避免弱组稀释与高换手失控
5. 围绕 `core_plus_alpha158(kbar_shape)` 做组合层兑现优化: 优先看 `hold_buffer / min_score / Top-N` 对成本的压缩空间
6. `fwd_ret_1d_rank_pct` 仍可保留观察, 但继续后置于组合层优化
7. 暂不急于上全量 `158` 因子, 继续按小分组渐进扩容

备注:
- `Rotation` 当前标的池已经是 **80~500 亿**, 这不是下一阶段待修正项
- 导出侧 EMA 扫描已完成, `0.30` 为当前净收益最佳点, `0.28` 为次优平衡点
- `cross_section_rotation.py` 已修复 Cell 6 的 `LABEL` 动态过滤 bug, 后续切换标签不会再误用写死的 `fwd_ret_1d`
- `cross_section_rotation.py` 已新增 `fwd_ret_{1/2/3/5}d_rank_pct`，可在不改模型结构的前提下直接验证排序化标签
- `cross_section_rotation.py` 已新增 `NORMALIZE_MODE`，可直接对照 `zscore / rank_pct / rank_gauss` 三种特征截面归一化
- `utils/signal_export.py` 已把 `normalize_mode` 写入 artifact 元数据，后续导出可直接区分不同归一化实验
- `cross_section_rotation.py` 的 Cell 7 现已采用双口径评估: `Target IC` 跟随 `LABEL`, 经济分层固定按 `fwd_ret_1d`
- `cross_section_rotation.py` 的 Cell 6 现已支持 `core_plus_alpha158_top1`, 可直接消费 `Cell 3` 产出的各组 `top1` 因子清单
- `core_plus_alpha158_top1` 首轮验证已失败, 说明当前不宜把“各组 top1 全并入训练”作为默认扩容路径
- `fwd_ret_1d_rank_pct` 首轮观察未显示优于当前 `fwd_ret_1d` 主线, 尤其经济分层与最终回测暂未改善
- `NORMALIZE_MODE = rank_pct / rank_gauss` 两轮实验均显著弱于 `zscore`, 当前不再作为主线优化方向
- 组合参数扫描显示 `max_hold_days=15` 的纯净收益更高, 但当前不升格为“早期日频对标锚点”, 以免偏离博主早期公开基线“平均持仓 2.8 天”的节奏特征
- `stock_daily.volume` 单位已验证为“手”, 因此 `vwap_raw` 与 `turnover_rate` 现统一按 `volume * 100` 还原股数后再计算
- `Alpha158` 已完成本地 Polars 复刻并接入 `Rotation` notebook, 下一步进入首轮基线对照
- `Alpha158` 最重的 `35` 个窗口因子现已去除 Python 回调, 与旧 `rolling_map` 版本数值对齐
- `alpha158` 单跑实验现会自动跳过 `Rotation` 因子分析 Cell, 以缩短实验等待时间
- `Alpha158(kbar_shape)` 首轮 `core_plus_alpha158` 对照已给出当前更优净收益: `Gross +59.79% / Net +18.61% / MDD 14.45%`
- 从当前特征重要性看, 新增 `kbar_shape` 因子里 `KUP2 / KLOW2 / KUP / KLOW / KMID2 / KSFT / KMID / KSFT2 / KLEN` 整体贡献突出
- `alpha158(kbar_shape)` 单跑虽有较高 `IC / ICIR`, 但 Rust 回测仅 `Gross +7.65% / Net -21.70% / MDD 28.68%`
- 这说明 `kbar_shape` 当前更像是 `core_12` 的交互增强器, 而非可单独替代主线的独立特征集

---

## 二、B1 超跌反转策略

### 已完成
- `utils/b1_factors_opt.py`: B1 信号计算 (V3.0 + 周线 MACD 过滤)
- `utils/signal_export.py`: B1 事件信号导出, `export_for_rust()` 支持 `extra_sort_cols`
- `notebooks/b1_ml_explore.py`: 全市场 ML 打分 → B1 排序 (56 因子, MFE-10)
- `notebooks/b1_ml_dedicated.py`: B1 专属模型 (仅 B1 信号日训练, 38 因子)
- Rust `bt-b1` 回测引擎: 止损/止盈/弱势出场/追踪止损/WL 跌破

### B1 排序方案对比 (2026-03-25)

| 排序方式 | 近期收益 (290天) | 长周期收益 (774天) | 最大回撤 |
|---|---|---|---|
| 手搓 `rw_dif_pct` | +30.49% | **+81.05%** | ~9.3% |
| **全市场 ML** | **+36.63%** | +78.36% | ~13% |
| B1 专属 ML | +21.38% | +43.83% | ~8% |

### 当前最优方案

**全市场 ML 模型排序** (`b1_ml_explore.py`):
- 近期跑赢手搓 +6pp, 长周期基本持平
- IC +0.1373, L/S +3.95% — 信号质量远强于 B1 专属模型
- 回撤偏大 (13% vs 9%), 但整体收益更高

### B1 后续方向

1. **全市场模型迭代**: 特征工程和标签优化, 收益同时辐射 B1 和 rotation 两个策略
2. **降低回撤**: 可能需要对 B1 出场逻辑做调参 (止损比例、弱势天数)
3. **活跃市值择时**: 当前依赖手工标注 `LOOSE_PERIODS`, 后续自动化

---

## 三、Renko 短线策略

### 当前状态
- `notebooks/renko_ml_explore.py`: 已启动时间线重构, 统一为 `T 日收盘确认 → T+1 开盘买入`
- Renko 专属 `rk_*` 因子已统一为 **T 日收盘可得**
- 标签已改为 **`buy_open_t1` 基准**:
  - `fwd_mfe_5d = max(high[T+1:T+5]) / open[T+1] - 1`
  - `fwd_ret_1d = close[T+1] / open[T+1] - 1`
- 已新增可切换标签入口:
  - `fwd_ret_open_2d`
  - `fwd_ret_close_2d`
  - `fwd_ret_close_3d`
- 已新增 notebook 实验面板:
  - EMA 平滑实验
  - Top-N 扩大实验
  - 高分阈值过滤实验
- 导出侧已支持独立 `EXPORT_EMA_ALPHA`, 且参数位于 Cell 6 本地, 可单独重跑 Cell 6 生成不同平滑版本 parquet
- 导出侧“raw score → export EMA”模式后续可迁移到 `cross_section_rotation.py`, 便于只调导出平滑而不重训
- Rust 导出 / 回测格式 **暂不继续深入调整**
- 当前结论: `fwd_ret_open_2d` 在 notebook 层有一定统计信号, 但组合回测对 EMA 过于敏感, 且成本后净收益持续为负, **Renko 方向暂时暂停**

### 当前待办
1. 保留当前 notebook / 导出 / Rust 代码, 作为后续重启 Renko 研究的基线
2. 若未来重启, 优先重新审视 label 语义、分钟级数据与事件驱动回测框架
3. 将导出侧独立 EMA 的模式择机复用到截面轮动导出链路

---

## 四、共享基础设施

### 当前架构

```
Python (信号层)                    Rust (回测/执行层)
┌────────────────────┐            ┌──────────────────────────────┐
│ rotation_factors   │            │ Cargo Workspace              │
│   42 通用因子 (T日) │            │ ├─ bt-core (共享)             │
│ add_price_limit    │            │ │  涨跌停判定 / Portfolio     │
│   涨跌停标记        │            │ ├─ bt-b1   (B1 超跌反转)     │
│ LightGBM 1d/1d    │──Parquet──→│ └─ bt-rotation (截面轮动)     │
│   排除涨停训练      │            │    涨停过滤 / 跌停锁仓       │
│ signal_export      │            │ report.json + backtest.jsonl │
│ b1_factors_opt     │            │ 报告: signal/backtests/ 目录  │
└────────────────────┘            └──────────────────────────────┘
```

### Artifact 追踪约定
- `Rotation` 训练 artifact:
  - `artifacts/rotation/<train_run_id>/train.meta.json`
  - `artifacts/rotation/<train_run_id>/raw_scores.parquet`
- `Rotation` 导出 artifact:
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/signal.parquet`
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/signal.meta.json`
  - 所有路径尽量使用相对路径, 方便 Windows / macOS 双设备共用
- `Rotation` 回测 artifact:
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/backtests/<backtest_timestamp_ms>/report.txt`
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/backtests/<backtest_timestamp_ms>/report.json`
  - `artifacts/rotation/<train_run_id>/signals/<signal_timestamp_ms>/backtests/<backtest_timestamp_ms>/effective.config.toml`
- 全局索引:
  - `artifacts/rotation/<train_run_id>/signals.jsonl`
  - `artifacts/rotation/<train_run_id>/backtest.jsonl`
  - 可按 `signal_id / label / feature_hash / EXPORT_EMA_ALPHA / 回测参数` 检索
- 设计原则:
  - 公共部分下沉到 `bt-core`
  - 策略专属统计和配置结构仍留在各自 crate
  - 避免为统一而统一, 兼顾复用性与策略个性
  - Git 追踪的 `config.toml` 只保存稳定基线, 实验参数优先通过脚本 CLI 覆盖
- `signals/<signal_timestamp_ms>/` 目录采用纯时间戳:
  - 目录只负责唯一性与顺序
  - 真正的导出参数统一记录在 `signal.meta.json` / `signals.jsonl`
- 后续待办:
  - 将 `bt-renko`
  - 将 `bt-b1`
  - 逐步接入同一套 `bt-core` artifact/report bundle I/O, 但保留各策略独立配置与额外统计

### 实验记录

详见 `experiments/` 目录:
- `target-strategy-evolution.md` — 博主策略演化与多策略全景
- `rotation-factors.md` — Rotation 因子实验 (128d 失败、处置效应、EMA)
- `b1-ml-fullmarket.md` — B1 全市场 ML 排序实验
- `b1-ml-dedicated.md` — B1 专属模型实验 (结论: 不如全市场)
