# Project Status

> 实验详情见 `experiments/` 目录，本文件仅跟踪当前状态和最优指标。

---

## 一、截面轮动策略

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
- `notebooks/cross_section_rotation.py`: 研究 notebook (7 个 cell)
  - Cell 1-2: 数据加载 + 因子计算 + 涨跌停标记 + 截面标准化 + 超额收益标签
  - Cell 3: Spearman IC 分析 + 排行榜 + 累积 IC 曲线
  - Cell 3a/3d: 因子分组概览 + 核心因子筛查, 输出建议 `core feature set`
  - Cell 3b: Alpha Decay 分析 (因子预测力随持仓天数衰减)
  - Cell 4-5: (已清空, 旧线性回测)
  - Cell 6: LightGBM Walk-Forward 打分, 本地控制 `FEATURE_MODE`, 输出 `df_scores_raw + rotation_train_meta` (训练时排除涨停样本)
  - Cell 6b: 导出 parquet, 独立控制 `EXPORT_EMA_ALPHA`, 写入 artifact 元数据, 无需重训模型
  - Cell 7: Signal Quality Analysis (基于原始分数, 独立 EMA, 无需重训模型)
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

### 我们的核心数据 (2026-03-31 更新, 排除涨停幻觉后的真实 alpha)
| 指标 | 值 | 备注 |
|---|---|---|
| Gross Return (809天) | +48.12% | 约 15%/年, 真实 alpha |
| Total Return | +10.51% | 0.1% 滑点 |
| 最大回撤 | 21.34% | |
| 胜率 | 45.6% | |
| 总交易笔数 | 2,202 | 日均 2.7 笔 |
| 信号 IC Mean | +0.0376 | t-stat = 8.81 ✅ |
| ICIR | +0.3096 | |
| L/S 年化 Sharpe | 1.65 | |
| L/S 胜率 | 54.8% | |
| Top-20 日均换手 | 62.5% | 年化 151x |
| LABEL | fwd_ret_1d | |
| 涨停过滤 (Rust) | 日均 2.0 只 | 模型残余偏好 |

#### 失败实验记录
- **涨停幻觉**: 旧模型 +586% 几乎全是虚假买入涨停股, 过滤后 Gross 仅 +5.7%
- **超额收益标签** (fwd_ret_1d_excess): 五分位单调性崩塌, Top-20 选股退化为随机
- **fwd_ret_2d 标签**: Gross +54% 但换手增加, 净效果不如 fwd_ret_1d
- **128 天长周期因子** (42→55 因子): IC 下降, 回测净收益从 +82% 降至 +30%, 已回退
- **EMA α=1.0** (无平滑): 日均 14.1 笔交易, 成本 > 本金, 净收益 -45%
- **单因子 IC 剪枝** (55→50): 反而更差, 树模型非线性交互使单因子 IC 不适合做删减依据

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

详见 `experiments/rotation-benchmark.md`。核心差距: 回撤控制 (9% vs 27%)、胜率 (54% vs 42%)、多策略集成 (12 vs 1)。

### Rotation 后续方向

详见 `experiments/rotation-next-phase.md`。

当前下一阶段优先级:
1. 导出侧独立 `EXPORT_EMA_ALPHA`, 提升研究效率 (已在 `Rotation` / `Renko` notebook 落地)
2. 因子治理, 收敛 `core feature set` (分组基础设施 + 核心筛查入口已完成)
3. 增加 `Ridge/ElasticNet` 与 `CatBoost/XGBoost` 基线对照
4. 在固定研究基线后再收敛组合参数

备注:
- `Rotation` 当前标的池已经是 **80~500 亿**, 这不是下一阶段待修正项

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
- 全局索引:
  - `artifacts/rotation/<train_run_id>/signals.jsonl`
  - `artifacts/rotation/<train_run_id>/backtest.jsonl`
  - 可按 `signal_id / label / feature_hash / EXPORT_EMA_ALPHA / 回测参数` 检索
- 设计原则:
  - 公共部分下沉到 `bt-core`
  - 策略专属统计和配置结构仍留在各自 crate
  - 避免为统一而统一, 兼顾复用性与策略个性
- `signals/<signal_timestamp_ms>/` 目录采用纯时间戳:
  - 目录只负责唯一性与顺序
  - 真正的导出参数统一记录在 `signal.meta.json` / `signals.jsonl`
- 后续待办:
  - 将 `bt-renko`
  - 将 `bt-b1`
  - 逐步接入同一套 `bt-core` artifact/report bundle I/O, 但保留各策略独立配置与额外统计

### 实验记录

详见 `experiments/` 目录:
- `rotation-benchmark.md` — 目标策略指标
- `rotation-factors.md` — Rotation 因子实验 (128d 失败、处置效应、EMA)
- `b1-ml-fullmarket.md` — B1 全市场 ML 排序实验
- `b1-ml-dedicated.md` — B1 专属模型实验 (结论: 不如全市场)
