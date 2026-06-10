# QuantLab 当前状态

这是项目第一阅读入口，也是唯一当前状态看板。它只描述 raw execution 修正后的当前可交易 AMV 状态、关键判断、活跃风险和日常命令。

## 策略命名规则

`家族-变体`：family 表达策略逻辑，variant 表达参数/规则差异。

| 家族 | 逻辑 | 代表策略 |
|---|---|---|
| `trend` | 趋势突破延续（贴近 20 日新高） | trend-p2, trend-p3, trend-p3-medium, trend-p3-enhanced |
| `pullback` | 回调反弹（均线/离散回踩） | pullback-pb3 |
| `event` | 事件驱动（涨停生态） | event-firstboard |

## 真实口径

- 可交易结论以 Rust `bt-amv-topn` 为准。
- 交易执行使用 `raw_ohlc_pre_close`（raw OHLC + raw pre-close）。
- 旧 adjusted-execution 回测只作为历史参考，不能当当前真实指标。

## 策略指标（raw execution，2026-06-10 Mac TDX 重跑）

| 策略 | Return | MaxDD | Trades | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|--------|-------|--------|------|------|------|------|------|------|
| **trend-p3** (基线) | +169.5% | 17.6% | 282 | +1.2 | +38.2 | +16.6 | +61.1 | +12.8 | -8.9 |
| **trend-p3-medium** | +162.6% | 19.8% | 282 | — | — | — | — | — | — |
| **pullback-pb3** (互补) | +42.5% | 15.0% | 1082 | — | — | — | — | — | — |
| **event-firstboard** (研究) | +192.2% | 36.1% | 268 | — | — | — | — | — | — |
| **event-firstboard-base** | +119.0% | 42.3% | 320 | — | — | — | — | — | — |

> 注：数据源 Mac TDX（排除北交所）。trend-p3-enhanced = trend-p3-medium（sector 在申万分类下暂未生效）。

## 当前 Baseline

`trend-p3`（趋势突破 P3 挑战者，**升格为基线**）

- Raw execution `6td static strict Top3`，282 笔交易，+169.5% / 17.6%
- trend-p3-medium 在 Mac TDX 上增量不成立（+162.6% vs trend-p3 +169.5%），降格为保留
- AMV 牛市确认了 P3 的 alpha 来源：当 AMV bull 时随机买收益显著高（20 日 +2.0% vs 全年 +0.9%），P3 在此基础上做 Top3 选股 +169.5%（详见 `research/explore_regime.py`）
- trend-p2 已归档，不再研究

## 当前 Challenger

`trend-p3-enhanced`（trend-p3 + 行业顺风）

- 当前与 trend-p3-medium **完全等价**（sector-tailwind 在申万分类下 penalty 0.02 太小，不改变排名）

## 历史参考

`trend-p2`（已归档）— 未重跑
`trend-p3-medium`（保留）— +162.6% / 19.8%，用于隔离 medium 增强的边际效果。在 Mac TDX 上相比纯 trend-p3 无增量

## Pullback Sleeve

`pullback-pb3`（回调 PB3/CP1/RV0 rolling + AMV 风控）

- Raw execution `6td rolling21 refill Top10`，1089 笔交易
- 规则: amv-regime-gate（aged + 非加速 OR 混沌期连续阴跌，开仓日 gate）
- 2026 年全为正（+7.2%），与 trend 家族日收益相关性 ~0.26
- 状态: 互补 sleeve，待做 raw-execution allocation 分析

## 涨停生态 Sleeve

`event-firstboard`（首板后回踩 + 弱窗口过滤）

- Raw execution `5td static strict Top3`，265 笔交易（weakgate），315 笔（base）
- 规则: event-weakgate (gate)
- 2021 年爆赚（+54.4%），但 2022 年唯一亏损策略（-4.7%）
- 年度分布与 trend/pullback 家族完全不同，是潜在独立第三 sleeve
- MaxDD 34.1% 仍是主要障碍，尚未 allocation-ready

## 关键判断

- 当前主线已从 Rotation / B1 / B3 收敛到 AMV Bull Pool TopN。
- raw execution 没有推翻趋势突破方向，但压低旧 adjusted 口径收益约 25-30pp。
- **AMV 活跃市值牛熊识别本身就是 alpha**：牛市中随机买 5 只拿 20 天月均 +2.3%（t=+2.02），比全年无脑买多赚 +1.4%。P3 Top3 选股在此基础上进一步做截面分化（2023 年 AMV 择时亏 -1.5% 但 P3 选股赚 +16.6%）。
- 经典择时指标（RSRS、Breadth、CSVC）均不如 AMV 适用于全市场选股场景。
- pullback-pb3 与 trend 家族低相关，是自然互补 sleeve。
- 中期结构/趋势质量增强（medium penalty）在 Mac TDX 上增量不成立，基线从 medium 降为纯 P3。
- 行业顺风（sector-tailwind）从东方财富切换到申万分类后，原有参数不生效，需重新调参。

## 活跃风险与未决项

- Mac TDX 与 Windows TDX 存在系统性差异（北交所过滤 + 股票池不同），跨设备 baseline 尚未对齐。
- 2026 年：trend 家族全线亏损，pullback 和 event 正收益。
- sector-tailwind 需要针对申万分类重新做参数扫描（当前最优 candidate: `5d/none/bt=0.4/p=0.15` 但 Rust 验证未通过）。
- pullback-pb3 需要 redo raw-execution allocation 分析。
- AKShare 已完全移除，行业分类从东方财富 → 申万（Baostock，`utils/baostock_utils.py`）。
- regime 慢退出机制逻辑成立但样本量太小（43 段牛市），需要更多数据避免过拟合。

### ETF 动量轮动（原型，2026-06-03）

- 脚本: `strategies/etf_momentum_rotation.py`
- 数据源: TDX `v_etf_qfq`（动量）/ `v_etf_bfq`（成交），通达信活跃ETF block 自然筛选 22 只
- 核心逻辑: 25d 加权 OLS 动量分 + AMV 择时 + 信号切换时换仓
- **纯 Python 原型，未经 Rust 回测引擎验证，不含涨跌停/流动性过滤**
- 原型指标: +362% / MaxDD 21.3% / Sharpe 1.0 / 116 笔 / 2019-2026 年仅 1 年亏损
- 回撤偏高（21%），参数敏感性高（15d→+130%，20d→+290%，25d→+362%），尚不适合作为正式 sleeve
- 与 trend/pullback/event 低相关（2026 年互补明显），是潜在第四维度，待后续 Rust 验证与回撤改善

### 数据源：TDX 通达信行情（2026-06-03）

- `utils/duckdb_utils.py` `load_daily_data_full()` 新增 `db_source="tdx"` 支持
- TDX `v_stock_qfq` + `v_stock_bfq` → 输出列与 QMT 完全一致（5751 只含北交所 vs QMT 4969）
- 代码格式自动转换 `sh600000` → `sh.600000`
- 通过 `WorkflowExportConfig.db_source` 切换，下游 pipeline 无感
- 可解除 QMT 数据更新的跨设备依赖，但仍需更多验证后才切换为默认源

### 研究工具（2026-06-10）

- `research/explore_factor.py`：因子截面 IC 探索（改 FACTOR_TAG → 跑 → 看 IC/IR），自动记录 `factor_ledger.jsonl`。支持 registry 因子 + make_factor_expr() 自定义双路径
- `research/explore_regime.py`：择时 gate 探索（B1 alpha proof 方法——随机买 vs 择时买），蒙特卡洛 100 次采样，多持有期扫描。已验证 AMV > RSRS > Breadth > CSVC
- `QuantLab-0.1.0-alpha/`：历史 alpha 版本，保留完整的 B1/B3/Rotation 探索脚本和结论

### 因子发现管线

- 因子注册表 `factors/registry.py`：顶层共享模块，17 个因子（12 active + 5 experimental），按需计算，按状态分 active/experimental/dead
- QuantsPlaybook Tier-1：10 个候选已测 6 个（terrified_score IC -0.085 IR -0.64 ⭐、cgo_100d、coin_team、ubl、quality_momentum、stv_score_20d）
- 因子探索：改 FACTOR_TAG → `uv run python research/explore_factor.py` → 看 IC/IR

### 架构重构（2026-06-05 ~ 2026-06-10）

- `strategies/amv/hooks.py`: RuleHook 基类 + 3 个 hook
- `strategies/amv/pipeline.py`: 4 阶段流程，投影裁剪，无 if 分支
- `strategies/amv/pipeline_event.py`: 合并双数据源（10GB→8GB）
- `factors/registry.py`: 顶层共享因子模块
- `research/explore_factor.py`: 因子探索（接入 registry）
- `research/explore_regime.py`: 择时 gate 探索（B1 alpha proof 方法）

- `strategies/amv/hooks.py`: RuleHook 基类 + 3 个 hook（MediumTrendQualityHook / AmvRegimeGateHook / EventWeakgateHook）
  - 规则从 pipeline 的 if 分支解耦为可插拔 hook，JSON 配置驱动
  - penalty 合并为单次 `with_columns`，gate 合并为单次 `filter`，零中间副本
- `strategies/amv/pipeline.py`: ranker 系统一入口，4 阶段流程（lazy → select pushdown → one with_columns → one filter）
- `strategies/amv/pipeline_event.py`: event 专用管道，合并双数据源为单 lazy chain（内存 10GB → 8GB）
- 删除：medium_trend_quality.py（内联进 hook）、workflows.py / signals.py / scoring.py / market.py / rules/
- 因子注册表 `compute_required_factors` 按需计算（trend 只算 4 因子 + 3 中间列，vs 全量 20+）

## 新增能力 (2026-06-02)

### Rust 引擎：ATR 止损（可选）

- `bt-amv-topn` 新增 `[early_stop]` 配置 section，ATR-based 动态止损
- 从 `trigger_hold_trading_days`（默认 2）起，每天检查 `close < entry_price - atr_multiple × ATR_14` → 卖出
- 默认关闭。最佳参数 `d2/ATR×3.0` 在测试中给出 +240.3%（+13pp vs baseline）
- 配置项：`enabled`, `trigger_hold_trading_days`, `atr_multiple`
- 影响文件：`systems.rs`, `resources.rs`, `main.rs`

### 数据源：上证指数日线

- `utils/baostock_utils.py` 新增 `get_sh_index_daily()`，Baostock 拉取 sh.000001 日线
- 缓存为 `data/sh_index_daily.parquet`，覆盖 2019-01-02 ~ 至今
- 可用于 AMV 牛市确认的交叉验证

### 组合天花板分析

- 三 sleeve 等权组合：年化 19.6%，MaxDD 14.6%，Sharpe 1.36
- 最优权重（50/30/20）：年化 21.0%，MaxDD 9.3%，Sharpe 1.53
- 三条 sleeve 相关性：trend-pb3 0.247, trend-event 0.222, pb3-event 0.238

### 其他已验证结论

- event-firstboard 不适合 rolling（static +155% → rolling +20%），应保持 static + regime gate 改善
- event-firstboard 涨停污染极低（仅 3 次/256 笔），天然避开涨停不可买问题
- sector-tailwind 申万分类 focused grid scan 已完成（240 组合），但最优参数在 Rust 验证中未通过

## 不要再当当前真实结论使用

- 旧 adjusted-execution 指标。
- Python close-to-close label 收益。
- 任何没有通过 Rust raw execution 复核的策略结论。
- 旧名称 `ref` / `p3` / `context` / `pb3-gated` / `limit-weakgate`。
- `qlab status` 的数值来自 `artifacts/<strategy>/backtests/<latest>/result.json`，不是硬编码。

## 文档归属

- `CURRENT_STATE.md`: 当前真实口径、核心指标、关键判断、活跃风险和日常命令。
- `strategies/target-strategy-evolution.md`: 小红书博主外部对标与 QuantLab 长期方向。
- `strategies/archive-index.md`: B1 / B3 / Rotation / LTR / oracle 等已归档路线的回看索引。

## 日常命令

```bash
# 查看状态（自动从 result.json 读取）
uv run python scripts/qlab.py status

# 导出信号 → artifacts/<strategy>/signal.parquet
uv run python scripts/qlab.py export trend-p3

# 回测 → artifacts/<strategy>/backtests/<ts>/
uv run python scripts/qlab.py backtest trend-p3

# 查看历史结果
uv run python scripts/qlab.py results trend-p3
uv run python scripts/qlab.py results trend-p3 --diff

# 一键导出 + 回测
uv run python scripts/qlab.py run trend-p3

# 回测参数覆盖
uv run python scripts/qlab.py backtest trend-p3 --top-n 5 --max-hold 3
```
