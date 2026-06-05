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

## 策略指标（raw execution，2026-06-05 Windows TDX 重跑）

| 策略 | Return | MaxDD | Trades | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|--------|-------|--------|------|------|------|------|------|------|
| **trend-p3-medium** (基线) | +178.6% | 18.0% | 280 | +5.4 | +29.7 | +9.4 | +68.4 | +20.7 | -8.2 |
| **trend-p3** (挑战) | +114.9% | 17.6% | 280 | — | — | — | — | — | — |
| **pullback-pb3** (互补) | +46.4% | 15.8% | 1075 | — | — | — | — | — | — |
| **event-firstboard** (研究) | +203.6% | 41.3% | 271 | — | — | — | — | — | — |

> 注：数据源为 Windows TDX，与 Mac QMT 存在系统性差异但相对排序一致。trend-p3-enhanced = trend-p3-medium（sector 在申万分类下暂未生效）。

## 当前 Baseline

`trend-p3-medium`（趋势突破 P3 + 中期结构 / 趋势质量）

- Raw execution `6td static strict Top3`，280 笔交易
- 总收益 `+178.6%`，MaxDD `18.0%`
- 相对 trend-p3: 总收益 `+63.7pp`
- 规则: medium-trend-quality (linear penalty, p=0.03)
- trend-p2 已归档，不再研究

## 当前 Challenger

`trend-p3-enhanced`（trend-p3-medium + 行业顺风）

- 当前与 trend-p3-medium **完全等价**（sector-tailwind 在申万分类下 penalty 0.02 太小，不改变排名）
- 待行业分类参数重新调优后再评估

## 历史参考

`trend-p2`（已归档）— 未重跑
`trend-p3`（保留）— +147.1% / 16.3%，用于隔离 P-block 权重提升的边际效果

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
- pullback-pb3 与 trend 家族低相关，是自然互补 sleeve。
- 中期结构/趋势质量增强（medium penalty）是当前最强单因子提升，+49.6pp vs raw P3。
- 行业顺风（sector-tailwind）从东方财富切换到申万分类后，原有参数（p=0.02 linear）不再生效，需要重新调参。
- 涨停生态有独立 alpha 线索，但 MaxDD 34% 仍需改善。

## 活跃风险与未决项

- 2026 年：trend 家族全线亏损（-8~-9%），pullback 正收益（+7.2%），event 持平。
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

### 架构重构（2026-06-05）

- `strategies/amv/data.py`: MarketConfig + build_market_lazy（reader 生命周期外提，一次 collect）
- `strategies/amv/factors/__init__.py`: 因子公式唯一真相源 + lazy medium features
- `strategies/amv/pipeline.py`: ranker 系统一入口（trend/pullback），JSON 配置驱动
- `strategies/amv/pipeline_event.py`: event 专用管道
- 删除：workflows.py / signals.py / scoring.py / market.py / rules/
- 从 22 文件收敛到 ~10 文件，一条 `qlab export` 穿过 4 个文件
- 内存尚未优化（全量 15 因子 + 128d 特征，约 6.5GB），后续迭代

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
