# QuantLab 全貌架构文档

**日期**: 2026-06-09  
**用途**: 架构讨论参考，如实描述当前状态（含痛点）

---

## 1. 顶层视图

```
┌─────────────────────────────────────────────────────────────────┐
│                        QuantLab                                 │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐ │
│  │ TDX 日线  │ → │ strategies/  │ → │  signal. │ → │  Rust   │ │
│  │ DuckDB   │   │ amv/ (Python) │   │  parquet │   │ bt-amv  │ │
│  │          │   │ Polars 因子   │   │          │   │ -topn   │ │
│  └──────────┘   └──────────────┘   └──────────┘   └─────────┘ │
│       │                │                             │         │
│       │          ┌─────┴─────┐                  result.json   │
│       │          │ rules/    │                       │         │
│       │          │ hooks.py  │                  qlab status    │
│       │          └───────────┘                                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │ scripts/explore_factor.py — 因子快速验证          │          │
│  │ (改 FACTOR_TAG + make_factor_expr → uv run → IC) │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据源

### 2.1 TDX 通达信日线 (DuckDB)

**位置**: `D:/WorkSpace/Tinkering/QuantData/Ashare/tdx.db` (Windows)  
**引擎**: DuckDB v1.5.3, 只读模式  
**读取层**: `utils/data_source.py`

| 视图 | 内容 | 列 |
|------|------|-----|
| `v_stock_qfq` | 前复权日线 | symbol, date, open, high, low, close, preclose, volume, amount, floatmv, turnover |
| `v_stock_bfq` | 不复权日线 | 同上 |
| `v_etf_qfq/bfq` | ETF 日线 | 同上 |

**TdxDailyReader.load_daily_full()** SQL:
```sql
SELECT
  left(q.symbol,2)||'.'||right(q.symbol,-2) AS code,  -- sh600000 → sh.600000
  q.date, q.open AS open_adj, q.close AS close_adj, ...,
  b.open AS open_raw, b.close AS close_raw, ...,
  CAST(b.volume AS DOUBLE)/100.0 AS volume,  -- 股→手
  b.amount, q.floatmv/1e8 AS market_cap_100m,
  q.turnover AS turnover,
  CASE WHEN q.turnover>0 THEN volume/turnover ELSE 0 END AS circulating_capital
FROM v_stock_qfq q JOIN v_stock_bfq b USING(symbol, date)
WHERE q.date >= '2019-01-01' AND q.close>0 AND b.close>0 AND b.volume>0
```

**输出**: DuckDB-backed `pl.LazyFrame`, 19 列 (open_adj, close_adj, ..., open_raw, close_raw, ..., market_cap_100m, amount, volume, turnover, circulating_capital)

**协议抽象**: `DailyMarketReader` Protocol, `QmtDailyReader` / `TdxDailyReader` 两个实现, `daily_reader()` context manager。

### 2.2 活跃市值 Regime (独立 DuckDB)

**位置**: `../QuantData/Ashare/active_market_value.duckdb`  
**读取**: `utils/active_market_value_regime.py`

生成每日 AMV bull/bear/neutral 状态, 基于活跃市值指数的日线 OHLC:
- bull trigger: `max(ret_1d...ret_Nd) >= 4.0%` (N=2)
- bear trigger: `ret_1d <= -2.3%`
- 有效 regime = lag(observed, 1d) (T+1 可用)

**输出**: `pl.DataFrame` (~1800 行), columns: date, is_bull_regime, amv_mechanical_regime, amv_ret_1d~5d, amv_close, etc.

### 2.3 ST 黑名单 (缓存)

**读取**: `utils/__init__.py` → `get_st_blacklist_pl(date)`  
**来源**: Baostock API, 本地缓存 `data/.cache/st_blacklist.json`  
**输出**: Python `list[str]` (237 只 ST 股票代码)

---

## 3. strategies/amv/ 模块架构

### 3.1 文件清单与职责

```
strategies/amv/
├── data.py              ← 市场数据层 (MarketConfig + build_market_lazy)
├── factors/__init__.py  ← 因子公式唯一真相源 (re-exports + scoring/ranker + lazy medium)
├── factors/base.py      ← calc_amv_core_factors (11 基础因子)
├── factors/registry.py  ← factor registry (按需计算, 自描述依赖)
├── factors/limit_ecology.py ← 涨停生态特征 (40+ 列)
├── factors/sector_tailwind.py ← 行业特征 (20+ 列)
├── hooks.py             ← Rule Hook 系统 (插件化规则)
├── pipeline.py          ← ranker 系统一入口 (trend/pullback, JSON 驱动)
├── pipeline_event.py    ← event 策略入口 (首板回踩 + weakgate)
├── export.py            ← signal.parquet 写出
├── registry.py          ← Strategy 加载器 (configs/*.json)
├── specs.py             ← 类型定义 (RankerSpec, ScoreComponent, RuleSpec, RULES)
├── regime.py            ← AMV 内部阶段诊断 (27 列, gate frame)
└── configs/             ← JSON 策略配置 (8 个文件)
    ├── _rankers.json    ← 打分模板 (trend: P-block+K, pullback: PB+CP)
    ├── trend-p2.json    ← P=2, 无 rules (archived)
    ├── trend-p3.json    ← P=3, 无 rules
    ├── trend-p3-medium.json ← P=3 + medium-trend-quality rule
    ├── trend-p3-enhanced.json ← P=3 + sector + medium rules
    ├── pullback-pb3.json ← PB=3/CP=1 + amv-regime-gate rule
    ├── event-firstboard.json ← event + weakgate rule
    └── event-firstboard-base.json ← event, 无 weakgate
```

### 3.2 核心数据流 (trend-p3-medium export)

```
qlab export trend-p3-medium
  │
  ├─► registry.load_strategy("trend-p3-medium")
  │     └─ 读取 configs/trend-p3-medium.json
  │         ranker: template="trend", P=3.0 (→ price_pos_20d×3, close_to_high_20d×3, KLEN×0.5, KMID2×0.5)
  │         rules: [{"id":"medium-trend-quality", "params":{"medium_penalty":0.03, "weak_threshold":0.50}}]
  │
  ├─► pipeline.export_ranker_strategy(strategy, config, output_dir)
  │     │
  │     ├─ [Phase 1: Lazy 链]
  │     │   ├─ data.build_market_lazy(config) → (reader, LazyFrame)
  │     │   │   ├─ reader.load_daily_full() → DuckDB LazyFrame
  │     │   │   ├─ filter(ST, date_range)
  │     │   │   ├─ amount_ma20
  │     │   │   └─ join(regime) on date
  │     │   │
  │     │   ├─ calc_amv_core_factors(lf) → 追加 11 基础因子 + K-bar 列
  │     │   │
  │     │   └─ [for each rule hook]
  │     │       └─ MediumTrendQualityHook.lazy_features(lf) → 追加 _structure_score_128d, _quality_score_128d
  │     │         (128d rolling features: ret, pos, trend_eff, up_ratio, ret_vol, ma_slope, body_eff)
  │     │
  │     ├─ [Phase 2: ONE collect]
  │     │   └─ lf.collect() → market_df (8.2M rows, ~50 cols)
  │     │   └─ reader.close()
  │     │
  │     ├─ [Phase 3: Eager 后处理]
  │     │   ├─ _compute_medium_penalty_cols(market_df, params)
  │     │   │   └─ 从 _structure/quality → penalty = strength × 0.03
  │     │   │
  │     │   └─ _score_and_select(market_df, ranker, candidate)
  │     │       ├─ market.with_columns(candidate_expr, base_score, context_penalty)
  │     │       ├─ 非候选人 _signal_score = None
  │     │       ├─ ordinal rank over date (null → 末尾)
  │     │       └─ filter rank ≤ 3 → signal_rows (2492 rows)
  │     │
  │     ├─ [Phase 4: Gate]
  │     │   └─ (trend-p3-medium 无 gate, 直接过)
  │     │
  │     └─ [Phase 5: Export]
  │         ├─ market.select(16 export columns) → 释放 ~3GB 因子列
  │         ├─ shift signal_date → execution_date (T+1)
  │         ├─ left-join: market × execution_signals → full panel
  │         └─ write_signal_artifact → signal.parquet (8.2M rows)
```

### 3.3 Hook 系统

**位置**: `hooks.py`

```python
class RuleHook:
    def lazy_features(lf, params) → LazyFrame   # collect 前, 纯 lazy
    def penalty(params) → Expr                   # 合并到一个 with_columns
    def gate(signals, params) → DataFrame        # TopN 后, ~2000 行

# 已注册的 Hook:
_HOOK_REGISTRY = {
    "medium-trend-quality": MediumTrendQualityHook,  # 128d 结构/质量 penalty
    "amv-regime-gate":      AmvRegimeGateHook,       # AMV 内部阶段 gate
    "event-weakgate":       EventWeakgateHook,        # 7 维弱窗口 gate
}
```

**Pipeline 中使用**:
```python
hooks = resolve_hooks(strategy.rules)  # JSON rule IDs → Hook 实例
for hook in hooks:
    lf = hook.lazy_features(lf, params)     # Phase 1: lazy
market = lf.collect()
for hook in hooks:
    penalty_exprs.append(hook.penalty(params))  # Phase 3: eager
for hook in hooks:
    signal_rows = hook.gate(signal_rows, params)  # Phase 4: gate
```

**关键**: pipeline 本身零 if 分支——新增 rule = 注册 Hook + 改 JSON，不动 pipeline.py。

### 3.4 因子体系

**因子计算入口**: `factors/__init__.py` (唯一真相源)

| 函数 | 来源 | 用途 |
|------|------|------|
| `calc_amv_core_factors(lf)` | `factors/base.py` | 计算 11 基础因子 + K-bar 中间列 (NLEN/KMID/KUP/KSFT等) |
| `build_amv_base_factors(lf, required)` | `factors/base.py` | 调度器: required=None→全量, 否则→registry 按需 |
| `compute_required_factors(lf, names)` | `factors/registry.py` | 按需计算: 解析依赖链, 计算 tier1(6个)+tier2(6个)+因子表达式 |
| `ranker_score_expr(ranker)` | `factors/__init__.py` | RankerSpec → Polars 排名表达式 |
| `ranker_required_columns(ranker)` | `factors/__init__.py` | 提取 ranker 需要的因子名列表 |
| `finite_expr(col_name)` | `factors/__init__.py` | not-null & not-NaN 检测 |
| `add_medium_trend_features_lazy(lf)` | `factors/__init__.py` | 纯 lazy 128d medium 特征 (7 滚动 + 7 rank_pct + composite) |

**基础因子 (11个)**: price_pos_20d, close_to_high_20d, KLEN, KMID2, ret_5d, ret_20d, ma_bias_20, disp_bias_20, intraday_pos, atr_14_pct, panic_vol_ratio_20d

**Ranker 模板** (`_rankers.json`):
- `trend`: P×price_pos_20d + P×close_to_high_20d + K×KLEN + K×KMID2
- `pullback`: PB×ma_bias_20 + PB×disp_bias_20 + CP×KSFT + CP×intraday_pos

---

## 4. scripts/explore_factor.py — 因子探索流程

### 4.1 当前工作流 (如实描述)

1. **修改源代码**: 打开 `scripts/explore_factor.py`
2. **改两个地方**:
   - `FACTOR_TAG = "xxx"` (因子标签)
   - `make_factor_expr()` 函数中激活/注释对应的因子公式
3. **运行**: `uv run python scripts/explore_factor.py`
4. **看输出**: Rank IC, IC IR, 年度明细, 分组收益
5. **Commit**: `git commit -m "factor: xxx IC=0.xx IR=0.xx"` 存档结果

### 4.2 脚本内部结构

```python
# 配置区 (改这里)
START_DATE = "2019-01-01"
END_DATE = "2026-06-03"
FORWARD = 5      # 前向收益天数
FACTOR_TAG = "xxx"

def make_factor_expr() -> ([[step1], [step2], ...], final_expr):
    """返回多步中间列 + 最终因子表达式
    每个内层 list 对应一次 with_columns 调用
    Polars 限制: 同一 with_columns 内不能引用刚创建的列
    """
    # 激活的因子 (当前):
    return [[g], [G], [w]], (close / rp - 1).alias("factor")

    # 注释掉的历史因子: Terrified Score, STV, Quality Momentum,
    #   Upper Shadow, MA Convergence PCF, Price Position

# main():
#   1. TdxDailyReader → load_daily_full → LazyFrame
#   2. for step in steps: lf = lf.with_columns(step)   # 多步中间列
#   3. lf.with_columns([factor_expr, fwd_return])
#   4. lf.collect() → DataFrame
#   5. Rank IC 计算: group_by(date) → corr(rank(factor), rank(fwd_ret))
#   6. 年度汇总 + 分组收益 (5 quintiles)
```

### 4.3 已测试因子总结

| # | 因子 | IC | IR | 单调 | 数据需求 | 备注 |
|---|------|----|----|------|---------|------|
| 1 | Quality Momentum | +0.064 | 0.33 | ❌ | OHLC | Q3>Q5, 极值有问题 |
| 2 | Upper Shadow | -0.012 | -0.16 | ❌ | OHLC | 太弱 |
| 3 | MA Convergence PCF | ~+0.05 | ~0.3 | ✅ | OHLC | 突破前兆 |
| 4 | **Terrified Score** | **-0.085** | **0.64** | ✅ | OHLC | ⭐ 最佳 |
| 5 | STV | -0.067 | 0.40 | ✅ | OHLC+turnover | 不如 Terrified |
| 6 | CGO | -0.052 | 0.43 | ✅ | OHLC+turnover | 处置效应, 纯 Lazy |

### 4.4 当前探索流程的痛点

1. **必须改源码**: 新增因子 = 修改 `make_factor_expr()`, 旧公式注释保留 → 文件越来越长
2. **无法按需加载**: 不能从配置文件或 CLI 参数指定因子
3. **中间列步数管理**: Polars 限制 (同一 `with_columns` 不能引用刚创建的列) 需要在 `make_factor_expr()` 中手动拆步
4. **结果靠人工记忆**: IC/IR 数值需要手动记录, git commit message 是唯一的实验日志
5. **没有与 pipeline 对接**: explore_factor.py 验证通过后, 需要另外在 hooks.py 中实现 Hook 版本

---

## 5. Rust 回测引擎 (bt-amv-topn)

### 5.1 位置与构建

**路径**: `backtest-engine/crates/amv-topn/`  
**构建**: `cargo run -p bt-amv-topn --release`  
**架构**: Bevy ECS (Entity Component System)

### 5.2 输入

**signal.parquet** 格式 (Python 写出):
```
date, code, open_adj, high_adj, ..., close_raw, pre_close_raw,
is_bull_regime, amv_mechanical_regime, market_cap_100m, amount_ma20,
is_signal, signal_date, score, rank, sleeve_id
```

### 5.3 执行模型

| 参数 | 值 |
|------|-----|
| 成交价基准 | raw OHLC + raw pre_close (T+1 open) |
| 最大持仓日 | 6 (static) |
| 最大持仓数 | 3 |
| 仓位分配 | 等权 33.3% |
| 每日最大买入 | 3 |
| 涨跌停过滤 | close 涨停禁开仓 (gap < 9.8%) |
| 牛市过滤 | require_bull_regime (entry-only) |
| 手续费 | 佣金 0.025%, 印花税 0.1%, 滑点 0.1% |
| 重复持仓 | 禁止 (allow_duplicate=false) |

### 5.4 输出

| 文件 | 内容 |
|------|------|
| `report.json` | total_return_pct, max_drawdown_pct, total_trades, win_rate, costs, trading_days |
| `report.txt` | 人类可读摘要 |
| `daily_equity.csv` | 每日净值序列 |
| `trades.csv` | 每笔交易明细 (日期, 代码, 买卖价, PnL) |

### 5.5 可选功能 (默认关闭)

- **ATR 早期止损**: d2+ ATR×3.0 动态止损
- **Trailing Stop**: 激活 10%, 追踪 5%
- **固定止损**: 5% 价格止损

### 5.6 Crate 结构

```
backtest-engine/
├── core/       → lib.rs: 账户, 手续费, 涨跌停判断, 手数计算
├── amv-topn/   → 主线回测 (当前活跃)
│   ├── main.rs:    入口, ATR 计算, 行情加载
│   ├── resources.rs: 配置结构 (AmvTopnConfig), PriceBar
│   ├── systems.rs:   交易逻辑 (买卖/止损/持仓检查)
│   └── components.rs: ECS 组件标记
├── b1/         → 旧 B1 策略线 (保留)
├── b3/         → 旧 B3 策略线 (保留)
├── renko/      → 旧 Renko (保留)
└── rotation/   → 旧 Rotation 轮动 (保留)
```

---

## 6. qlab CLI

### 6.1 命令

```bash
uv run python scripts/qlab.py status              # 扫描 artifacts/ → 显示所有策略指标
uv run python scripts/qlab.py export <strategy>   # 生成 signal.parquet
uv run python scripts/qlab.py backtest <strategy> # 运行 Rust 回测
uv run python scripts/qlab.py run <strategy>      # export + backtest
uv run python scripts/qlab.py results <strategy>  # 查看历史回测记录
```

### 6.2 status 数据来源

`qlab status` 扫描 `artifacts/<strategy>/backtests/*/result.json`, 取每个策略最新的 `is_canonical=true` 记录。不是硬编码。

### 6.3 数据源切换

```bash
uv run python scripts/qlab.py export trend-p3 --data-source qmt  # 切换到 QMT
export QLAB_DATA_SOURCE=tdx && uv run python scripts/qlab.py export trend-p3  # 环境变量
```

默认: TDX。通过 `utils/data_source.py::resolve_data_source()` 解析优先级: CLI args > env vars > default。

---

## 7. 数据格式

### 7.1 signal.parquet

| 列 | 类型 | 说明 |
|----|------|------|
| date | date | 执行日期 (signal_date + 1) |
| code | str | sh.600000 格式 |
| open_adj/high_adj/low_adj/close_adj/pre_close_adj | f64 | 前复权 OHLC (因子计算用) |
| open_raw/high_raw/low_raw/close_raw/pre_close_raw | f64 | 原始 OHLC (成交用) |
| is_bull_regime | bool | 牛市标记 |
| amv_mechanical_regime | str | bull/bear/neutral |
| market_cap_100m | f64 | 流通市值(亿) |
| amount_ma20 | f64 | 20日均成交额 |
| is_signal | bool | 是否为信号日 |
| signal_date | date | 信号产生日期 |
| score | f64 | 信号得分 |
| rank | u32 | 当日前 3 排名 |
| sleeve_id | str | 策略名 (trend-p3-medium 等) |

~8.2M rows, ~1.5GB。每行 = 一个股票-日期对, 信号日期 is_signal=True 的 score/rank 有效, 其余 fill_null(0/9999)。

### 7.2 result.json

```json
{
  "strategy": "trend-p3-medium",
  "is_canonical": true,
  "from_report": {
    "total_trades": 280,
    "total_return_pct": 178.57,
    "max_drawdown_pct": 18.05,
    "win_rate_pct": 52.14,
    "total_costs": 283272
  },
  "yearly": {"2021": 5.42, "2022": 29.68, ...}
}
```

---

## 8. 跨设备情况

| 设备 | 数据源 | 路径 |
|------|--------|------|
| Windows | TDX (tdx.db) | D:\WorkSpace\Tinkering\QuantData\Ashare\ |
| Mac | TDX (tdx.db) | ~/Workspace/QuantData/Ashare/ |
| Mac (旧) | QMT (qmt_data.duckdb) | 已基本弃用 |

**已知差异**: Windows TDX 和 Mac TDX 数据存在系统性差异 (相对排序一致但绝对数值不同)。CURRENT_STATE.md 指标以 Windows 为准。

---

## 9. 已知痛点与限制

### 9.1 内存

- `calc_amv_core_factors` 计算全量 11 因子 + 中间列 → collect 后 ~6.5GB kernel memory
- trend 策略只用 4 个因子, 但全量全算了
- factor registry 按需计算已实现但 pipeline 中用的是 `calc_amv_core_factors`

### 9.2 因子探索闭环断裂

- `explore_factor.py` 验证 → 手动在 `hooks.py` 中重新实现 → 无自动化衔接
- 改一次因子 = 改两个地方 (explore 脚本 + hook 实现)

### 9.3 因子探索配置不灵活

- 新增因子必须改源码 (注释/取消注释)
- 没有 CLI 参数或配置文件驱动的因子选择
- 多步中间列的 `with_columns` 链需要手动管理

### 9.4 Polars 表达式限制

- 同一 `with_columns` 不能引用刚创建的列 → 因子表达式拆步在 `make_factor_expr()` 中硬编码
- `cumprod` 可用但需小心数值精度 (CGO: 5000+ 股 × 2000 天, 权重比值在 float64 范围内)

### 9.5 跨设备

- TDX 数据在 Mac/Windows 间通过 Google Drive 同步
- DuckDB 数据更新流程尚未完全自动化
