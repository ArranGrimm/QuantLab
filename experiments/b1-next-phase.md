# B1 Next Phase

## 当前结论

- `B1` 当前应视为可随市场 regime 切回的候选主线, 但不等于立即放弃 `Rotation`
- `活跃市值` 继续保留为**人工日常判断**，当前不做自动化复刻
- 当前最稳定的 `B1` 主链不是旧 notebook，而是:
  - `utils/b1_factors_opt.py::calc_b1_factors_wmacd`
  - `utils/signal_export.py::export_for_rust`
  - `backtest-engine/crates/b1`
  - `agent/`

## 现状梳理

### 规则侧

- `calc_b1_factors_wmacd` 已不只是通达信原公式直译，而是:
  - `B1 V3.0` 日线条件
  - `running weekly MACD`
  - `monthly MACD`
  - 可选 `WEEKLY_WL_YL_FILTER / WEEKLY_TREND_FILTER / WAVE_OVERHEAT_FILTER`
- `agent/` 当前默认使用:
  - 规则版 `B1` 候选
  - `rw_dif_pct` 排序
  - 大模型图表 + 结构化指标评审

### ML 侧

- 全市场 ML 排序 (`experiments/b1-ml-fullmarket.md`)：
  - 近期优于手搓 `rw_dif_pct`
  - 长周期与手搓基本持平
  - 回撤高于手搓
- `B1` 专属 ML (`experiments/b1-ml-dedicated.md`)：
  - 因候选池太窄，截面噪声大
  - 当前整体弱于全市场 ML

### 执行侧

- `bt-b1` 的交易语义已经稳定:
  - `T` 日收盘确认信号
  - `T+1` 开盘买入
  - 分批止盈 / 弱势清仓 / 最大持有天数 / 跌破 `WL` / 移动止损
- `is_loose` 目前仍来自手工 `LOOSE_PERIODS`

## 明确不做的事

- 暂不自动化 `活跃市值`
- 暂不把 `B1` 重新退回“完全手工看图挑股”的旧工作流
- 暂不把 `B1` 主线直接切成“纯 B1 专属 ML”

## 下一步可选路线

### 路线 A：规则版 B1 先回主线

- 用 `calc_b1_factors_wmacd + rw_dif_pct + bt-b1` 直接重启
- 优点:
  - 最贴近历史实盘经验
  - 主链最稳定
  - 调试成本最低
- 风险:
  - 每日候选偏多
  - 最终仍容易回到人工图形决策

### 路线 B：规则召回 + 全市场 ML 精排

- 先用规则版 `B1` 生成候选池
- 再用全市场 ML 分数排序候选
- 优点:
  - 保留 `B1` 策略边界
  - 避免把最终选股完全交给人工
  - 已有历史实验支持
- 风险:
  - 需要重新确认当前 regime 下 ML 排序是否仍稳定
  - 回撤控制仍需配合 `bt-b1` 出场规则优化

### 路线 C：规则召回 + Agent 辅助决策

- 规则先筛
- `Agent` 负责图表与结构化指标二次评审
- 优点:
  - 最接近原始实盘工作流
  - 容易解释
- 风险:
  - 仍带有人机协作链路
  - 一致性和吞吐量取决于 `Agent` 评审质量

## 当前更推荐的方向

- 当前默认优先考虑 **路线 B**
- 即:
  - `活跃市值` 人工判断是否进入多头区间
  - 规则版 `B1` 负责召回候选
  - 全市场 ML 负责精排
  - `Agent` 作为辅助审查层，而非唯一决策层

## 重新切回 B1 前的准备工作

1. 先确认最新一段 `活跃市值` 是否已进入新的多头区间
2. 明确第一版主线到底采用:
   - `rw_dif_pct`
   - 还是全市场 ML `score`
3. 重新核对 `bt-b1` 当前配置是否仍适配新 regime
4. 明确 `Agent` 在第一版里是:
   - 仅解释辅助
   - 还是参与准入过滤

## 备注

- 如果后续确认多头区间持续，`B1` 的研究优先级可正式升回主线
- `Rotation` 当前可先保留为已验证可用的辅线基线，不必继续优先深挖
