# B1 Next Phase

## 当前结论

- `B1` 后续统一以本文档作为主结论文档，不再拆散维护。
- `活跃市值` 继续保留为**人工 regime 判断**，当前不做自动化复刻。
- `B1` 当前必须拆成两条互不混淆的链路:
  - 规则链独立运行: `utils/b1_factors_opt.py::calc_b1_factors_wmacd` -> `utils/signal_export.py::export_for_rust` -> `backtest-engine/crates/b1`
  - ML 链严格按 `Lab -> Train -> Export` 拆分:
    - `notebooks/b1_condition_mining.py`
    - `notebooks/b1_seed_ml_baseline.py`
    - `utils/b1_feature_pool.py`
- 历史结论仍成立:
  - 全市场 ML 排序有价值
  - 旧 `B1 dedicated ML` 因最终候选日截面过窄，不适合作为默认主线
- 基于 `2021-06-21 ~ 2026-03-31` 的六组起始日期对比，`seed_mid + ML score` 在 `2021/2022/2023/2024/2025` 五组窗口里都显著跑赢规则版，当前仍应视为 `B1` 主线。
- 规则版 `B1` 的定位当前更适合作为:
  - 低回撤基线
  - regime 切换期的 sanity check
- `2026-01-01` 起的超短窗口里，规则版阶段性反超 `ML`，说明当前 `ML` 的主要问题已经不是“有没有 alpha”，而是**近端 regime 适应能力与回撤控制**。

## 六组回测对比

### 规则版 vs `seed_mid + ML score` (`2026-04-12`)

| 起始日期 | ML 收益 | ML 回撤 | ML 胜率 | 规则收益 | 规则回撤 | 规则胜率 | 阶段判断 |
|---|---|---|---|---|---|---|---|
| `2021-06-21` | `+244.87%` | `27.69%` | `51.6%` | `+101.99%` | `9.69%` | `46.2%` | `ML` 明显更强 |
| `2022-01-01` | `+221.67%` | `28.68%` | `55.0%` | `+84.73%` | `9.90%` | `44.9%` | `ML` 明显更强 |
| `2023-01-01` | `+153.49%` | `25.54%` | `56.5%` | `+74.11%` | `9.47%` | `47.2%` | `ML` 明显更强 |
| `2024-01-01` | `+183.73%` | `27.23%` | `60.6%` | `+33.80%` | `9.17%` | `46.5%` | `ML` 显著碾压 |
| `2025-01-01` | `+98.09%` | `24.63%` | `59.1%` | `+30.49%` | `9.27%` | `45.2%` | `ML` 显著更强 |
| `2026-01-01` | `+1.74%` | `25.81%` | `58.3%` | `+22.12%` | `4.87%` | `100.0%` | 规则版短窗反超 |

### 当前归纳

- `2021-06-21` 到 `2025-01-01` 这五组样本里，`ML` 的优势非常稳定:
  - 收益显著更高
  - 胜率持续更高
  - 交易笔数并没有系统性放大到不可比
- 规则版唯一稳定优势仍然是低回撤，`MDD` 基本稳定在 `9%` 左右，因此更适合作为保守基线，而不是默认主线。
- `2026-01-01` 这组只有 `56` 个交易日，且 `ML=12` 笔、规则版 `=6` 笔，统计稳定性明显弱于前五组；它更像是一个**regime 告警样本**，而不是对前述主线结论的推翻。

### 当前最终判断

- `ML B1` 仍是主线，当前不需要再反复证明它是否优于规则版。
- 规则版 `B1` 当前应保留为:
  - 低回撤对照基线
  - 近端 market regime 变化时的快速 sanity check
- 下一步优化重点应从“继续比较谁更强”转为:
  1. 压低 `ML` 的回撤，尤其是 `24% ~ 29%` 这段区间。
  2. 提高 `ML` 对近端 regime 的适应能力，避免出现“收益没起来、回撤先放大”。
  3. 继续保留规则链独立运行，作为 `B1` 边界定义与回测基准。

## 当前工作流

### 规则链

- 规则链已经有自己的完整导出与 Rust 回测闭环。
- 规则链的目标是定义 `B1` 边界，而不是在 ML notebook 里做旁路验证。
- 当前主入口仍是:
  - `utils/b1_factors_opt.py::calc_b1_factors_wmacd`
  - `utils/signal_export.py::export_for_rust`
  - `backtest-engine/crates/b1`

### ML 链

#### 1. Factor Lab

- 主入口: `notebooks/b1_condition_mining.py`
- 当前只负责统计研究，不再承担规则收敛主线。
- 当前固定产出:
  - `seed_loose / seed_mid / seed_strict` 样本概览
  - 因子 `IC / ICIR / t-stat`
  - 特征分组汇总
  - 单变量分箱得分榜
  - 多周期衰减 (`1d / 2d / 3d / 5d`)
  - 相关性与冗余诊断
  - watchlist / 下一版冻结特征集候选

#### 2. Train / Export Entry

- 主入口: `notebooks/b1_seed_ml_baseline.py`
- 当前只负责训练、评估、导出，不再承担特征探索叙事。
- 当前固定流程:
  - 读取统一研究底表
  - 消费冻结特征集
  - `LightGBM walk-forward`
  - 输出 `IC / ICIR / q4-q0`
  - 导出 Rust parquet
- 当前已支持:
  - 将 `seed_col` 直接映射为 parquet 内的 `b1_signal`
  - 在相同 `bt-b1` 回测流程下，用不同 parquet 对照“规则版 B1”与“seed + score 版 B1”
  - 训练后直接落盘 `artifacts/b1/...`，并通过 `scripts/b1_backtest.py` 选择 signal 回测

#### 3. Shared Feature Base

- 共享入口: `utils/b1_feature_pool.py`
- 当前统一提供:
  - `core / candidate / selected` 三档特征集
  - 第一批 + 第二批 B1 连续特征池
  - `fwd_ret_1d / 2d / 3d / 5d`
  - `fwd_mfe_10d / fwd_mae_10d`
  - `seed pool` 与手工 bull regime 标注

## 当前研究口径

### 1. seed 选择

- `bull_only` 下，`seed_strict` 质量只比 `seed_mid` 略高一点点。
- 当前更合理的分工是:
  - `seed_mid` 更宽，适合作为默认 lab / train 宇宙
  - `seed_strict` 更干净，适合作为保守对照

### 2. 当前最值得关注的方向

- `Bias_WL_YL`
- `range_pct`
- `rw_dif_pct`
- `Bias_C_YL`

当前主线不再只看“均线关系 + 周动能”，而是同时关注:

- 趋势强度
- 波动展开
- 周月动能
- 价格相对黄线位置
- 量价结构

### 3. 冻结训练特征集

- 当前训练 notebook 默认消费 `selected` 特征集。
- 当前目的不是每次 lab 跑完都动态改训练列，而是:
  - 先用 lab 证明一批因子稳定有效
  - 再手工更新 `selected`
  - 之后由训练 notebook 稳定消费

## 当前推荐顺序

1. 先在 `b1_condition_mining.py` 里看 `seed_mid` 的 IC、分组、分箱、衰减和相关性诊断。
2. 如果 lab 结论稳定，再决定是否更新 `utils/b1_feature_pool.py` 里的 `selected` 冻结特征集。
3. 再用 `b1_seed_ml_baseline.py` 跑 `seed_mid` 训练 / 评估 / 导出。
4. 最后用 `seed_strict` 做保守对照，判断更窄宇宙是否能换来更稳的排序结果。

## 当前不做的事

- 暂不自动化 `活跃市值`
- 暂不把规则链和 ML 链重新揉成一个 notebook
- 暂不把主线直接切成旧意义上的“B1 专属窄截面纯模型”
- 暂不在 lab notebook 内做大规模 rule sweep
- 暂不引入更重的 manifest 体系

## 当前推荐方向

- 规则链继续独立负责 `B1` 边界定义。
- ML 链默认走 `Lab -> Train -> Export`，并继续作为 `B1` 主线。
- 规则链继续保留为低回撤基线与 regime sanity check。
- 当前优先级先放在 `ML` 的回撤控制与近端适应性，而不是继续扩写新分支。
- `Agent` 只做辅助解释与审查，不作为唯一决策层。
