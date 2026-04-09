# B1 Next Phase

## 当前结论

- `B1` 当前已经不适合再拆成多份实验文档维护，后续统一以本文档作为主结论文档。
- `活跃市值` 继续保留为**人工 regime 判断**，当前不做自动化复刻。
- `B1` 主链已经稳定在:
  - `utils/b1_factors_opt.py::calc_b1_factors_wmacd`
  - `notebooks/b1_condition_mining.py`
  - `notebooks/b1_seed_ml_baseline.py`
  - `utils/signal_export.py::export_for_rust`
  - `backtest-engine/crates/b1`
- 历史结论仍成立:
  - 全市场 ML 排序有价值
  - 旧 `B1 dedicated ML` 因最终候选日截面过窄而不该作为默认主线

## 当前研究框架

### 路线 A：条件挖掘

- 在 `seed_loose / seed_mid / seed_strict` 上继续做可解释条件挖掘
- 当前目标不是“发现完整 B1 公式”，而是收敛出 `2~5` 条真正有增量的条件
- 当前 notebook:
  - `notebooks/b1_condition_mining.py`

### 路线 B：seed 内纯模型

- 在同一批扩充后的 B1 连续特征上，直接训练:
  - `seed_mid + 纯模型`
  - `seed_strict + 纯模型`
- 当前 notebook:
  - `notebooks/b1_seed_ml_baseline.py`

## 当前实现状态

- 共享特征底表已统一到 `utils/b1_feature_pool.py`
- 条件挖掘线已具备:
  - 三档 `seed pool`
  - 第一批 + 第二批 B1 连续特征池
  - `Step 5 / 6 / 7 / 7b / 8`
  - 自动收敛 `2~5` 条候选条件
- 纯模型线已具备:
  - `seed_mid / seed_strict` 切换
  - `LightGBM walk-forward`
  - `IC / ICIR / q4-q0`
  - Rust parquet 导出

## 最新研究结论

### 1. seed 选择

- `bull_only` 下:
  - `seed_mid`: `rows=72118`, `avg/day=157.81`, `mfe10_mean=0.0898`, `mfe_hit_rate=39.20%`
  - `seed_strict`: `rows=60016`, `avg/day=131.33`, `mfe10_mean=0.0899`, `mfe_hit_rate=39.21%`
- 结论:
  - `seed_strict` 质量只比 `seed_mid` 略高一点点
  - 当前不能再把 `seed_strict` 视作绝对主战场
  - 更合理的理解是:
    - `seed_mid` 更宽，适合模型与条件搜索
    - `seed_strict` 更干净，适合做更保守的对照

### 2. 当前最强特征

- 单变量分箱当前最强的 4 个方向:
  - `Bias_WL_YL`
  - `range_pct`
  - `rw_dif_pct`
  - `Bias_C_YL`
- 其中新增最有价值的信息是:
  - `range_pct` 已进入第一梯队
  - 且单调性非常强 (`monotonicity=0.9947`)
- 当前主线不再只是“均线关系 + 周动能”，而是:
  - 趋势强度
  - 波动展开
  - 周动能
  - 价格相对黄线位置

### 3. 当前候选规则

- 浅树当前第一候选:
  - `range_pct > 3.6248 & Bias_WL_YL > 8.0843`
  - `samples=13936`
  - `positive_rate=57.32%`
  - `lift_vs_base=+0.1812`
- 手工规则当前第一候选:
  - `Bias_WL_YL > 9 & rw_dif_pct > 10 & Bias_C_YL > 7.65`
  - `samples=5553`
  - `positive_rate=59.59%`
  - `hit_lift=+0.2039`
  - `mfe10_lift=+0.0472`
- 当前已经不是“继续盲挖”，而是已形成可直接进入下一轮验证的候选条件清单

## 最小对照集

当前只保留 4 条路线，避免同时比较过多变量:

| Route | 宇宙 | 主要入口 | 当前作用 |
| --- | --- | --- | --- |
| 规则基线 | `seed_mid` 或 `seed_strict` | `b1_condition_mining.py` Step 7b | 保留最简单手工基线 |
| 条件增强 | `seed_mid` 或 `seed_strict` | `b1_condition_mining.py` Step 8 | 验证候选条件是否能升级为增强版规则 |
| `seed_mid + 纯模型` | `seed_mid` | `b1_seed_ml_baseline.py` | 验证更宽宇宙上的排序能力 |
| `seed_strict + 纯模型` | `seed_strict` | `b1_seed_ml_baseline.py` | 验证更干净宇宙上的排序能力 |

## 当前推荐顺序

1. 先继续以 `b1_condition_mining.py` 收敛和验证条件增强版规则。
2. 纯模型先跑 `seed_mid`，因为它更宽，更适合作为模型主对照。
3. 再跑 `seed_strict`，看更窄但更干净的宇宙是否能换来更稳定的排序结果。
4. 最后再决定主线更偏:
   - 条件增强
   - 还是 `seed` 内纯模型排序

## 当前不做的事

- 暂不自动化 `活跃市值`
- 暂不把 `B1` 重新退回完全手工看图工作流
- 暂不把主线直接切成旧意义上的“B1 专属窄截面纯模型”
- 暂不升格 `manifest`
- 暂不做大规模 rule sweep

## 当前推荐方向

- 当前默认优先考虑:
  - `规则增强 + seed 内纯模型并行对照`
- 若后续要正式切回 B1 主线，优先顺序仍是:
  - 规则负责定义 B1 边界
  - 模型负责在边界内排序
  - `Agent` 只做辅助解释和审查，不做唯一决策层
