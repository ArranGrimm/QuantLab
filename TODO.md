# TODO

## 马上要做

- [ ] 跨设备 baseline 对齐：Mac TDX vs Windows TDX 数据源差异确认

## 近期

- [ ] sector-tailwind 针对申万分类重新调参
- [ ] pullback-pb3 raw-execution allocation 分析（当前 +42.5% / 15.0%，与 trend 日相关 0.26）
- [ ] event-firstboard MaxDD 改善（当前 36.1%）
- [ ] 补 `qlab results --diff` 的年度归因拆解
- [ ] 探索循环脚本：grid → Rust → 收集结果 → 结构化反馈
- [x] Terrified Score 接入策略验证：36 组合权重扫描全部跑输 trend-p3 baseline（best +84% vs +171%）

## 探索中/待跟进

- [ ] **因子探索**（research/explore_factor.py）：已验证 6 个实验因子
- [ ] QuantsPlaybook Tier-1 待测：隔夜-日间网络因子（O(N²) 跳过）、上下影线单独因子（蜡烛上_mean IC -0.031、威廉下_mean IC -0.046）
- [ ] ETF 动量轮动：原型 +362% 但回撤 21%、参数敏感、未经 Rust 验证
- [ ] 上证交叉验证 AMV 牛市真伪（数据已就绪，未分析）
- [ ] regime 慢退出机制（逻辑成立但样本量小，待更多数据）

## 择时 gate 已验证结论（2026-06-10）

- [x] AMV 活跃市值是最好的全市场择时 gate（牛市随机买 20 日月均 +2.3%，t=+2.02）
- [x] RSRS、Breadth、CSVC 均不如 AMV 适用于全市场选股场景
- [x] AMV 在趋势年（2022/2024）极其有效，震荡年（2023）反效
- [x] P3 Top3 选股可以与 AMV 择时互补（2023 择时亏 -1.5% 但选股赚 +16.6%）

## 已完成

- [x] 架构重构：Hook 系统 + pipeline 无 if 分支 + projection pushdown
- [x] factors/ 顶层共享模块 + research/ 探索管线
- [x] QuantsPlaybook 审计：Tier-1 10 个候选，6 个已测
- [x] Mac 全量 canonical 回测重跑
- [x] TDX 数据源 Mac 可用性验证 + 北交所过滤
- [x] AGENTS.md 精简（228→76 行）
- [x] trend-p3 升格为 baseline（medium 在 Mac 上不敌纯 P3）
- [x] explore_regime.py：MC 100 次 × 4 个持有期 × 4 个 gate 扫描

## 已确认但暂缓

- [ ] P3 早期止损（ATR 模式已实现在 Rust，默认关闭）
- [ ] 延长持有（what-if 不成立，信噪比太低）
- [ ] 板块宽度 gating（P3 在窄基反而更强，二元过滤是错的）

## 搁置

- [ ] `qlab explore` 子命令
- [ ] `qlab backtest --top-n` / `--max-hold` 的临时 TOML 合成
