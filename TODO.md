# TODO

## 马上要做

- [ ] 跨设备 baseline 对齐：Mac TDX vs Windows TDX 数据源差异确认（trend-p3-medium Mac +165.6% vs Windows +178.6%）
- [ ] event pipeline 内存继续优化（10GB → 8GB，仍有空间）

## 近期

- [ ] sector-tailwind 针对申万分类重新调参（当前 penalty=0.02 linear 未生效）
- [ ] pullback-pb3 raw-execution allocation 分析（当前 +44.3% / 15.0%，与 trend 日相关 0.26）
- [ ] event-firstboard MaxDD 改善（当前 36.1%，base 为 42.3%）
- [ ] 补 `qlab results --diff` 的年度归因拆解
- [ ] 探索循环脚本：grid → Rust → 收集结果 → 结构化反馈
- [ ] **抽取 top-level factors/**: 从 strategies/amv/factors/ 提升为顶级模块，strategy 和 research 共享

## 探索中/待跟进

- [ ] **因子探索**（research/explore_factor.py）：已验证 Terrified Score（IC -0.085 IR 0.64 ⭐）、高质量动量（IC 0.064 IR 0.33）、MA 收敛 PCF（IC ~0.05）、STV（IC -0.067 IR 0.40）、CGO（IC -0.052 IR 0.43）
- [ ] QuantsPlaybook Tier-1 待测：球队硬币、RSRS gate、CSVC 熊牛指标、扩散指标 Breadth
- [ ] ETF 动量轮动：原型 +362% 但回撤 21%、参数敏感、未经 Rust 验证
- [ ] 上证交叉验证 AMV 牛市真伪（数据已就绪，未分析）
- [ ] regime 慢退出机制（逻辑成立但样本量小，待更多数据）

## 已完成

- [x] 架构重构：22 → ~10 文件，pipeline.py 统一入口，Hook 系统解耦
- [x] research/ 模块独立：explore_factor.py + factor_ledger.jsonl 因子实验账本
- [x] QuantsPlaybook 审计：6-agent 并行扫描，Tier-1 10 个候选，report 写入
- [x] 架构全貌文档：reports/architecture.md
- [x] pdf skill: pdfplumber 可读取研报 PDF
- [x] Mac 全量 canonical 回测重跑
- [x] TDX 数据源 Mac 可用性验证 + 北交所过滤
- [x] AGENTS.md 精简（228→76 行）

## 已确认但暂缓

- [ ] P3 早期止损（ATR 模式已实现在 Rust，默认关闭）
- [ ] 延长持有（what-if 不成立，信噪比太低）
- [ ] 板块宽度 gating（P3 在窄基反而更强，二元过滤是错的）

## 搁置

- [ ] `qlab explore` 子命令
- [ ] `qlab backtest --top-n` / `--max-hold` 的临时 TOML 合成
