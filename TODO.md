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

## 探索中/待跟进

- [ ] ETF 动量轮动：原型 +362% 但回撤 21%、参数敏感、未经 Rust 验证
- [ ] 上证交叉验证 AMV 牛市真伪（数据已就绪，未分析）
- [ ] regime 慢退出机制（逻辑成立但样本量小，待更多数据）

## 已完成

- [x] factor registry 按需计算：compute_required_factors 替代全量 calc_amv_core_factors（trend 只算 4 因子）
- [x] Rule Hook 系统：ranker + rules 完全解耦，JSON 配置驱动，pipeline 零 if 分支
- [x] post-collect 中间副本优化：penalty 合并为单次 with_columns，零中间副本
- [x] medium_trend_quality.py 内联进 MediumTrendQualityHook
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
