# TODO

## 马上要做

- [ ] 同步到 Mac，在 Mac 上重跑所有 canonical 回测确认两设备一致性
- [ ] 验证 TDX 数据源在 Mac 上的可用性

## 近期

- [ ] sector-tailwind 针对申万分类重新调参（当前 penalty=0.02 linear 未生效）
- [ ] pullback-pb3 raw-execution allocation 分析（当前 +79.3% / 11.8%，与 trend 日相关 0.26）
- [ ] event-firstboard MaxDD 改善（当前 34.1%，base 5td 无 weakgate 为 45.4%）
- [ ] 补 `qlab results --diff` 的年度归因拆解

## 探索中/待跟进

- [ ] ETF 动量轮动：原型 +362% 但回撤 21%、参数敏感、未经 Rust 验证。待 上证 MA20 叠加 + 回撤改善后重评估
- [ ] 上证交叉验证 AMV 牛市真伪（数据已就绪，未分析）
- [ ] regime 慢退出机制（逻辑成立但样本量小，待更多数据）
- [ ] TDX 数据源：已验证兼容，待更多测试后考虑切换为默认

## 已确认但暂缓

- [ ] P3 早期止损（ATR 模式已实现在 Rust，默认关闭）
- [ ] 延长持有（what-if 不成立，信噪比太低）
- [ ] 板块宽度 gating（P3 在窄基反而更强，二元过滤是错的）

## 搁置

- [ ] `qlab explore` 子命令
- [ ] `qlab backtest --top-n` / `--max-hold` 的临时 TOML 合成
- [ ] ETF 相关文件（strategies/etf_momentum_rotation.py 仍引用 akshare，后续处理）
