# TODO

## 马上要做

- [ ] 提交今日改动（AKShare 移除 + 申万分类切换 + 基线变更）
- [ ] 同步到 Mac，在 Mac 上重跑所有 canonical 回测确认两设备一致性

## 近期

- [ ] sector-tailwind 针对申万分类重新调参（当前 penalty=0.02 linear 未生效）
- [ ] pullback-pb3 raw-execution allocation 分析（当前 +79.3% / 11.8%，与 trend 日相关 0.26）
- [ ] event-firstboard MaxDD 改善（当前 34.1%，base 5td 无 weakgate 为 45.4%）
- [ ] 补 `qlab results --diff` 的年度归因拆解

## 已确认但暂缓

- [ ] P3 早期止损（d2 cum_ret < -3%，what-if +166K，需改 Rust 退出逻辑）
- [ ] 延长持有（what-if 不成立，信噪比太低）
- [ ] 板块宽度 gating（P3 在窄基反而更强，二元过滤是错的）
- [ ] 动量策略探索（涨停污染未解决，P-block 已是折中）

## 搁置

- [ ] `qlab explore` 子命令（因子扫描、规则诊断、网格探索）
- [ ] `qlab backtest --top-n` / `--max-hold` 的临时 TOML 合成（基础框架已写好，未测）
- [ ] ETF 相关文件（strategies/etf_momentum_rotation.py, notebooks/etf_momentum_rotation.py）仍引用 akshare，后续处理
