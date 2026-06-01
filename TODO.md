# TODO

## 马上要做

- [ ] 从 Windows 复制 `data/sector_map_em.csv`，或运行 `uv run python -c "from strategies.amv.factors.sector_tailwind import refresh_em_sector_map; from pathlib import Path; refresh_em_sector_map(Path('data/sector_map_em.csv'), request_sleep=0.5)"` 生成
- [ ] `qlab run trend-p3-enhanced` 验证最复杂策略端到端通过
- [ ] 同步 DuckDB 数据库，使两设备 raw execution 数据一致

## 近期

- [ ] 补 `qlab results --diff` 的年度归因（目前只做总结 delta，未拆解到 yearly 逐项 diff）
- [ ] 跑一遍所有已知策略的 canonical 回测，填充 `artifacts/` 使 `qlab status` 展示完整

## 搁置

- [ ] `qlab explore` 子命令（因子扫描、规则诊断、网格探索）
- [ ] `qlab backtest --top-n` / `--max-hold` 的临时 TOML 合成（目前基础框架已写好，未测）
