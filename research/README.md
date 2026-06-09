# Research — 因子探索沙盒

人类与 Agent 进行因子研究的工作目录。与生产代码 (`strategies/`, `scripts/qlab.py`) 完全隔离。

## 工作流

1. 打开 `explore_factor.py`，修改 `FACTOR_TAG` 和 `make_factor_expr()`
2. 运行 `uv run python research/explore_factor.py`
3. 查看 Rank IC、IC IR、分组收益
4. 结果自动记录到 `factor_ledger.jsonl`

## 文件

- `explore_factor.py` — 因子快速验证脚本 (改公式 → 跑 → 看 IC)
- `factor_ledger.jsonl` — 实验账本 (自动追加，每行一条 JSON)

## 查看历史

```bash
cat research/factor_ledger.jsonl | python -m json.tool
```

或用 Polars:

```python
pl.read_ndjson("research/factor_ledger.jsonl")
```
