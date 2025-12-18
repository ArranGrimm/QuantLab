# QuantLab 量化研究实验室

这是一个用于量化交易策略研究、回测和数据分析的个人工作空间。

## 📁 目录结构

项目采用了模块化的结构设计，以便于管理数据、策略代码和研究文档。

```text
QuantLab/
├── strategies/         # 策略核心逻辑与实现代码 (.py)
├── notebooks/          # 探索性数据分析与策略原型 (.ipynb)
├── data/               # 市场数据存放目录 (被 git 忽略)
│   ├── raw/            # 原始数据
│   └── processed/      # 清洗后的数据
├── scripts/            # 实用工具脚本 (爬虫、数据处理等)
├── docs/               # 项目文档与知识库
└── results/            # 回测日志、图表与统计报告 (被 git 忽略)
```

## 🚀 快速开始

1. **环境配置**
   确保安装了必要的依赖库（建议使用 `uv` 或 `pip`）：
   ```bash
   pip install -r requirements.txt
   # 或者如果使用 uv
   uv sync
   ```

2. **数据准备**
   将所需的 CSV/Parquet 数据文件放入 `data/` 目录。

3. **运行策略**
   - 研究环境：在 `notebooks/` 中打开 `.ipynb` 文件进行交互式分析。
   - 实盘/回测：运行 `strategies/` 下的 Python 脚本。

## 📝 注意事项

- **数据隐私**：`data/` 目录下的所有数据文件已被 `.gitignore` 忽略，请勿提交敏感数据或大文件。
- **路径引用**：在 `notebooks/` 中引用 `strategies/` 模块时，可能需要添加父目录到系统路径：
  ```python
  import sys
  import os
  sys.path.append(os.path.abspath(".."))
  ```

