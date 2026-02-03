# 重新开始运行项目指南

## 快速重启步骤

### 1. 清理缓存数据（推荐）

清理之前下载和处理的缓存数据，确保重新下载最新数据：

```bash
# 交互式清理（会询问确认）
python scripts/clean_cache.py

# 或者直接清理（不询问）
python scripts/clean_cache.py --yes
```

这会删除：
- ✅ 缓存的价格数据 (`data/raw/*.parquet`)
- ✅ 缓存的基本面数据 (`data/raw/*.parquet`)
- ✅ 处理过的数据 (`data/processed/*`)
- ✅ 之前的结果 (`results/*`)
- ✅ 之前的图表 (`reports/figures/*`)
- ✅ 日志文件 (`logs/*.log`)

### 2. 检查依赖

确保所有依赖都已安装：

```bash
pip install -r requirements.txt
```

**重要**: 确保已安装 `pyarrow`（用于 parquet 文件支持）：
```bash
pip install pyarrow>=10.0.0
```

### 3. 运行回测

```bash
python scripts/run_backtest.py
```

## 详细步骤

### 选项 A: 完全清理并重新开始（推荐）

如果你想确保一切都是全新的：

```bash
# 1. 清理所有缓存
python scripts/clean_cache.py --yes

# 2. 运行回测（会自动重新下载数据）
python scripts/run_backtest.py
```

### 选项 B: 保留缓存数据

如果你想保留之前下载的数据（更快，但可能使用旧数据）：

```bash
# 只清理结果，保留原始数据
# 手动删除 results/ 和 reports/figures/ 目录中的文件

# 然后运行
python scripts/run_backtest.py
```

### 选项 C: 只清理特定类型的数据

如果你想手动清理：

```bash
# Windows PowerShell
Remove-Item data\raw\*.parquet
Remove-Item data\processed\*
Remove-Item results\*
Remove-Item reports\figures\*

# Windows CMD
del data\raw\*.parquet
del data\processed\*
del results\*
del reports\figures\*

# Linux/Mac
rm data/raw/*.parquet
rm data/processed/*
rm results/*
rm reports/figures/*
```

## 常见问题

### Q: 为什么要清理缓存？

A: 清理缓存可以：
- 确保使用最新的代码逻辑下载数据
- 修复之前下载数据时的问题（如缺少 'Adj Close' 列）
- 重新处理数据以应用最新的预处理逻辑

### Q: 清理后需要重新下载数据吗？

A: 是的。清理缓存后，运行 `run_backtest.py` 时会自动重新下载数据。首次下载可能需要几分钟时间。

### Q: 数据下载失败怎么办？

A: 
1. 检查网络连接
2. 检查 yfinance 是否正常工作：`pip install --upgrade yfinance`
3. 查看日志文件：`logs/multifactor.log`

### Q: 如何只清理结果而不重新下载数据？

A: 只删除 `results/` 和 `reports/figures/` 目录中的文件，保留 `data/raw/` 中的 parquet 文件。

## 验证安装

运行前可以验证环境：

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查关键依赖
python -c "import pandas, numpy, yfinance, pyarrow; print('✓ All dependencies OK')"
```

## 下一步

清理完成后，运行：

```bash
python scripts/run_backtest.py
```

这将：
1. ✅ 重新下载价格和基本面数据（如果缓存已清理）
2. ✅ 预处理和过滤数据
3. ✅ 计算因子
4. ✅ 构建投资组合
5. ✅ 运行回测
6. ✅ 生成结果和图表

结果将保存在：
- `results/backtest_results.parquet` - 完整回测时间序列
- `results/metrics.json` - 性能指标
- `results/performance_report.txt` - 格式化报告
- `reports/figures/*.png` - 可视化图表
