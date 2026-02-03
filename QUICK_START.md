# 快速开始 - 运行回测

## 前置要求

1. **Python 环境**：确保已安装 Python 3.8 或更高版本
2. **依赖包**：确保已安装所有必需的 Python 包

## 运行步骤

### 步骤 1: 打开终端/命令行

在 Windows 上：
- 按 `Win + R`，输入 `cmd` 或 `powershell`，按回车
- 或者直接在项目文件夹中，按住 `Shift` 键，右键点击空白处，选择"在此处打开 PowerShell 窗口"

### 步骤 2: 进入项目目录

```bash
cd D:\QUANT\multi_factor
```

### 步骤 3: (可选) 安装/更新依赖

如果是第一次运行，或者依赖包有更新：

```bash
pip install -r requirements.txt
```

### 步骤 4: 运行回测

```bash
python scripts/run_backtest.py
```

## 运行过程

回测脚本会依次执行以下步骤：

1. **加载数据** - 从 CSV 文件加载股票列表，下载价格数据
2. **数据预处理** - 清洗和过滤数据
3. **计算因子** - 计算 value 和 momentum 因子
4. **组合因子** - 计算 composite scores
5. **构建组合** - 根据 scores 选择股票并分配权重
6. **运行回测** - 模拟交易，计算收益
7. **计算指标** - 计算 Sharpe 比率、最大回撤等
8. **生成报告** - 保存结果和图表

整个过程可能需要几分钟到十几分钟，取决于：
- 网络速度（首次下载数据时）
- 数据量大小
- 计算机性能

## 查看结果

运行完成后，结果会保存在以下位置：

- **回测数据**: `results/backtest_results.parquet`
- **性能指标**: `results/metrics.json`
- **图表**: `reports/figures/`

### 查看性能指标

```bash
# 使用 Python 查看
python -c "import json; print(json.dumps(json.load(open('results/metrics.json')), indent=2))"
```

### 查看回测数据

```bash
# 使用 Python 查看（需要 pandas）
python -c "import pandas as pd; df = pd.read_parquet('results/backtest_results.parquet'); print(df.head(20)); print('\n...\n'); print(df.tail(20))"
```

## 常见问题

### 1. "ModuleNotFoundError" 或 "ImportError"

**解决方法**：
```bash
pip install -r requirements.txt
```

### 2. 数据下载失败

**可能原因**：
- 网络连接问题
- yfinance API 临时不可用

**解决方法**：
- 检查网络连接
- 稍后重试
- 如果之前下载过数据，会使用缓存，不需要重新下载

### 3. 想要重新开始（清理缓存）

如果需要重新下载数据或清理之前的结果：

```bash
python scripts/clean_cache.py
```

然后重新运行回测。

### 4. 想要修改回测参数

编辑 `src/utils/config.py` 文件中的 `BacktestConfig` 类，修改：
- `start_date` / `end_date` - 回测日期范围
- `rebalance_frequency` - 调仓频率（monthly/quarterly）
- `long_pct` / `short_pct` - 多空比例
- `initial_capital` - 初始资金

## 完整示例

```bash
# 1. 进入项目目录
cd D:\QUANT\multi_factor

# 2. 安装依赖（首次运行）
pip install -r requirements.txt

# 3. 运行回测
python scripts/run_backtest.py

# 4. 查看结果
python -c "import json; metrics = json.load(open('results/metrics.json')); print(f\"Final NAV: ${metrics['final_nav']:,.2f}\"); print(f\"Total Return: {metrics['total_return']*100:.2f}%\")"
```

## 预期输出

成功运行后，你应该看到类似以下的输出：

```
==============================================================
Step 1: Loading data
==============================================================
...
Step 10: Generating plots and reports
==============================================================

BACKTEST COMPLETE
==============================================================

Final NAV: $81,178,594.02
```如果遇到任何问题，请检查日志文件 `logs/` 目录中的日志信息。
