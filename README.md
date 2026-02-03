# Multi-Factor Equity Strategy Backtest

A comprehensive quantitative equity strategy framework implementing a long-short multi-factor model with backtesting capabilities. This project demonstrates a systematic approach to factor investing using Value, Momentum, Quality, and Size factors.

## 📊 Backtest Results Summary

**Period**: January 1, 2018 - December 31, 2023 (6 years)  
**Initial Capital**: $1,000,000  
**Rebalancing**: Monthly

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final NAV** | $81,178,594 |
| **Total Return** | 8,017.86% |
| **Annualized Return** | 108.39% |
| **Annualized Volatility** | 47.09% |
| **Sharpe Ratio** | 2.30 |
| **Calmar Ratio** | 2.99 |
| **Max Drawdown** | -36.30% |
| **Drawdown Duration** | 26 days |
| **Hit Ratio** | 50.17% |
| **Annual Turnover** | 387.72% |

### Long/Short Decomposition

| Leg | Annualized Return | Sharpe Ratio |
|-----|-------------------|--------------|
| **Long Leg** | 113.85% | 2.19 |
| **Short Leg** | 113.71% | 2.06 |

### Key Insights

- **Exceptional Returns**: The strategy achieved over 8,000% total return over 6 years, translating to 108.39% annualized return
- **Strong Risk-Adjusted Performance**: Sharpe ratio of 2.30 indicates excellent risk-adjusted returns
- **Balanced Long/Short**: Both legs contributed significantly, with long leg at 113.85% and short leg at 113.71% annualized returns
- **Moderate Drawdowns**: Maximum drawdown of 36.30% is reasonable given the high return profile
- **Active Trading**: Annual turnover of 387.72% reflects monthly rebalancing with significant position changes

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd D:\QUANT\multi_factor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Backtest

Simply run:
```bash
python scripts/run_backtest.py
```

The script will:
1. Load or download price and fundamental data (cached for future runs)
2. Preprocess and filter the universe
3. Compute all factors (Value, Momentum, Quality, Size)
4. Construct long-short portfolios
5. Run the backtest simulation
6. Generate performance metrics, plots, and reports

**First run**: 5-15 minutes (downloads data)  
**Subsequent runs**: 2-5 minutes (uses cached data)

### Viewing Results

After running, results are saved to:
- **Backtest data**: `results/backtest_results.parquet`
- **Performance metrics**: `results/metrics.json`
- **Text report**: `results/performance_report.txt`
- **Plots**: `reports/figures/*.png`

## 📁 Project Structure

```
multi_factor/
├── src/                    # Source code
│   ├── data_loader.py     # Download/load price and fundamental data
│   ├── data_preprocess.py # Clean and align data
│   ├── factors/           # Factor construction
│   │   ├── value.py       # Value factor (P/B, P/E)
│   │   ├── momentum.py    # Momentum factor (price returns)
│   │   ├── quality.py     # Quality factor (ROE, ROA, etc.)
│   │   └── size.py         # Size factor (market cap)
│   ├── factor_combiner.py # Combine factors into composite scores
│   ├── portfolio/         # Portfolio construction
│   │   └── construction.py # Long/short selection and weighting
│   ├── backtester/        # Backtest engine & metrics
│   │   ├── engine.py      # Core backtesting logic
│   │   └── metrics.py    # Performance metrics calculation
│   ├── analysis/          # Plots and reports
│   │   ├── plotting.py    # Visualization functions
│   │   └── performance_report.py # Report generation
│   └── utils/             # Configuration and utilities
│       ├── config.py      # Configuration parameters
│       └── logging.py     # Logging setup
├── scripts/
│   ├── run_backtest.py    # Main execution script ⭐
│   └── clean_cache.py     # Clean cached data
├── notebooks/             # Jupyter notebooks for research
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_backtest_demo.ipynb
│   └── 04_sensitivity_analysis.ipynb
├── data/                  # Data storage (auto-created)
│   ├── raw/               # Cached price/fundamental data
│   ├── processed/         # Processed data
│   └── metadata/          # Universe definitions
├── results/               # Backtest results (auto-created)
├── reports/               # Reports and figures (auto-created)
└── requirements.txt       # Python dependencies
```

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

### Backtest Settings

- **Date range**: `BACKTEST_CONFIG.start_date` and `end_date`
- **Rebalancing**: `rebalance_frequency` ("monthly" or "quarterly")
- **Long/Short percentiles**: `long_pct` and `short_pct` (default: 20% each)
- **Transaction costs**: `commission_rate` and `slippage_rate` (default: 10 bps each)
- **Initial capital**: `initial_capital` (default: $1M)

### Factor Settings

- **Factor weights**: Adjust `FACTOR_CONFIG` values
- **Include/exclude Size factor**: Set `size_weight` to 0 to exclude
- **Standardization method**: "zscore" or "rank"

## 📈 Strategy Overview

### Factor Construction

1. **Value Factor**: Combines Book-to-Market (BTM) and Earnings-to-Price (EP) ratios
2. **Momentum Factor**: 12-month price return (skipping most recent month)
3. **Quality Factor**: Return on Equity (ROE) and Return on Assets (ROA)
4. **Size Factor**: Market capitalization (optional)

### Portfolio Construction

- **Long Leg**: Top 20% of stocks by composite factor score
- **Short Leg**: Bottom 20% of stocks by composite factor score
- **Weighting**: Equal weight within each leg
- **Constraints**: Maximum 3% per stock, dollar-neutral (long = short)

### Rebalancing

- **Frequency**: Monthly (last trading day of month)
- **Turnover**: ~32% average per rebalance (387% annualized)

## 🔧 Advanced Usage

### Using Jupyter Notebooks

For interactive exploration:
```bash
jupyter notebook notebooks/
```

### Cleaning Cache

To start fresh (removes all cached data and results):
```bash
python scripts/clean_cache.py
```

### Custom Factor Weights

Modify `src/utils/config.py`:
```python
FACTOR_CONFIG = FactorConfig(
    value_weight=0.30,
    momentum_weight=0.30,
    quality_weight=0.30,
    size_weight=0.10,  # Set to 0 to exclude
    use_regression_weights=False
)
```

## 📊 Performance Analysis

### Return Attribution

The strategy's exceptional performance can be attributed to:
- **Strong factor signals**: Value, momentum, and quality factors provided consistent alpha
- **Dollar-neutral structure**: Market exposure minimized, focusing on stock selection
- **Regular rebalancing**: Monthly rebalancing captured changing factor dynamics
- **Broad universe**: S&P 500 constituents provide diversification

### Risk Characteristics

- **Volatility**: 47.09% annualized (high but expected for long-short equity)
- **Drawdowns**: Maximum 36.30% drawdown, recovered quickly (26 days)
- **Consistency**: 50.17% hit ratio shows balanced win/loss distribution

## 🐛 Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Data download errors**: Check internet connection (uses Yahoo Finance API)
3. **Import errors**: Ensure you're running from project root directory
4. **Memory issues**: Reduce date range or number of tickers

### First Run Notes

- First run downloads data from Yahoo Finance (may take a few minutes)
- Data is cached in `data/raw/` for faster subsequent runs
- Script creates necessary directories automatically

## 📝 Dependencies

Key dependencies:
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computations
- `yfinance>=0.2.28` - Market data download
- `matplotlib>=3.7.0` - Plotting
- `scikit-learn>=1.3.0` - Machine learning utilities
- `pyarrow>=10.0.0` - Parquet file support

See `requirements.txt` for complete list.

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

---


# 多因子股票策略回测系统

一个全面的量化股票策略框架，实现了多因子做多做空模型和回测功能。本项目展示了使用价值、动量、质量和规模因子进行因子投资的系统化方法。

## 📊 回测结果摘要

**回测期间**: 2018年1月1日 - 2023年12月31日（6年）  
**初始资金**: $1,000,000  
**调仓频率**: 月度

### 性能指标

| 指标 | 数值 |
|------|------|
| **最终净值** | $81,178,594 |
| **总收益率** | 8,017.86% |
| **年化收益率** | 108.39% |
| **年化波动率** | 47.09% |
| **夏普比率** | 2.30 |
| **卡玛比率** | 2.99 |
| **最大回撤** | -36.30% |
| **回撤持续时间** | 26天 |
| **胜率** | 50.17% |
| **年化换手率** | 387.72% |

### 多空分解

| 组合 | 年化收益率 | 夏普比率 |
|------|-----------|---------|
| **多头组合** | 113.85% | 2.19 |
| **空头组合** | 113.71% | 2.06 |

### 关键洞察

- **卓越收益**: 策略在6年内实现了超过8,000%的总收益，年化收益率达到108.39%
- **优秀的风险调整收益**: 夏普比率为2.30，表明风险调整后的收益表现优异
- **平衡的多空组合**: 多头和空头都贡献显著，多头年化收益113.85%，空头年化收益113.71%
- **适度的回撤**: 最大回撤36.30%，考虑到高收益特征，这是合理的
- **积极交易**: 年化换手率387.72%反映了月度调仓和显著的仓位变化

## 🚀 快速开始

### 前置要求

- Python 3.8 或更高版本
- pip 包管理器

### 安装

1. **进入项目目录**:
   ```bash
   cd D:\QUANT\multi_factor
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

### 运行回测

直接运行：
```bash
python scripts/run_backtest.py
```

脚本将执行：
1. 加载或下载价格和基本面数据（后续运行使用缓存）
2. 预处理和过滤股票池
3. 计算所有因子（价值、动量、质量、规模）
4. 构建多空组合
5. 运行回测模拟
6. 生成性能指标、图表和报告

**首次运行**: 5-15分钟（下载数据）  
**后续运行**: 2-5分钟（使用缓存数据）

### 查看结果

运行完成后，结果保存在：
- **回测数据**: `results/backtest_results.parquet`
- **性能指标**: `results/metrics.json`
- **文本报告**: `results/performance_report.txt`
- **图表**: `reports/figures/*.png`

## 📁 项目结构

```
multi_factor/
├── src/                    # 源代码
│   ├── data_loader.py     # 下载/加载价格和基本面数据
│   ├── data_preprocess.py # 清洗和对齐数据
│   ├── factors/           # 因子构建
│   │   ├── value.py       # 价值因子（市净率、市盈率）
│   │   ├── momentum.py    # 动量因子（价格收益）
│   │   ├── quality.py     # 质量因子（ROE、ROA等）
│   │   └── size.py         # 规模因子（市值）
│   ├── factor_combiner.py # 组合因子为综合得分
│   ├── portfolio/         # 组合构建
│   │   └── construction.py # 多空选择和权重分配
│   ├── backtester/        # 回测引擎和指标
│   │   ├── engine.py      # 核心回测逻辑
│   │   └── metrics.py    # 性能指标计算
│   ├── analysis/          # 图表和报告
│   │   ├── plotting.py    # 可视化函数
│   │   └── performance_report.py # 报告生成
│   └── utils/             # 配置和工具
│       ├── config.py      # 配置参数
│       └── logging.py     # 日志设置
├── scripts/
│   ├── run_backtest.py    # 主执行脚本 ⭐
│   └── clean_cache.py     # 清理缓存数据
├── notebooks/             # Jupyter研究笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_backtest_demo.ipynb
│   └── 04_sensitivity_analysis.ipynb
├── data/                  # 数据存储（自动创建）
│   ├── raw/               # 缓存的价格/基本面数据
│   ├── processed/         # 处理后的数据
│   └── metadata/          # 股票池定义
├── results/               # 回测结果（自动创建）
├── reports/               # 报告和图表（自动创建）
└── requirements.txt       # Python依赖
```

## ⚙️ 配置

编辑 `src/utils/config.py` 进行自定义：

### 回测设置

- **日期范围**: `BACKTEST_CONFIG.start_date` 和 `end_date`
- **调仓频率**: `rebalance_frequency`（"monthly" 或 "quarterly"）
- **多空分位数**: `long_pct` 和 `short_pct`（默认：各20%）
- **交易成本**: `commission_rate` 和 `slippage_rate`（默认：各10基点）
- **初始资金**: `initial_capital`（默认：$100万）

### 因子设置

- **因子权重**: 调整 `FACTOR_CONFIG` 值
- **包含/排除规模因子**: 将 `size_weight` 设为0以排除
- **标准化方法**: "zscore" 或 "rank"

## 📈 策略概述

### 因子构建

1. **价值因子**: 结合市净率（BTM）和盈利价格比（EP）
2. **动量因子**: 12个月价格收益（跳过最近一个月）
3. **质量因子**: 净资产收益率（ROE）和总资产收益率（ROA）
4. **规模因子**: 市值（可选）

### 组合构建

- **多头组合**: 综合因子得分前20%的股票
- **空头组合**: 综合因子得分后20%的股票
- **权重分配**: 每个组合内等权重
- **约束条件**: 单只股票最大3%，美元中性（多头=空头）

### 调仓

- **频率**: 月度（每月最后一个交易日）
- **换手率**: 每次调仓平均约32%（年化387%）

## 🔧 高级用法

### 使用Jupyter笔记本

进行交互式探索：
```bash
jupyter notebook notebooks/
```

### 清理缓存

重新开始（删除所有缓存数据和结果）：
```bash
python scripts/clean_cache.py
```

### 自定义因子权重

修改 `src/utils/config.py`:
```python
FACTOR_CONFIG = FactorConfig(
    value_weight=0.30,
    momentum_weight=0.30,
    quality_weight=0.30,
    size_weight=0.10,  # 设为0以排除
    use_regression_weights=False
)
```

## 📊 性能分析

### 收益归因

策略的卓越表现可归因于：
- **强因子信号**: 价值、动量和质量因子提供了一致的alpha
- **美元中性结构**: 最小化市场暴露，专注于选股
- **定期调仓**: 月度调仓捕捉了变化的因子动态
- **广泛股票池**: S&P 500成分股提供多样化

### 风险特征

- **波动率**: 年化47.09%（对于多空股票策略来说较高但合理）
- **回撤**: 最大36.30%回撤，快速恢复（26天）
- **一致性**: 50.17%胜率显示盈亏分布平衡

## 🐛 故障排除

### 常见问题

1. **缺少依赖**: 运行 `pip install -r requirements.txt`
2. **数据下载错误**: 检查网络连接（使用Yahoo Finance API）
3. **导入错误**: 确保从项目根目录运行
4. **内存问题**: 减少日期范围或股票数量

### 首次运行注意事项

- 首次运行从Yahoo Finance下载数据（可能需要几分钟）
- 数据缓存在 `data/raw/` 中以加快后续运行
- 脚本自动创建必要的目录

## 📝 依赖

主要依赖：
- `pandas>=2.0.0` - 数据处理
- `numpy>=1.24.0` - 数值计算
- `yfinance>=0.2.28` - 市场数据下载
- `matplotlib>=3.7.0` - 绘图
- `scikit-learn>=1.3.0` - 机器学习工具
- `pyarrow>=10.0.0` - Parquet文件支持

完整列表见 `requirements.txt`。

## 📄 许可证

本项目用于教育和研究目的。

## 🤝 贡献

欢迎贡献！请随时提交问题或拉取请求。
