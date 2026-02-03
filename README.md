# Multi-Factor Equity Strategy

A complete, production-quality research framework for implementing and backtesting a cross-sectional multi-factor long-short equity strategy on U.S. large-cap stocks.

## Overview

This project implements a systematic quantitative equity strategy that combines multiple factors—Value, Momentum, and Quality—to construct a dollar-neutral long-short portfolio. The framework includes:

- **Factor Construction**: Value (BTM, E/P), Momentum (12-1 month), Quality (ROE, Leverage, Earnings Stability)
- **Portfolio Construction**: Long-short selection with multiple weighting schemes and risk constraints
- **Backtest Engine**: Event-driven daily simulation with transaction costs and slippage
- **Performance Analytics**: Comprehensive metrics, visualizations, and reporting
- **Research Documentation**: Detailed methodology and results analysis

## Features

- ✅ Modular, extensible architecture
- ✅ Multiple factor definitions with clear economic rationale
- ✅ Flexible portfolio construction (equal-weight, score-proportional)
- ✅ Risk controls (sector limits, position limits, beta-neutrality)
- ✅ Realistic transaction cost modeling
- ✅ Comprehensive performance metrics (Sharpe, IR, drawdown, etc.)
- ✅ Automated visualization and reporting
- ✅ Example Jupyter notebooks for research and analysis

## Project Structure

```
multi-factor-equity-strategy/
├── src/                      # Source code
│   ├── data_loader.py       # Data downloading/loading
│   ├── data_preprocess.py   # Data cleaning and preprocessing
│   ├── factors/             # Factor construction modules
│   ├── factor_combiner.py   # Multi-factor combination
│   ├── portfolio/           # Portfolio construction and constraints
│   ├── backtester/          # Backtest engine and metrics
│   ├── analysis/            # Performance analysis and plotting
│   └── utils/               # Configuration and utilities
├── notebooks/               # Example Jupyter notebooks
├── scripts/                 # Execution scripts
├── data/                    # Data storage
│   ├── raw/                # Raw downloaded data
│   ├── processed/          # Processed data
│   └── metadata/           # Universe and metadata files
├── results/                 # Backtest results
└── reports/                 # Research reports and figures
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-factor-equity-strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n multifactor python=3.10
conda activate multifactor
pip install -r requirements.txt
```

3. Configure settings (optional):
   - Edit `src/utils/config.py` to modify backtest parameters, factor weights, transaction costs, etc.

## Quick Start

### Running a Backtest

The simplest way to run a complete backtest:

```bash
python scripts/run_backtest.py
```

This will:
1. Load or download price and fundamental data
2. Preprocess and filter the universe
3. Compute all factors (Value, Momentum, Quality, Size)
4. Construct portfolios and run the backtest
5. Generate performance metrics, plots, and reports

Results are saved to:
- `results/backtest_results.parquet` - Full backtest time series
- `results/metrics.json` - Performance metrics
- `results/performance_report.txt` - Formatted report
- `reports/figures/*.png` - Visualization plots

### Using Jupyter Notebooks

Example notebooks are provided in the `notebooks/` directory:

1. **01_data_exploration.ipynb** - Explore price and fundamental data
2. **02_factor_research.ipynb** - Analyze individual factors and correlations
3. **03_backtest_demo.ipynb** - Run and visualize backtest results
4. **04_sensitivity_analysis.ipynb** - Test parameter sensitivity

To run notebooks:
```bash
jupyter notebook notebooks/
```

## Configuration

Key configuration parameters are in `src/utils/config.py`:

### Backtest Configuration

- `start_date`, `end_date`: Backtest date range
- `rebalance_frequency`: "monthly" or "quarterly"
- `long_pct`, `short_pct`: Percentage of stocks for long/short legs (default: 20%)
- `weighting_scheme`: "equal" or "score_proportional"
- `commission_rate`, `slippage_rate`: Transaction costs (default: 10 bps each)
- `initial_capital`: Starting capital (default: $1M)

### Factor Configuration

- `value_btm_weight`, `value_ep_weight`: Value factor component weights
- `momentum_lookback_months`: Momentum lookback period (default: 12)
- `quality_roe_weight`, etc.: Quality factor component weights
- `size_weight`: Size factor weight (0 to exclude, >0 to include)
- `standardization_method`: "zscore" or "rank"
- `use_regression_weights`: Use regression-based factor weighting

See `src/utils/config.py` for all available parameters.

## Data Sources

**Current Implementation:**
- **Price Data**: Yahoo Finance API (via `yfinance`)
- **Fundamental Data**: Yahoo Finance (simplified; production would use Compustat/FactSet)
- **Universe**: S&P 500 or custom ticker list in `data/metadata/sp500_constituents.csv`

**Data Caching:**
- Downloaded data is cached to `data/raw/` as Parquet files
- Preprocessed data saved to `data/processed/`

**Note**: For production use, replace the data loading functions with professional data providers (Compustat, FactSet, Bloomberg, etc.).

## Factor Definitions

### Value Factor

Combines Book-to-Market (BTM) and Earnings-to-Price (E/P) ratios:
- BTM = Book Value of Equity / Market Cap
- E/P = Earnings Per Share / Price Per Share

### Momentum Factor

12-1 month momentum (excluding most recent month):
- Mom = (Price_{t-1m} - Price_{t-12m}) / Price_{t-12m}

### Quality Factor

Combines profitability, leverage, and earnings stability:
- ROE = Net Income / Book Equity
- Leverage = Total Liabilities / Total Assets (negative weight)
- Earnings Stability = -Variance(EPS over past N quarters)

### Size Factor (Optional)

- Size = -z(log(Market Cap))

## Performance Metrics

The backtest computes comprehensive performance metrics:

- **Returns**: Annualized return, total return
- **Risk**: Annualized volatility, maximum drawdown, drawdown duration
- **Risk-Adjusted**: Sharpe ratio, Calmar ratio, Information ratio (vs benchmark)
- **Other**: Hit ratio, turnover, long/short decomposition

See `src/backtester/metrics.py` for full list.

## Example Results

After running a backtest, view results:

```python
import pandas as pd
import json
from src.utils.config import RESULTS_DIR

# Load backtest results
results = pd.read_parquet(RESULTS_DIR / "backtest_results.parquet")

# Load metrics
with open(RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2%}")
```

## Research Report

A detailed research report is available in `reports/multifactor_strategy_report.md`, covering:
- Strategy methodology and factor definitions
- Data sources and universe selection
- Portfolio construction and backtesting methodology
- Empirical results and performance analysis
- Risk management and limitations
- Future enhancements

## Contributing

This is a research framework designed for extension and modification. Key areas for contribution:

- Additional factors (e.g., Low Volatility, Earnings Revisions)
- Advanced portfolio optimization (mean-variance, risk parity)
- Integration with professional data providers
- Enhanced risk models (Barra, Axioma)
- Walk-forward analysis and out-of-sample testing

## Limitations & Disclaimers

**This is a research framework, not investment advice.**

- Simplified data sources (Yahoo Finance) vs. professional providers
- Backtested results may not predict future performance
- Transaction costs and slippage models are simplified
- No consideration of borrowing costs, taxes, or regulatory constraints
- Assumes perfect execution and market access

For production use, integrate professional data, risk models, and execution systems.

## License

[Specify your license here]

## Author

[Your name/contact information]

## Acknowledgments

- Fama-French factor models
- Academic research on factor investing
- Open-source quantitative finance libraries (pandas, numpy, matplotlib, etc.)

---

**Version**: 1.0.0  
**Last Updated**: 2024

