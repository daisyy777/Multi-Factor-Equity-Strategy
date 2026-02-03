# How to Run the Multi-Factor Equity Strategy Project

## Quick Start

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
# Navigate to the project root
cd D:\QUANT\multi_factor

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Backtest

The simplest way to run the complete backtest:

```bash
python scripts/run_backtest.py
```

This script will:
1. ✅ Load or download price and fundamental data (cached for future runs)
2. ✅ Preprocess and filter the universe
3. ✅ Compute all factors (Value, Momentum, Quality, Size)
4. ✅ Construct long-short portfolios
5. ✅ Run the backtest simulation
6. ✅ Generate performance metrics, plots, and reports

### 3. View Results

After running, results are saved to:

- **Backtest results**: `results/backtest_results.parquet`
- **Performance metrics**: `results/metrics.json`
- **Text report**: `results/performance_report.txt`
- **Plots**: `reports/figures/*.png`

## Configuration (Optional)

Edit `src/utils/config.py` to customize:

### Backtest Settings
- **Date range**: `BACKTEST_CONFIG.start_date` and `end_date` (default: 2018-01-01 to 2023-12-31)
- **Rebalancing**: `rebalance_frequency` ("monthly" or "quarterly")
- **Long/Short percentiles**: `long_pct` and `short_pct` (default: 20% each)
- **Transaction costs**: `commission_rate` and `slippage_rate` (default: 10 bps each)
- **Initial capital**: `initial_capital` (default: $1M)

### Factor Settings
- **Factor weights**: Adjust `FACTOR_CONFIG` values
- **Include/exclude Size factor**: Set `size_weight` to 0 to exclude, >0 to include
- **Standardization method**: "zscore" or "rank"

## Using Jupyter Notebooks

For interactive exploration and analysis:

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/
```

Available notebooks:
- **01_data_exploration.ipynb** - Explore price and fundamental data
- **02_factor_research.ipynb** - Analyze individual factors and correlations
- **03_backtest_demo.ipynb** - Run and visualize backtest results
- **04_sensitivity_analysis.ipynb** - Test parameter sensitivity

## Project Structure

```
multi_factor/
├── src/                    # Source code
│   ├── data_loader.py     # Download/load data
│   ├── data_preprocess.py # Clean and align data
│   ├── factors/           # Factor construction
│   ├── factor_combiner.py # Combine factors
│   ├── portfolio/         # Portfolio construction
│   ├── backtester/        # Backtest engine & metrics
│   ├── analysis/          # Plots and reports
│   └── utils/             # Configuration
├── scripts/
│   └── run_backtest.py    # Main execution script ⭐
├── notebooks/             # Jupyter notebooks for research
├── data/                  # Data storage (created automatically)
├── results/               # Backtest results (created automatically)
└── reports/               # Reports and figures (created automatically)
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure all packages from `requirements.txt` are installed
2. **Data download errors**: Check internet connection (uses Yahoo Finance API)
3. **Import errors**: Make sure you're running from the project root directory
4. **Memory issues**: Reduce date range or number of tickers if running out of memory

### First Run Notes

- The first run will download data from Yahoo Finance, which may take a few minutes
- Data is cached in `data/raw/` for faster subsequent runs
- The script creates necessary directories automatically

## Command Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (main script)
python scripts/run_backtest.py

# Run Jupyter notebooks
jupyter notebook notebooks/

# View results (after running backtest)
# - Check: results/metrics.json
# - Check: results/performance_report.txt
# - Check: reports/figures/*.png
```

## Need Help?

- See `README.md` for detailed documentation
- See `QUICK_START.md` for a condensed guide
- Check `src/utils/config.py` for all configuration options




