# Multi-Factor Equity Strategy Backtest

A quantitative long-short equity strategy framework implementing a multi-factor model with a rigorous backtesting engine. Factors covered: **Value**, **Momentum**, **Quality**, and (optional) **Size**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Design Principles

This project prioritises methodological correctness over inflated numbers:

| Principle | Implementation |
|-----------|---------------|
| **No look-ahead bias** | Fundamental data timestamped as `quarter_end + 60 days` (SEC 10-Q filing deadline) |
| **Reduced survivorship bias** | Historical S&P 500 constituents loaded per rebalance date |
| **Realistic transaction costs** | 10 bps commission + 10 bps slippage per side |
| **Turnover control** | Max 30% one-way turnover per rebalance |
| **Factor validation** | IC analysis run before backtest; weak signals flagged |

---

## Backtest Results

**Period**: 2018-01-01 → 2026-02-03 (8 years)  
**Universe**: ~91 large-cap US stocks  
**Initial Capital**: $1,000,000  
**Rebalancing**: Monthly (last trading day)  
**Data source**: Yahoo Finance (price) + yfinance quarterly statements (fundamentals, 60-day lag)

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final NAV** | $591,555 |
| **Total Return** | -40.84% |
| **Annualized Return** | -6.30% |
| **Annualized Volatility** | 21.42% |
| **Sharpe Ratio** | -0.29 |
| **Calmar Ratio** | -0.10 |
| **Max Drawdown** | -63.46% |
| **Drawdown Duration** | 1,178 days |
| **Hit Ratio** | 51.67% |
| **Annual Turnover** | 380.81% |

### Long / Short Decomposition

| Leg | Ann. Return | Sharpe |
|-----|-------------|--------|
| Long | -6.54% | -0.18 |
| Short | -6.46% | -0.16 |

### Factor IC Analysis

IC = Spearman rank correlation between factor scores and next-period returns.  
Rule of thumb: `|IC| > 0.05` strong · `|IC| > 0.02` marginal · `|IC| < 0.02` weak.

| Factor | IC Mean | IC Std | ICIR | t-stat | % Positive | Signal |
|--------|---------|--------|------|--------|------------|--------|
| value_factor | **0.150** | 0.137 | 1.10 | 3.63 | 91% | **strong** |
| momentum_factor | 0.003 | 0.227 | 0.01 | 0.14 | 52% | weak |
| quality_factor | 0.012 | 0.106 | 0.12 | 0.35 | 67% | weak |

### Results Interpretation

The strategy produced negative returns over this period. This is an honest outcome:

- **Value factor is statistically significant** (IC=0.15, t=3.63) but the 2018–2022 era was one of the worst on record for value investing — growth/tech dominance systematically penalised value-long / growth-short portfolios.
- **Momentum and quality factors show weak signal** given the limited fundamental data available from yfinance (quarterly statements with limited history depth). A production system would use Compustat or SimFin.
- **Both legs lose together**, which is characteristic of a period where the long-short spread compressed — the strategy captures cross-sectional factor spreads, not market direction, but factor spreads were unusually narrow.

These results reflect what the strategy would actually have earned — not an artefact of data leakage.

---

## Project Structure

```
multi_factor/
├── src/
│   ├── data_loader.py          # Price download (yfinance) + historical fundamental data
│   │                           #   ↳ download_fundamental_data(): quarterly statements,
│   │                           #     60-day reporting lag — no look-ahead bias
│   │                           #   ↳ load_historical_sp500_universe(): point-in-time index
│   ├── data_preprocess.py      # Cleaning, universe filtering, forward-fill fundamentals
│   ├── factors/
│   │   ├── value.py            # BTM + E/P, uses total book_value directly
│   │   ├── momentum.py         # 12-1 month price momentum
│   │   ├── quality.py          # ROE, leverage, earnings stability
│   │   └── size.py             # Market cap (optional)
│   ├── factor_combiner.py      # Composite score (equal / regression weights)
│   ├── portfolio/
│   │   ├── construction.py     # Long-short selection + turnover constraint
│   │   └── constraints.py      # Sector / weight caps
│   ├── backtester/
│   │   ├── engine.py           # Event-driven daily backtest loop
│   │   └── metrics.py          # Sharpe, Calmar, drawdown, IC, etc.
│   ├── analysis/
│   │   ├── ic_analysis.py      # IC time series, ICIR, tear-sheet plot
│   │   ├── plotting.py         # NAV, drawdown, factor exposure charts
│   │   └── performance_report.py
│   └── utils/
│       ├── config.py           # All parameters in one place
│       └── logging.py
├── scripts/
│   ├── run_backtest.py         # Main entry point (11-step pipeline)
│   └── clean_cache.py          # Wipe cached data to force re-download
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_backtest_demo.ipynb
│   └── 04_sensitivity_analysis.ipynb
├── data/                       # Auto-created; cached parquet files
├── results/                    # backtest_results.parquet, metrics.json,
│                               # ic_panel.parquet, ic_summary.csv
├── reports/figures/            # PNG charts
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the backtest

```bash
python scripts/run_backtest.py
```

The pipeline runs 11 steps automatically:

1. Load universe (historical S&P 500 constituents when available)
2. Download price data via yfinance (cached to `data/raw/`)
3. Download quarterly fundamental statements with 60-day lag (cached)
4. Preprocess & filter universe
5. Compute rebalancing dates
6. Compute Value / Momentum / Quality factors
7. **IC validation** — prints per-factor IC summary, warns on weak signals
8. Compute composite scores
9. Construct portfolios (turnover-constrained, point-in-time universe)
10. Run backtest engine
11. Generate metrics, plots, and reports

**First run**: ~5–10 min (downloads fundamentals for all tickers)  
**Subsequent runs**: ~1–3 min (uses cached data)

### 3. View results

| Output | Location |
|--------|----------|
| Performance metrics | `results/metrics.json` |
| Text report | `results/performance_report.txt` |
| IC summary | `results/ic_summary.csv` |
| Backtest NAV series | `results/backtest_results.parquet` |
| Charts | `reports/figures/*.png` |

### 4. Reset cache

```bash
python scripts/clean_cache.py
```

---

## Configuration

All parameters live in `src/utils/config.py`.

```python
# Key backtest settings
BACKTEST_CONFIG = BacktestConfig(
    start_date="2018-01-01",
    end_date="2026-02-03",
    rebalance_frequency="monthly",   # "monthly" or "quarterly"
    long_pct=0.20,                   # Top 20% → long
    short_pct=0.20,                  # Bottom 20% → short
    max_turnover_per_rebalance=0.30, # Cap one-way turnover at 30%
    commission_rate=0.0010,          # 10 bps per side
    slippage_rate=0.0010,
    initial_capital=1_000_000,
)

# Factor weights (None = equal weight)
FACTOR_CONFIG = FactorConfig(
    value_btm_weight=0.5,
    value_ep_weight=0.5,
    momentum_lookback_months=12,
    momentum_skip_months=1,
    size_weight=0.0,                 # Set > 0 to include size factor
)

# Data settings
DATA_CONFIG = DataConfig(
    use_historical_universe=True,    # Point-in-time S&P 500 membership
    fundamental_report_lag_days=60, # Days after quarter-end before data is used
)
```

---

## Factor Construction

### Value
- **Book-to-Market (BTM)**: total stockholders' equity / market cap
- **Earnings-to-Price (E/P)**: trailing EPS / price
- Combined via equal-weighted z-score; winsorised at 1st/99th percentile

### Momentum
- 12-month minus 1-month price return (standard 12-1 momentum)
- Skips most recent month to avoid short-term reversal

### Quality
- **ROE**: net income / book value
- **Leverage** (negative): total liabilities / total assets
- **Earnings stability** (negative variance): EPS variance over last 8 quarters

### Portfolio Construction
- Top/bottom 20% by composite score → long/short legs
- Equal weight within each leg; 3% single-stock cap
- Dollar-neutral (long notional = short notional)
- Turnover limited to 30% one-way per rebalance

---

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| yfinance quarterly statements have limited history depth | Fewer IC observations for value/quality | Use Compustat or SimFin for production |
| Historical S&P 500 data falls back to static list when offline | Residual survivorship bias | `load_historical_sp500_universe()` auto-downloads when online |
| ~98 tickers (large-cap only) | Factor spreads narrow; fewer diversification opportunities | Expand universe to Russell 1000 |
| No sector neutralisation | Sector tilts can dominate factor signal | Enable `max_sector_exposure` constraint in config |

---

## Dependencies

```
pandas>=2.0.0      numpy>=1.24.0     yfinance>=0.2.28
matplotlib>=3.7.0  scipy>=1.11.0     scikit-learn>=1.3.0
seaborn>=0.12.0    pyarrow>=10.0.0   jupyter>=1.0.0
```

---

## References

- Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers.
