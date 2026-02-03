"""
Main script to run the multi-factor equity strategy backtest.

This script:
1. Loads and preprocesses data
2. Computes factors
3. Constructs portfolios
4. Runs backtest
5. Generates performance metrics and plots
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import load_universe, download_price_data, download_fundamental_data
from src.data_preprocess import preprocess_data, filter_universe
from src.factors import (
    compute_value_factor_panel,
    compute_momentum_factor_panel,
    compute_quality_factor_panel,
    compute_size_factor_panel,
)
from src.factor_combiner import compute_composite_scores_panel
from src.portfolio import construct_portfolio_panel
from src.backtester import BacktestEngine, get_rebalance_dates
from src.backtester.metrics import compute_metrics, format_metrics_report
from src.analysis import generate_all_plots, generate_performance_report
from src.utils.config import (
    BACKTEST_CONFIG,
    FACTOR_CONFIG,
    DATA_CONFIG,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)
from src.utils.logging import get_logger

logger = get_logger("scripts.run_backtest")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("MULTI-FACTOR EQUITY STRATEGY BACKTEST")
    logger.info("=" * 60)
    
    # Configuration summary
    logger.info("\nConfiguration:")
    logger.info(f"  Start Date: {BACKTEST_CONFIG.start_date}")
    logger.info(f"  End Date: {BACKTEST_CONFIG.end_date}")
    logger.info(f"  Rebalance Frequency: {BACKTEST_CONFIG.rebalance_frequency}")
    logger.info(f"  Long/Short Percentiles: {BACKTEST_CONFIG.long_pct:.0%}/{BACKTEST_CONFIG.short_pct:.0%}")
    logger.info(f"  Initial Capital: ${BACKTEST_CONFIG.initial_capital:,.0f}")
    
    # Step 1: Load universe
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading universe")
    logger.info("=" * 60)
    universe_df = load_universe()
    tickers = universe_df['ticker'].tolist()
    logger.info(f"Loaded {len(tickers)} tickers from universe")
    
    # Step 2: Download/Load data
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Loading price and fundamental data")
    logger.info("=" * 60)
    
    # Extend start date for momentum calculation
    data_start_date = pd.Timestamp(BACKTEST_CONFIG.start_date) - pd.DateOffset(months=15)
    data_start_date_str = data_start_date.strftime('%Y-%m-%d')
    
    price_data = download_price_data(
        tickers,
        data_start_date_str,
        BACKTEST_CONFIG.end_date,
        cache=True
    )
    logger.info(f"Loaded price data: {len(price_data)} records")
    
    fundamental_data = download_fundamental_data(
        tickers,
        data_start_date_str,
        BACKTEST_CONFIG.end_date,
        cache=True
    )
    logger.info(f"Loaded fundamental data: {len(fundamental_data)} records")
    
    # Step 3: Preprocess data
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Preprocessing data")
    logger.info("=" * 60)
    
    clean_prices, clean_fundamentals, market_cap, eligible_tickers = preprocess_data(
        price_data,
        fundamental_data
    )
    logger.info(f"Eligible tickers after filtering: {len(eligible_tickers)}")
    
    # Filter to backtest period
    backtest_start = pd.Timestamp(BACKTEST_CONFIG.start_date)
    clean_prices = clean_prices.loc[clean_prices.index.get_level_values('date') >= backtest_start]
    
    # Step 4: Get rebalancing dates
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Computing rebalancing dates")
    logger.info("=" * 60)
    
    trading_dates = clean_prices.index.get_level_values('date').unique().sort_values()
    rebalance_dates = get_rebalance_dates(
        trading_dates,
        BACKTEST_CONFIG.rebalance_frequency,
        BACKTEST_CONFIG.rebalance_day
    )
    logger.info(f"Rebalancing dates: {len(rebalance_dates)} dates")
    if len(rebalance_dates) > 0:
        logger.info(f"  First: {rebalance_dates[0]}, Last: {rebalance_dates[-1]}")
    else:
        logger.error("No rebalancing dates found. Check if price data is available for the backtest period.")
        raise ValueError("No rebalancing dates available. Cannot proceed with backtest.")
    
    # Step 5: Compute factors
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Computing factors")
    logger.info("=" * 60)
    
    # Value factor
    logger.info("Computing value factor...")
    value_factor = compute_value_factor_panel(
        clean_prices,
        clean_fundamentals if clean_fundamentals is not None else clean_prices[[]],
        market_cap,
        rebalance_dates
    )
    logger.info(f"Value factor computed: {len(value_factor)} records")
    
    # Momentum factor
    logger.info("Computing momentum factor...")
    momentum_factor = compute_momentum_factor_panel(
        clean_prices,
        rebalance_dates
    )
    logger.info(f"Momentum factor computed: {len(momentum_factor)} records")
    
    # Quality factor
    logger.info("Computing quality factor...")
    quality_factor = compute_quality_factor_panel(
        clean_prices,
        clean_fundamentals if clean_fundamentals is not None else clean_prices[[]],
        market_cap,
        rebalance_dates
    )
    logger.info(f"Quality factor computed: {len(quality_factor)} records")
    
    # Size factor (if enabled)
    factor_dict = {
        'value_factor': value_factor,
        'momentum_factor': momentum_factor,
        'quality_factor': quality_factor,
    }
    
    if FACTOR_CONFIG.size_weight > 0:
        logger.info("Computing size factor...")
        size_factor = compute_size_factor_panel(market_cap, rebalance_dates)
        factor_dict['size_factor'] = size_factor
        logger.info(f"Size factor computed: {len(size_factor)} records")
    
    # Combine into DataFrame
    all_factors = pd.concat(factor_dict, axis=1)
    all_factors.columns = all_factors.columns.droplevel(1) if isinstance(all_factors.columns, pd.MultiIndex) else all_factors.columns
    
    # Step 6: Compute composite scores
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Computing composite factor scores")
    logger.info("=" * 60)
    
    # Compute returns for regression weighting (if needed)
    returns = clean_prices['adj_close'].pct_change()
    
    composite_scores = compute_composite_scores_panel(
        all_factors,
        returns=returns if FACTOR_CONFIG.use_regression_weights else None
    )
    logger.info(f"Composite scores computed: {len(composite_scores)} records")
    
    # Step 7: Construct portfolios
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Constructing portfolios")
    logger.info("=" * 60)
    
    target_weights = construct_portfolio_panel(
        composite_scores,
        eligible_tickers=None,  # Already filtered
        sector_info=None,  # Would need sector data
        market_cap=market_cap
    )
    logger.info(f"Portfolios constructed: {len(target_weights)} weight records")
    
    # Step 8: Run backtest
    logger.info("\n" + "=" * 60)
    logger.info("Step 8: Running backtest")
    logger.info("=" * 60)
    
    engine = BacktestEngine()
    backtest_results = engine.run(
        clean_prices,
        target_weights,
        rebalance_dates
    )
    logger.info(f"Backtest complete. Final NAV: ${backtest_results['nav'].iloc[-1]:,.2f}")
    
    # Step 9: Compute metrics
    logger.info("\n" + "=" * 60)
    logger.info("Step 9: Computing performance metrics")
    logger.info("=" * 60)
    
    metrics = compute_metrics(backtest_results)
    
    # Print metrics
    logger.info("\n" + format_metrics_report(metrics))
    
    # Step 10: Generate plots and reports
    logger.info("\n" + "=" * 60)
    logger.info("Step 10: Generating plots and reports")
    logger.info("=" * 60)
    
    # Generate all plots
    generate_all_plots(
        backtest_results,
        metrics,
        benchmark_returns=None,  # Would need to load benchmark
        factor_scores=all_factors,
        returns=returns,
        output_dir=FIGURES_DIR
    )
    
    # Generate performance report
    report_path = RESULTS_DIR / "performance_report.txt"
    generate_performance_report(metrics, backtest_results, output_path=report_path)
    
    # Save backtest results
    results_path = RESULTS_DIR / "backtest_results.parquet"
    backtest_results.to_parquet(results_path)
    logger.info(f"Saved backtest results to {results_path}")
    
    # Save metrics
    metrics_path = RESULTS_DIR / "metrics.json"
    import json
    # Convert non-serializable values
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            metrics_serializable[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        elif pd.isna(v):
            metrics_serializable[k] = None
        else:
            metrics_serializable[k] = v
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Plots saved to: {FIGURES_DIR}")
    
    # Turnover diagnostic
    print("\n=== FINAL TURNOVER DIAGNOSTIC ===")
    turnover_series = backtest_results['turnover']
    non_zero_turnover = turnover_series[turnover_series > 0]
    print(f"Total rebalances executed: {len(non_zero_turnover)}")
    if len(non_zero_turnover) > 0:
        print(f"Avg turnover per rebalance: {non_zero_turnover.mean()*100:.1f}%")
        print(f"Max turnover per rebalance: {non_zero_turnover.max()*100:.1f}%")
    else:
        print("WARNING: No non-zero turnover found!")
    print(f"Rebalance dates in calendar: {len(rebalance_dates)}")
    if len(rebalance_dates) > 0:
        print(f"Sample date: {rebalance_dates[0]}")
    print("==================================")


if __name__ == "__main__":
    main()


