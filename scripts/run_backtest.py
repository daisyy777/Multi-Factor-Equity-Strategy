"""
Main script to run the multi-factor equity strategy backtest.

Steps
-----
1.  Load universe (historical S&P 500 components when available).
2.  Download / cache price and fundamental data.
3.  Preprocess data.
4.  Compute rebalancing dates.
5.  Compute factors.
6.  [NEW] IC validation — verify factor signals before running the full
    backtest.  Weak factors (|IC| < 0.02) are flagged with a warning.
7.  Compute composite scores.
8.  Construct portfolios (with turnover constraint + historical universe
    filtering at each rebalance date to eliminate survivorship bias).
9.  Run backtest.
10. Compute metrics.
11. Generate plots and reports.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import (
    load_universe,
    load_historical_sp500_universe,
    download_price_data,
    download_fundamental_data,
)
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
from src.analysis import (
    generate_all_plots,
    generate_performance_report,
    compute_ic_panel,
    summarise_ic,
    plot_ic_tearsheet,
)
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
    logger.info("=" * 60)
    logger.info("MULTI-FACTOR EQUITY STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"  Start Date             : {BACKTEST_CONFIG.start_date}")
    logger.info(f"  End Date               : {BACKTEST_CONFIG.end_date}")
    logger.info(f"  Rebalance Frequency    : {BACKTEST_CONFIG.rebalance_frequency}")
    logger.info(f"  Long/Short Percentiles : "
                f"{BACKTEST_CONFIG.long_pct:.0%}/{BACKTEST_CONFIG.short_pct:.0%}")
    logger.info(f"  Max Turnover/Rebalance : "
                f"{BACKTEST_CONFIG.max_turnover_per_rebalance:.0%}")
    logger.info(f"  Fundamental Lag        : "
                f"{DATA_CONFIG.fundamental_report_lag_days} days")
    logger.info(f"  Historical Universe    : {DATA_CONFIG.use_historical_universe}")

    # ------------------------------------------------------------------
    # Step 1: Load universe
    # ------------------------------------------------------------------
    logger.info("\nStep 1: Loading universe")
    if DATA_CONFIG.use_historical_universe:
        # Load the full history now; we’ll filter per-date later.
        try:
            hist_universe_df = load_historical_sp500_universe(date=None)
            logger.info(
                f"Loaded historical S&P 500 universe "
                f"({len(hist_universe_df)} snapshots)"
            )
            # Collect ALL tickers that ever appeared to download price data.
            if isinstance(hist_universe_df, pd.DataFrame) and "tickers" in hist_universe_df.columns:
                all_tickers_ever: set = set()
                for row in hist_universe_df["tickers"]:
                    all_tickers_ever.update(
                        [t.strip() for t in str(row).split(",") if t.strip()]
                    )
                tickers = sorted(all_tickers_ever)
            else:
                # Fallback: hist_universe_df is already a DataFrame with 'ticker'
                tickers = hist_universe_df["ticker"].tolist()
            logger.info(f"Total unique tickers across history: {len(tickers)}")
        except Exception as e:
            logger.warning(f"Historical universe load failed ({e}); using static list.")
            hist_universe_df = None
            tickers = load_universe()["ticker"].tolist()
    else:
        hist_universe_df = None
        tickers = load_universe()["ticker"].tolist()

    logger.info(f"Universe size: {len(tickers)} tickers")

    # ------------------------------------------------------------------
    # Step 2: Download / load data
    # ------------------------------------------------------------------
    logger.info("\nStep 2: Loading price and fundamental data")
    data_start = (
        pd.Timestamp(BACKTEST_CONFIG.start_date) - pd.DateOffset(months=15)
    ).strftime("%Y-%m-%d")

    price_data = download_price_data(
        tickers, data_start, BACKTEST_CONFIG.end_date, cache=True
    )
    logger.info(f"Price data: {len(price_data)} records")

    fundamental_data = download_fundamental_data(
        tickers, data_start, BACKTEST_CONFIG.end_date, cache=True
    )
    logger.info(f"Fundamental data: {len(fundamental_data)} records")

    # ------------------------------------------------------------------
    # Step 3: Preprocess
    # ------------------------------------------------------------------
    logger.info("\nStep 3: Preprocessing data")
    clean_prices, clean_fundamentals, market_cap, eligible_tickers = preprocess_data(
        price_data, fundamental_data
    )
    logger.info(f"Eligible tickers after filtering: {len(eligible_tickers)}")

    backtest_start = pd.Timestamp(BACKTEST_CONFIG.start_date)
    clean_prices = clean_prices.loc[
        clean_prices.index.get_level_values("date") >= backtest_start
    ]

    # ------------------------------------------------------------------
    # Step 4: Rebalancing dates
    # ------------------------------------------------------------------
    logger.info("\nStep 4: Computing rebalancing dates")
    trading_dates = clean_prices.index.get_level_values("date").unique().sort_values()
    rebalance_dates = get_rebalance_dates(
        trading_dates,
        BACKTEST_CONFIG.rebalance_frequency,
        BACKTEST_CONFIG.rebalance_day,
    )
    logger.info(f"Rebalancing dates: {len(rebalance_dates)}")
    if len(rebalance_dates) == 0:
        raise ValueError("No rebalancing dates found.")
    logger.info(f"  First: {rebalance_dates[0]}, Last: {rebalance_dates[-1]}")

    # ------------------------------------------------------------------
    # Step 5: Compute factors
    # ------------------------------------------------------------------
    logger.info("\nStep 5: Computing factors")

    fund_arg = clean_fundamentals if clean_fundamentals is not None else clean_prices[[]]

    logger.info("  Value factor...")
    value_factor = compute_value_factor_panel(
        clean_prices, fund_arg, market_cap, rebalance_dates
    )

    logger.info("  Momentum factor...")
    momentum_factor = compute_momentum_factor_panel(clean_prices, rebalance_dates)

    logger.info("  Quality factor...")
    quality_factor = compute_quality_factor_panel(
        clean_prices, fund_arg, market_cap, rebalance_dates
    )

    factor_dict = {
        "value_factor": value_factor,
        "momentum_factor": momentum_factor,
        "quality_factor": quality_factor,
    }

    if FACTOR_CONFIG.size_weight > 0:
        logger.info("  Size factor...")
        size_factor = compute_size_factor_panel(market_cap, rebalance_dates)
        factor_dict["size_factor"] = size_factor

    all_factors = pd.concat(factor_dict, axis=1)
    if isinstance(all_factors.columns, pd.MultiIndex):
        all_factors.columns = all_factors.columns.droplevel(1)

    # ------------------------------------------------------------------
    # Step 6: IC validation  [NEW]
    # ------------------------------------------------------------------
    logger.info("\nStep 6: IC validation (factor signal quality check)")
    try:
        ic_panel = compute_ic_panel(all_factors, clean_prices, rebalance_dates)
        ic_summary = summarise_ic(ic_panel)

        logger.info("\n" + "=" * 55)
        logger.info("FACTOR IC SUMMARY")
        logger.info("=" * 55)
        if not ic_summary.empty:
            logger.info("\n" + ic_summary.to_string())
            weak_factors = ic_summary[
                ic_summary["IC_mean"].abs() < 0.02
            ].index.tolist()
            if weak_factors:
                logger.warning(
                    f"\nWARNING: The following factors show very weak predictive "
                    f"signal (|IC| < 0.02) and may not contribute meaningful alpha:\n"
                    f"  {weak_factors}\n"
                    f"Consider reviewing the factor construction or data quality."
                )
            else:
                logger.info(
                    "\nAll factors show at least marginal signal strength (|IC| >= 0.02)."
                )
        else:
            logger.warning("IC summary is empty — check factor and price data.")
        logger.info("=" * 55)

        # Save IC panel
        ic_panel.to_parquet(RESULTS_DIR / "ic_panel.parquet")
        ic_summary.to_csv(RESULTS_DIR / "ic_summary.csv")

        # IC tear-sheet
        plot_ic_tearsheet(
            ic_panel,
            output_path=str(FIGURES_DIR / "ic_tearsheet.png"),
        )
    except Exception as e:
        logger.warning(f"IC validation failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Step 7: Composite scores
    # ------------------------------------------------------------------
    logger.info("\nStep 7: Computing composite factor scores")
    returns = clean_prices["adj_close"].pct_change()
    composite_scores = compute_composite_scores_panel(
        all_factors,
        returns=returns if FACTOR_CONFIG.use_regression_weights else None,
    )
    logger.info(f"Composite scores: {len(composite_scores)} records")

    # ------------------------------------------------------------------
    # Step 8: Construct portfolios
    # ------------------------------------------------------------------
    logger.info("\nStep 8: Constructing portfolios")

    # Build per-date eligible ticker sets from historical universe.
    # This removes stocks that were NOT in S&P 500 at a given rebalance
    # date, eliminating look-forward survivorship bias at portfolio level.
    if DATA_CONFIG.use_historical_universe and hist_universe_df is not None:
        logger.info(
            "  Filtering scores to historical S&P 500 constituents at each date"
        )
        filtered_scores_list: list = []
        for reb_date in rebalance_dates:
            date_uni = load_historical_sp500_universe(
                date=reb_date, fallback_to_static=True
            )
            date_tickers = set(date_uni["ticker"].tolist())
            try:
                date_scores = composite_scores.loc[(reb_date, slice(None)), :]
                # Keep only tickers in S&P 500 on this date
                tickers_in_idx = date_scores.index.get_level_values("ticker")
                mask = [t in date_tickers for t in tickers_in_idx]
                filtered_scores_list.append(date_scores.iloc[mask])
            except (KeyError, IndexError):
                pass
        if filtered_scores_list:
            composite_scores_filtered = pd.concat(filtered_scores_list)
        else:
            composite_scores_filtered = composite_scores
    else:
        composite_scores_filtered = composite_scores

    target_weights = construct_portfolio_panel(
        composite_scores_filtered,
        eligible_tickers=None,
        sector_info=None,
        market_cap=market_cap,
    )
    logger.info(f"Portfolio weights: {len(target_weights)} records")

    # ------------------------------------------------------------------
    # Step 9: Run backtest
    # ------------------------------------------------------------------
    logger.info("\nStep 9: Running backtest")
    engine = BacktestEngine()
    backtest_results = engine.run(clean_prices, target_weights, rebalance_dates)
    logger.info(
        f"Backtest complete. Final NAV: ${backtest_results['nav'].iloc[-1]:,.2f}"
    )

    # ------------------------------------------------------------------
    # Step 10: Performance metrics
    # ------------------------------------------------------------------
    logger.info("\nStep 10: Computing performance metrics")
    metrics = compute_metrics(backtest_results)
    logger.info("\n" + format_metrics_report(metrics))

    # ------------------------------------------------------------------
    # Step 11: Plots and reports
    # ------------------------------------------------------------------
    logger.info("\nStep 11: Generating plots and reports")
    generate_all_plots(
        backtest_results,
        metrics,
        benchmark_returns=None,
        factor_scores=all_factors,
        returns=returns,
        output_dir=FIGURES_DIR,
    )
    generate_performance_report(
        metrics, backtest_results, output_path=RESULTS_DIR / "performance_report.txt"
    )

    backtest_results.to_parquet(RESULTS_DIR / "backtest_results.parquet")

    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            metrics_serializable[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
            metrics_serializable[k] = None
        else:
            metrics_serializable[k] = v
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results : {RESULTS_DIR}")
    logger.info(f"Plots   : {FIGURES_DIR}")

    # Turnover diagnostic
    turnover_series = backtest_results["turnover"]
    non_zero = turnover_series[turnover_series > 0]
    print("\n=== TURNOVER DIAGNOSTIC ===")
    print(f"Total rebalances executed : {len(non_zero)}")
    if len(non_zero) > 0:
        print(f"Avg turnover/rebalance    : {non_zero.mean() * 100:.1f}%")
        print(f"Max turnover/rebalance    : {non_zero.max() * 100:.1f}%")
    print(f"Max allowed/rebalance     : "
          f"{BACKTEST_CONFIG.max_turnover_per_rebalance * 100:.0f}%")
    print("=" * 28)


if __name__ == "__main__":
    main()
