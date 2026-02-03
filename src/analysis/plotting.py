"""
Plotting module for backtest analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, List

from ..utils.config import FIGURES_DIR
from ..utils.logging import get_logger

logger = get_logger("analysis.plotting")

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def plot_cumulative_returns(
    backtest_results: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
    title: str = "Cumulative Returns"
) -> None:
    """
    Plot cumulative returns for strategy and benchmark.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with 'returns' column
    benchmark_returns : pd.Series, optional
        Benchmark returns
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Strategy cumulative returns
    strategy_cumret = (1 + backtest_results['returns']).cumprod()
    ax.plot(strategy_cumret.index, strategy_cumret.values, label='Strategy', linewidth=2)
    
    # Benchmark cumulative returns
    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(strategy_cumret.index, method='ffill')
        benchmark_cumret = (1 + aligned_benchmark).cumprod()
        ax.plot(benchmark_cumret.index, benchmark_cumret.values, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cumulative returns plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdown(
    backtest_results: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Drawdown"
) -> None:
    """
    Plot drawdown chart.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with 'nav' column
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nav = backtest_results['nav']
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown.values, linewidth=1.5, color='darkred')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved drawdown plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rolling_sharpe(
    backtest_results: pd.DataFrame,
    window: int = 252,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot rolling Sharpe ratio.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with 'returns' column
    window : int
        Rolling window size in days
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    # Set title if not provided
    if title is None:
        title = f"Rolling Sharpe Ratio ({window} days)"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    returns = backtest_results['returns']
    
    # Compute rolling Sharpe (annualized)
    rolling_mean = returns.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved rolling Sharpe plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_long_short_decomposition(
    backtest_results: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Long/Short Decomposition"
) -> None:
    """
    Plot cumulative returns for long and short legs separately.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with 'long_value' and 'short_value' columns
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    long_value = backtest_results['long_value']
    short_value = backtest_results['short_value']
    
    # Compute cumulative returns (normalize to start at 1)
    long_cumret = long_value / long_value.iloc[0]
    short_cumret = short_value / short_value.iloc[0] if short_value.iloc[0] > 0 else short_value
    
    ax.plot(long_cumret.index, long_cumret.values, label='Long Leg', linewidth=2, color='green')
    ax.plot(short_cumret.index, short_cumret.values, label='Short Leg', linewidth=2, color='red')
    
    # Combined (long - short)
    combined_cumret = (1 + backtest_results['returns']).cumprod()
    ax.plot(combined_cumret.index, combined_cumret.values, label='Long-Short', linewidth=2, color='blue', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved long/short decomposition plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_factor_deciles(
    factor_scores: pd.Series,
    returns: pd.Series,
    n_deciles: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Factor Decile Portfolio Returns"
) -> None:
    """
    Plot returns by factor decile portfolios.
    
    Parameters
    ----------
    factor_scores : pd.Series
        Factor scores with (date, ticker) MultiIndex
    returns : pd.Series
        Returns with (date, ticker) MultiIndex
    n_deciles : int
        Number of deciles (default 10)
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    # Align scores and returns
    aligned_scores, aligned_returns = factor_scores.align(returns, join='inner')
    
    # Group by date and compute decile returns
    dates = aligned_scores.index.get_level_values('date').unique()
    decile_returns = []
    
    for date in dates:
        date_scores = aligned_scores.loc[(date, slice(None))]
        date_returns = aligned_returns.loc[(date, slice(None))]
        
        # Find common tickers
        common_tickers = date_scores.index.get_level_values('ticker').intersection(
            date_returns.index.get_level_values('ticker')
        )
        
        if len(common_tickers) < n_deciles:
            continue
        
        date_scores_aligned = date_scores.loc[common_tickers]
        date_returns_aligned = date_returns.loc[common_tickers]
        
        # Create deciles
        deciles = pd.qcut(date_scores_aligned, q=n_deciles, labels=False, duplicates='drop')
        
        # Compute average return per decile
        decile_avg_returns = date_returns_aligned.groupby(deciles).mean()
        decile_avg_returns.index = decile_avg_returns.index + 1  # 1-indexed
        
        decile_returns.append(decile_avg_returns)
    
    if not decile_returns:
        logger.warning("No decile returns computed")
        return
    
    # Average across all dates
    decile_df = pd.DataFrame(decile_returns)
    decile_means = decile_df.mean()
    decile_stds = decile_df.std()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(decile_means))
    ax.bar(x_pos, decile_means.values * 252, yerr=decile_stds.values * np.sqrt(252), 
           capsize=5, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Factor Decile (1=Low, 10=High)', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(decile_means.index)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved factor decile plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_summary(
    metrics: dict,
    save_path: Optional[Path] = None,
    title: str = "Performance Metrics Summary"
) -> None:
    """
    Plot key metrics as a bar chart.
    
    Parameters
    ----------
    metrics : dict
        Performance metrics dictionary
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    # Select key metrics to plot
    metric_names = ['annualized_return', 'sharpe_ratio', 'calmar_ratio', 'hit_ratio']
    metric_labels = ['Ann. Return', 'Sharpe Ratio', 'Calmar Ratio', 'Hit Ratio']
    
    values = [metrics.get(name, 0.0) for name in metric_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(metric_labels))
    colors = ['steelblue', 'green', 'orange', 'purple']
    bars = ax.bar(x_pos, values, alpha=0.7, color=colors)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        label = f'{val:.2f}' if i < 3 else f'{val:.1%}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics summary plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(
    backtest_results: pd.DataFrame,
    metrics: dict,
    benchmark_returns: Optional[pd.Series] = None,
    factor_scores: Optional[pd.DataFrame] = None,
    returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate all standard plots and save to output directory.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results
    metrics : dict
        Performance metrics
    benchmark_returns : pd.Series, optional
        Benchmark returns
    factor_scores : pd.DataFrame, optional
        Factor scores for decile analysis
    returns : pd.Series, optional
        Returns for decile analysis
    output_dir : Path, optional
        Output directory (default from config)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating plots and saving to {output_dir}")
    
    # Cumulative returns
    plot_cumulative_returns(
        backtest_results,
        benchmark_returns,
        save_path=output_dir / "cumulative_returns.png"
    )
    
    # Drawdown
    plot_drawdown(
        backtest_results,
        save_path=output_dir / "drawdown.png"
    )
    
    # Rolling Sharpe
    plot_rolling_sharpe(
        backtest_results,
        save_path=output_dir / "rolling_sharpe.png"
    )
    
    # Long/short decomposition
    if 'long_value' in backtest_results.columns and 'short_value' in backtest_results.columns:
        plot_long_short_decomposition(
            backtest_results,
            save_path=output_dir / "long_short_decomposition.png"
        )
    
    # Metrics summary
    plot_metrics_summary(
        metrics,
        save_path=output_dir / "metrics_summary.png"
    )
    
    # Factor deciles (if data provided)
    if factor_scores is not None and returns is not None:
        for factor_name in factor_scores.columns:
            try:
                plot_factor_deciles(
                    factor_scores[factor_name],
                    returns,
                    save_path=output_dir / f"factor_deciles_{factor_name}.png",
                    title=f"{factor_name.replace('_', ' ').title()} Decile Portfolio Returns"
                )
            except Exception as e:
                logger.warning(f"Error plotting deciles for {factor_name}: {e}")
    
    logger.info("All plots generated successfully")

