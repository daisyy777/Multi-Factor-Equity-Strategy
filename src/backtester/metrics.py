"""
Performance metrics module.

Computes comprehensive performance metrics for backtest results.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

from ..utils.logging import get_logger

logger = get_logger("backtester.metrics")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized return (CAGR)."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility."""
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute Sharpe ratio."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute Information ratio."""
    excess_returns = returns - benchmark_returns
    ann_excess_ret = annualized_return(excess_returns, periods_per_year)
    tracking_error = annualized_volatility(excess_returns, periods_per_year)
    if tracking_error == 0:
        return 0.0
    return ann_excess_ret / tracking_error


def max_drawdown(nav: pd.Series) -> Dict[str, float]:
    """
    Compute maximum drawdown.
    
    Returns
    -------
    Dict with 'max_dd', 'max_dd_pct', 'dd_start', 'dd_end', 'dd_duration'
    """
    if len(nav) == 0:
        return {'max_dd': 0.0, 'max_dd_pct': 0.0, 'dd_start': None, 'dd_end': None, 'dd_duration': 0}
    
    # Running maximum
    running_max = nav.expanding().max()
    
    # Drawdown
    drawdown = nav - running_max
    drawdown_pct = (nav - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min()
    
    # Find drawdown period
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.loc[max_dd_idx]
    
    # Find start of drawdown (last time NAV was at peak)
    peak_value = nav.loc[max_dd_idx] - max_dd_value
    peak_idx = nav[nav == peak_value].index
    
    if len(peak_idx) > 0:
        dd_start = peak_idx[-1]
        dd_end = max_dd_idx
        
        # Duration in days
        dd_duration = (dd_end - dd_start).days if isinstance(dd_start, pd.Timestamp) else 0
    else:
        dd_start = None
        dd_end = None
        dd_duration = 0
    
    return {
        'max_dd': float(max_dd),
        'max_dd_pct': float(max_dd_pct),
        'dd_start': dd_start,
        'dd_end': dd_end,
        'dd_duration': dd_duration
    }


def hit_ratio(returns: pd.Series) -> float:
    """Compute hit ratio (percentage of positive returns)."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def calmar_ratio(returns: pd.Series, nav: pd.Series, periods_per_year: int = 252) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    ann_ret = annualized_return(returns, periods_per_year)
    dd_info = max_drawdown(nav)
    max_dd_pct = abs(dd_info['max_dd_pct'])
    if max_dd_pct == 0:
        return 0.0
    return ann_ret / max_dd_pct


def compute_metrics(
    backtest_results: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with columns: ['nav', 'returns', 'long_value', 'short_value', 'turnover']
    benchmark_returns : pd.Series, optional
        Benchmark returns (e.g., S&P 500) for comparison
    risk_free_rate : float
        Risk-free rate for Sharpe ratio
    
    Returns
    -------
    Dict with all performance metrics
    """
    returns = backtest_results['returns']
    nav = backtest_results['nav']
    
    metrics = {
        'annualized_return': annualized_return(returns),
        'annualized_volatility': annualized_volatility(returns),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate),
        'calmar_ratio': calmar_ratio(returns, nav),
        'hit_ratio': hit_ratio(returns),
    }
    
    # Maximum drawdown
    dd_info = max_drawdown(nav)
    metrics.update({
        'max_drawdown': dd_info['max_dd'],
        'max_drawdown_pct': dd_info['max_dd_pct'],
        'drawdown_duration_days': dd_info['dd_duration'],
    })
    
    # Information ratio if benchmark provided
    if benchmark_returns is not None:
        # Align benchmark returns
        aligned_benchmark = benchmark_returns.reindex(returns.index, method='ffill')
        metrics['information_ratio'] = information_ratio(returns, aligned_benchmark)
        metrics['benchmark_annualized_return'] = annualized_return(aligned_benchmark)
        metrics['benchmark_sharpe_ratio'] = sharpe_ratio(aligned_benchmark, risk_free_rate)
    
    # Turnover
    if 'turnover' in backtest_results.columns:
        # Turnover is only non-zero on rebalance dates
        # Annual turnover = average rebalance turnover * rebalances per year
        turnover_series = backtest_results['turnover']
        non_zero_turnover = turnover_series[turnover_series > 0]
        
        if len(non_zero_turnover) > 0:
            avg_rebalance_turnover = non_zero_turnover.mean()
            # Estimate rebalances per year (assuming monthly = 12, quarterly = 4)
            # Count non-zero turnover days to estimate frequency
            rebalance_count = len(non_zero_turnover)
            total_days = len(turnover_series)
            rebalances_per_year = (rebalance_count / total_days) * 252 if total_days > 0 else 12
            
            metrics['avg_daily_turnover'] = turnover_series.mean()
            metrics['annual_turnover'] = avg_rebalance_turnover * rebalances_per_year
        else:
            metrics['avg_daily_turnover'] = 0.0
            metrics['annual_turnover'] = 0.0
    
    # Long/short decomposition
    if 'long_value' in backtest_results.columns and 'short_value' in backtest_results.columns:
        long_values = backtest_results['long_value']
        short_values = backtest_results['short_value']
        
        # Compute returns properly: avoid division by zero
        # Long leg: positive positions, compute returns from value changes
        long_returns = pd.Series(index=long_values.index, dtype=float)
        for i in range(1, len(long_values)):
            prev_val = long_values.iloc[i-1]
            curr_val = long_values.iloc[i]
            if prev_val > 1e-10:  # Avoid division by zero
                long_returns.iloc[i] = (curr_val / prev_val) - 1.0
            elif curr_val > 1e-10:
                long_returns.iloc[i] = 0.0  # New position, no return yet
            else:
                long_returns.iloc[i] = 0.0
        
        # Short leg: negative positions, compute returns from value changes
        # Short value is already positive (abs), so we compute returns normally
        short_returns = pd.Series(index=short_values.index, dtype=float)
        for i in range(1, len(short_values)):
            prev_val = short_values.iloc[i-1]
            curr_val = short_values.iloc[i]
            if prev_val > 1e-10:  # Avoid division by zero
                short_returns.iloc[i] = (curr_val / prev_val) - 1.0
            elif curr_val > 1e-10:
                short_returns.iloc[i] = 0.0  # New position, no return yet
            else:
                short_returns.iloc[i] = 0.0
        
        # Remove NaN and inf values
        long_returns = long_returns.replace([np.inf, -np.inf], np.nan).dropna()
        short_returns = short_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(long_returns) > 0 and long_returns.notna().any():
            metrics['long_leg_annualized_return'] = annualized_return(long_returns)
            metrics['long_leg_sharpe'] = sharpe_ratio(long_returns, risk_free_rate)
        else:
            metrics['long_leg_annualized_return'] = 0.0
            metrics['long_leg_sharpe'] = 0.0
        
        if len(short_returns) > 0 and short_returns.notna().any():
            metrics['short_leg_annualized_return'] = annualized_return(short_returns)
            metrics['short_leg_sharpe'] = sharpe_ratio(short_returns, risk_free_rate)
        else:
            metrics['short_leg_annualized_return'] = 0.0
            metrics['short_leg_sharpe'] = 0.0
    
    # Final NAV
    metrics['final_nav'] = nav.iloc[-1] if len(nav) > 0 else 0.0
    metrics['total_return'] = (nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) > 0 and nav.iloc[0] > 0 else 0.0
    
    return metrics


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable report string."""
    report = "=" * 60 + "\n"
    report += "PERFORMANCE METRICS\n"
    report += "=" * 60 + "\n\n"
    
    # Returns
    report += "Returns:\n"
    report += f"  Annualized Return:     {metrics.get('annualized_return', 0.0):.2%}\n"
    report += f"  Total Return:          {metrics.get('total_return', 0.0):.2%}\n"
    if 'benchmark_annualized_return' in metrics:
        report += f"  Benchmark Return:      {metrics.get('benchmark_annualized_return', 0.0):.2%}\n"
    report += "\n"
    
    # Risk
    report += "Risk:\n"
    report += f"  Annualized Volatility: {metrics.get('annualized_volatility', 0.0):.2%}\n"
    report += f"  Max Drawdown:          {metrics.get('max_drawdown_pct', 0.0):.2%}\n"
    report += f"  Drawdown Duration:     {metrics.get('drawdown_duration_days', 0):.0f} days\n"
    report += "\n"
    
    # Risk-adjusted returns
    report += "Risk-Adjusted Returns:\n"
    report += f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0.0):.2f}\n"
    report += f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0.0):.2f}\n"
    if 'information_ratio' in metrics:
        report += f"  Information Ratio:     {metrics.get('information_ratio', 0.0):.2f}\n"
    report += "\n"
    
    # Other metrics
    report += "Other Metrics:\n"
    report += f"  Hit Ratio:             {metrics.get('hit_ratio', 0.0):.2%}\n"
    report += f"  Annual Turnover:       {metrics.get('annual_turnover', 0.0):.2%}\n"
    report += "\n"
    
    # Long/short decomposition
    if 'long_leg_annualized_return' in metrics:
        report += "Long/Short Decomposition:\n"
        report += f"  Long Leg Return:       {metrics.get('long_leg_annualized_return', 0.0):.2%}\n"
        report += f"  Long Leg Sharpe:       {metrics.get('long_leg_sharpe', 0.0):.2f}\n"
        report += f"  Short Leg Return:      {metrics.get('short_leg_annualized_return', 0.0):.2%}\n"
        report += f"  Short Leg Sharpe:      {metrics.get('short_leg_sharpe', 0.0):.2f}\n"
        report += "\n"
    
    report += f"Final NAV: ${metrics.get('final_nav', 0.0):,.2f}\n"
    report += "=" * 60 + "\n"
    
    return report


