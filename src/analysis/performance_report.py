"""
Performance report generation module.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from ..backtester.metrics import format_metrics_report
from ..utils.config import RESULTS_DIR
from ..utils.logging import get_logger

logger = get_logger("analysis.performance_report")


def generate_performance_report(
    metrics: Dict,
    backtest_results: pd.DataFrame,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive performance report.
    
    Parameters
    ----------
    metrics : Dict
        Performance metrics dictionary
    backtest_results : pd.DataFrame
        Backtest results
    output_path : Path, optional
        Path to save report
    
    Returns
    -------
    str
        Report text
    """
    report = format_metrics_report(metrics)
    
    # Add additional statistics
    report += "\n"
    report += "ADDITIONAL STATISTICS\n"
    report += "=" * 60 + "\n\n"
    
    returns = backtest_results['returns']
    nav = backtest_results['nav']
    
    # Monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    report += f"Best Monthly Return:    {monthly_returns.max():.2%}\n"
    report += f"Worst Monthly Return:   {monthly_returns.min():.2%}\n"
    report += f"Avg Monthly Return:     {monthly_returns.mean():.2%}\n"
    report += f"Monthly Return Std:     {monthly_returns.std():.2%}\n"
    report += "\n"
    
    # Win rate by month
    positive_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    report += f"Positive Months:        {positive_months}/{total_months} ({positive_months/total_months:.1%})\n"
    report += "\n"
    
    # Turnover
    if 'turnover' in backtest_results.columns:
        avg_turnover = backtest_results['turnover'].mean()
        report += f"Average Daily Turnover: {avg_turnover:.2%}\n"
        report += f"Annual Turnover:       {avg_turnover * 252:.1%}\n"
        report += "\n"
    
    report += "=" * 60 + "\n"
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved performance report to {output_path}")
    
    return report


