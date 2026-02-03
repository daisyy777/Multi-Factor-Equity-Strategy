"""Backtester module."""

from .engine import BacktestEngine, get_rebalance_dates
from .metrics import (
    compute_metrics,
    format_metrics_report,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    information_ratio,
    max_drawdown,
    hit_ratio,
    calmar_ratio,
)

__all__ = [
    "BacktestEngine",
    "get_rebalance_dates",
    "compute_metrics",
    "format_metrics_report",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "information_ratio",
    "max_drawdown",
    "hit_ratio",
    "calmar_ratio",
]


