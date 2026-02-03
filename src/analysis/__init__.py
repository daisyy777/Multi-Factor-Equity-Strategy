"""Analysis module."""

from .plotting import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_long_short_decomposition,
    plot_factor_deciles,
    plot_metrics_summary,
    generate_all_plots,
)
from .performance_report import generate_performance_report

__all__ = [
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_long_short_decomposition",
    "plot_factor_deciles",
    "plot_metrics_summary",
    "generate_all_plots",
    "generate_performance_report",
]


