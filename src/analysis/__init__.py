"""
Analysis package: plotting, performance reports, and IC analysis.
"""

from .plotting import generate_all_plots
from .performance_report import generate_performance_report
from .ic_analysis import (
    compute_ic_panel,
    compute_forward_returns,
    summarise_ic,
    plot_ic_tearsheet,
)

__all__ = [
    "generate_all_plots",
    "generate_performance_report",
    "compute_ic_panel",
    "compute_forward_returns",
    "summarise_ic",
    "plot_ic_tearsheet",
]
