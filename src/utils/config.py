"""
Configuration module for the multi-factor equity strategy.

This module contains all configuration parameters, paths, and constants
used throughout the project.
"""

from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, FIGURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    
    # Date range
    start_date: str = "2018-01-01"
    end_date: str = "2026-02-03"  # Extended to today (8 years total)
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # "monthly" or "quarterly"
    rebalance_day: int = -1  # -1 for last trading day, or specific day of month
    
    # Portfolio construction
    long_pct: float = 0.20  # Top 20% for long leg
    short_pct: float = 0.20  # Bottom 20% for short leg
    weighting_scheme: str = "equal"  # "equal" or "score_proportional"
    max_weight: float = 0.03  # Maximum weight per stock (3%)
    
    # Transaction costs
    commission_rate: float = 0.0010  # 10 bps per side
    slippage_rate: float = 0.0010  # 10 bps slippage
    
    # Constraints
    max_sector_exposure: float = 0.10  # Max net sector exposure (±10%)
    beta_neutral: bool = False  # Whether to enforce beta neutrality
    min_price: float = 1.0  # Minimum stock price filter
    min_history_months: int = 12  # Minimum history required
    
    # Initial capital
    initial_capital: float = 1000000.0  # $1M


@dataclass
class FactorConfig:
    """Factor construction configuration."""
    
    # Value factor
    value_btm_weight: float = 0.5
    value_ep_weight: float = 0.5
    value_winsorize_pct: tuple = (0.01, 0.99)  # 1st to 99th percentile
    
    # Momentum factor
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1  # Skip most recent month
    momentum_winsorize_pct: tuple = (0.01, 0.99)
    
    # Quality factor
    quality_roe_weight: float = 1.0
    quality_leverage_weight: float = -1.0
    quality_stability_weight: float = 1.0
    quality_eps_stability_quarters: int = 8
    quality_winsorize_pct: tuple = (0.01, 0.99)
    
    # Size factor (optional)
    size_weight: float = 0.0  # Set to 0 to exclude, >0 to include
    
    # Composite score
    use_regression_weights: bool = False  # Use regression-based factor weights
    regression_lookback_months: int = 24  # Lookback for regression weights
    factor_weights: Dict[str, float] = None  # Manual weights, None = equal weight
    
    # Standardization
    standardization_method: str = "zscore"  # "zscore" or "rank"
    winsorize_enabled: bool = True


@dataclass
class DataConfig:
    """Data source and processing configuration."""
    
    # Data source
    data_source: str = "yahoo"  # "yahoo", "csv", or "alpha_vantage"
    api_key: str = None  # For Alpha Vantage if used
    
    # Universe
    universe_file: str = "sp500_constituents.csv"
    universe_type: str = "sp500"  # "sp500" or "custom"
    
    # Fundamental data lag (days after reporting date before data is tradeable)
    fundamental_lag_days: int = 5
    
    # Data processing
    min_volume: float = 1000000.0  # Minimum daily volume
    forward_fill_fundamentals: bool = True
    outlier_threshold_sigma: float = 5.0  # Z-score threshold for outliers


# Global config instances
BACKTEST_CONFIG = BacktestConfig()
FACTOR_CONFIG = FactorConfig()
DATA_CONFIG = DataConfig()


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of all configurations."""
    return {
        "backtest": {
            "start_date": BACKTEST_CONFIG.start_date,
            "end_date": BACKTEST_CONFIG.end_date,
            "rebalance_frequency": BACKTEST_CONFIG.rebalance_frequency,
            "long_pct": BACKTEST_CONFIG.long_pct,
            "short_pct": BACKTEST_CONFIG.short_pct,
        },
        "factors": {
            "value_weight": FACTOR_CONFIG.value_btm_weight + FACTOR_CONFIG.value_ep_weight,
            "momentum_lookback": FACTOR_CONFIG.momentum_lookback_months,
            "quality_components": 3,
            "size_weight": FACTOR_CONFIG.size_weight,
        },
        "data": {
            "source": DATA_CONFIG.data_source,
            "universe": DATA_CONFIG.universe_type,
        }
    }


