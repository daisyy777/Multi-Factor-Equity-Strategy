"""
Configuration module for the multi-factor equity strategy.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, FIGURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""

    # Date range
    start_date: str = "2018-01-01"
    end_date: str = "2026-02-03"

    # Rebalancing
    rebalance_frequency: str = "monthly"  # "monthly" or "quarterly"
    rebalance_day: int = -1               # -1 = last trading day of period

    # Portfolio construction
    long_pct: float = 0.20   # Top 20% → long leg
    short_pct: float = 0.20  # Bottom 20% → short leg
    weighting_scheme: str = "equal"  # "equal" or "score_proportional"
    max_weight: float = 0.03          # Max weight per stock (3%)

    # Turnover constraint  (one-way, per rebalance)
    # Set to 1.0 to disable.  0.30 → max 30% of NAV traded each rebalance.
    max_turnover_per_rebalance: float = 0.30

    # Transaction costs
    commission_rate: float = 0.0010  # 10 bps per side
    slippage_rate: float = 0.0010    # 10 bps slippage

    # Constraints
    max_sector_exposure: float = 0.10
    beta_neutral: bool = False
    min_price: float = 1.0
    min_history_months: int = 12

    # Initial capital
    initial_capital: float = 1_000_000.0


@dataclass
class FactorConfig:
    """Factor construction configuration."""

    # Value factor
    value_btm_weight: float = 0.5
    value_ep_weight: float = 0.5
    value_winsorize_pct: tuple = (0.01, 0.99)

    # Momentum factor
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    momentum_winsorize_pct: tuple = (0.01, 0.99)

    # Quality factor
    quality_roe_weight: float = 1.0
    quality_leverage_weight: float = -1.0
    quality_stability_weight: float = 1.0
    quality_eps_stability_quarters: int = 8
    quality_winsorize_pct: tuple = (0.01, 0.99)

    # Size factor (optional — set weight to 0 to exclude)
    size_weight: float = 0.0

    # Composite score
    use_regression_weights: bool = False
    regression_lookback_months: int = 24
    factor_weights: Optional[Dict[str, float]] = None  # None → equal weight

    # Standardization
    standardization_method: str = "zscore"  # "zscore" or "rank"
    winsorize_enabled: bool = True


@dataclass
class DataConfig:
    """Data source and processing configuration."""

    # Data source
    data_source: str = "yahoo"
    api_key: Optional[str] = None

    # Universe
    universe_file: str = "sp500_constituents.csv"
    universe_type: str = "sp500"

    # Set to True to download historical S&P 500 constituents (eliminates
    # survivorship bias).  Falls back to static list on download failure.
    use_historical_universe: bool = True

    # Fundamental data reporting lag.
    # Data available_date = quarter_end_date + fundamental_report_lag_days.
    # 60 days is conservative but standard in academic factor research.
    # (SEC 10-Q deadline: 40-45 days; using 60 adds a safety margin.)
    fundamental_report_lag_days: int = 60

    # Data processing
    min_volume: float = 1_000_000.0
    forward_fill_fundamentals: bool = True
    outlier_threshold_sigma: float = 5.0


# Global config instances
BACKTEST_CONFIG = BacktestConfig()
FACTOR_CONFIG = FactorConfig()
DATA_CONFIG = DataConfig()


def get_config_summary() -> Dict[str, Any]:
    """Return a summary dict of all active configurations."""
    return {
        "backtest": {
            "start_date": BACKTEST_CONFIG.start_date,
            "end_date": BACKTEST_CONFIG.end_date,
            "rebalance_frequency": BACKTEST_CONFIG.rebalance_frequency,
            "long_pct": BACKTEST_CONFIG.long_pct,
            "short_pct": BACKTEST_CONFIG.short_pct,
            "max_turnover_per_rebalance": BACKTEST_CONFIG.max_turnover_per_rebalance,
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
            "use_historical_universe": DATA_CONFIG.use_historical_universe,
            "fundamental_report_lag_days": DATA_CONFIG.fundamental_report_lag_days,
        },
    }
