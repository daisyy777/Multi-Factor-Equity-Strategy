"""
Momentum factor construction module.

Implements 12-1 month momentum factor (excluding most recent month
to avoid short-term reversal).
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.momentum")


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series to clip extreme values."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def compute_momentum(
    price_data: pd.DataFrame,
    date: pd.Timestamp,
    lookback_months: int = 12,
    skip_months: int = 1
) -> pd.Series:
    """
    Compute 12-1 month momentum (excluding most recent month).
    
    Mom = (Price_{t-1m} - Price_{t-12m}) / Price_{t-12m}
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex, must have 'adj_close'
    date : pd.Timestamp
        Date for which to compute momentum
    lookback_months : int
        Lookback period in months (default 12)
    skip_months : int
        Number of months to skip at the end (default 1)
    
    Returns
    -------
    pd.Series
        Momentum returns indexed by ticker
    """
    # Get data up to this date
    price_slice = price_data.loc[price_data.index.get_level_values('date') <= date]
    
    # Resample to monthly (end of month)
    prices = price_slice['adj_close'].unstack('ticker')
    monthly_prices = prices.resample('M').last()
    
    if len(monthly_prices) < lookback_months + skip_months:
        # Not enough history
        return pd.Series(dtype=float, name='momentum')
    
    # Get price at t-skip_months and t-lookback_months-skip_months
    t_skip = monthly_prices.index[-1 - skip_months]
    t_lookback = monthly_prices.index[-1 - lookback_months - skip_months]
    
    price_t_skip = monthly_prices.loc[t_skip]
    price_t_lookback = monthly_prices.loc[t_lookback]
    
    # Compute momentum
    momentum = (price_t_skip - price_t_lookback) / price_t_lookback
    
    # Remove infinite and NaN values
    momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()
    
    return momentum.rename('momentum')


def compute_momentum_factor(
    price_data: pd.DataFrame,
    date: pd.Timestamp
) -> pd.Series:
    """
    Compute momentum factor z-score for a given date.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex
    date : pd.Timestamp
        Date for which to compute the factor
    
    Returns
    -------
    pd.Series
        Momentum factor z-scores indexed by ticker
    """
    # Compute raw momentum
    momentum = compute_momentum(
        price_data,
        date,
        FACTOR_CONFIG.momentum_lookback_months,
        FACTOR_CONFIG.momentum_skip_months
    )
    
    if momentum.empty:
        return pd.Series(dtype=float, name='momentum_factor')
    
    # Winsorize if enabled
    if FACTOR_CONFIG.winsorize_enabled:
        momentum = winsorize(
            momentum,
            FACTOR_CONFIG.momentum_winsorize_pct[0],
            FACTOR_CONFIG.momentum_winsorize_pct[1]
        )
    
    # Cross-sectional z-score
    if FACTOR_CONFIG.standardization_method == "rank":
        momentum_z = momentum.rank(pct=True) - 0.5
    else:  # zscore
        momentum_z = pd.Series(
            zscore(momentum.dropna(), nan_policy='omit'),
            index=momentum.index
        ).fillna(0)
    
    return momentum_z.rename('momentum_factor')


def compute_momentum_factor_panel(
    price_data: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Compute momentum factor for all rebalancing dates.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex
    rebalance_dates : pd.DatetimeIndex
        Rebalancing dates
    
    Returns
    -------
    pd.DataFrame
        Momentum factor scores with (date, ticker) MultiIndex
    """
    logger.info(f"Computing momentum factor for {len(rebalance_dates)} dates")
    
    all_scores = []
    
    for date in rebalance_dates:
        try:
            scores = compute_momentum_factor(price_data, date)
            
            if scores.empty:
                continue
            
            scores = scores.to_frame()
            scores['date'] = date
            
            # Reset index - scores.index should be ticker names
            scores = scores.reset_index()
            
            # The index column should be 'ticker' (from Series index)
            # If not, rename it
            if 'index' in scores.columns and 'ticker' not in scores.columns:
                scores = scores.rename(columns={'index': 'ticker'})
            elif 'ticker' not in scores.columns:
                # If no ticker column, the index values are tickers
                scores['ticker'] = scores.index
                scores = scores.reset_index(drop=True)
            
            scores = scores.set_index(['date', 'ticker'])
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"Error computing momentum factor for {date}: {e}")
            continue
    
    if not all_scores:
        raise ValueError("No momentum factor scores computed")
    
    result = pd.concat(all_scores, axis=0).sort_index()
    logger.info(f"Computed momentum factor: {len(result)} records")
    
    return result


