"""
Size factor construction module.

Implements size factor based on market capitalization.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.size")


def compute_size_factor(
    market_cap: pd.Series,
    date: pd.Timestamp
) -> pd.Series:
    """
    Compute size factor z-score for a given date.
    
    Size factor = -z(log(MarketCap))
    Negative because smaller caps should score higher (size premium).
    
    Parameters
    ----------
    market_cap : pd.Series
        Market cap series with (date, ticker) MultiIndex
    date : pd.Timestamp
        Date for which to compute the factor
    
    Returns
    -------
    pd.Series
        Size factor z-scores indexed by ticker (higher = smaller cap)
    """
    # Get most recent market cap for each ticker up to this date
    mc_slice = market_cap.loc[market_cap.index.get_level_values('date') <= date]
    latest_mc = mc_slice.groupby('ticker').last()
    
    # Remove NaN and negative values
    latest_mc = latest_mc[latest_mc > 0].dropna()
    
    if latest_mc.empty:
        return pd.Series(dtype=float, name='size_factor')
    
    # Log transform
    log_mc = np.log(latest_mc)
    
    # Cross-sectional z-score (negative so smaller = higher score)
    if FACTOR_CONFIG.standardization_method == "rank":
        size_score = -(log_mc.rank(pct=True) - 0.5)
    else:  # zscore
        size_score = -pd.Series(
            zscore(log_mc.dropna(), nan_policy='omit'),
            index=log_mc.index
        ).fillna(0)
    
    return size_score.rename('size_factor')


def compute_size_factor_panel(
    market_cap: pd.Series,
    rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Compute size factor for all rebalancing dates.
    
    Parameters
    ----------
    market_cap : pd.Series
        Market cap series
    rebalance_dates : pd.DatetimeIndex
        Rebalancing dates
    
    Returns
    -------
    pd.DataFrame
        Size factor scores with (date, ticker) MultiIndex
    """
    logger.info(f"Computing size factor for {len(rebalance_dates)} dates")
    
    all_scores = []
    
    for date in rebalance_dates:
        try:
            scores = compute_size_factor(market_cap, date)
            scores = scores.to_frame()
            scores['date'] = date
            scores = scores.reset_index().set_index(['date', 'ticker'])
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"Error computing size factor for {date}: {e}")
            continue
    
    if not all_scores:
        raise ValueError("No size factor scores computed")
    
    result = pd.concat(all_scores, axis=0).sort_index()
    logger.info(f"Computed size factor: {len(result)} records")
    
    return result


