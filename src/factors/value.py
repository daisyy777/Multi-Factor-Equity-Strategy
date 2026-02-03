"""
Value factor construction module.

Implements value-based factors including Book-to-Market (BTM) and
Earnings-to-Price (E/P) ratios.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.value")


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a series to clip extreme values.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    lower : float
        Lower percentile (default 0.01)
    upper : float
        Upper percentile (default 0.99)
    
    Returns
    -------
    pd.Series
        Winsorized series
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def compute_btm(
    market_cap: pd.Series,
    book_value: pd.Series
) -> pd.Series:
    """
    Compute Book-to-Market ratio.
    
    BTM = BookValueOfEquity / MarketCap
    
    Parameters
    ----------
    market_cap : pd.Series
        Market capitalization with (date, ticker) MultiIndex
    book_value : pd.Series
        Book value of equity with (date, ticker) MultiIndex
    
    Returns
    -------
    pd.Series
        Book-to-Market ratio
    """
    # Align indices
    aligned_mc, aligned_bv = market_cap.align(book_value, join='inner')
    
    # Compute ratio
    btm = aligned_bv / aligned_mc
    
    # Remove infinite and negative values
    btm = btm.replace([np.inf, -np.inf], np.nan)
    btm = btm[btm > 0]
    
    return btm.rename('btm')


def compute_ep(
    price: pd.Series,
    eps: pd.Series
) -> pd.Series:
    """
    Compute Earnings-to-Price ratio.
    
    E/P = EarningsPerShare / PricePerShare
    
    Parameters
    ----------
    price : pd.Series
        Stock price (adjusted close) with (date, ticker) MultiIndex
    eps : pd.Series
        Earnings per share with (date, ticker) MultiIndex
    
    Returns
    -------
    pd.Series
        Earnings-to-Price ratio
    """
    # Align indices
    aligned_price, aligned_eps = price.align(eps, join='inner')
    
    # Compute ratio
    ep = aligned_eps / aligned_price
    
    # Remove infinite values
    ep = ep.replace([np.inf, -np.inf], np.nan)
    
    return ep.rename('ep')


def compute_value_factor(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    date: pd.Timestamp
) -> pd.Series:
    """
    Compute value factor z-score for a given date.
    
    Combines BTM and E/P into a composite value score.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex
    fundamental_data : pd.DataFrame
        Fundamental data with (date, ticker) MultiIndex
    market_cap : pd.Series
        Market cap series
    date : pd.Timestamp
        Date for which to compute the factor
    
    Returns
    -------
    pd.Series
        Value factor z-scores indexed by ticker
    """
    # Get data up to this date
    price_slice = price_data.loc[price_data.index.get_level_values('date') <= date]
    fund_slice = fundamental_data.loc[fundamental_data.index.get_level_values('date') <= date]
    mc_slice = market_cap.loc[market_cap.index.get_level_values('date') <= date]
    
    # Get most recent data for each ticker
    latest_prices = price_slice.groupby('ticker')['adj_close'].last()
    latest_mc = mc_slice.groupby('ticker').last()
    
    # Get book value (book_value_per_share * shares_outstanding, or total book value)
    if 'book_value_per_share' in fund_slice.columns and 'shares_outstanding' in fund_slice.columns:
        latest_bv_ps = fund_slice.groupby('ticker')['book_value_per_share'].last()
        latest_shares = fund_slice.groupby('ticker')['shares_outstanding'].last()
        latest_book_value = latest_bv_ps * latest_shares
    else:
        # Fallback: estimate from available data
        logger.warning("Book value data incomplete. Using simplified estimate.")
        latest_book_value = pd.Series(index=latest_mc.index, dtype=float)
        latest_book_value[:] = np.nan
    
    # Get EPS
    if 'eps' in fund_slice.columns:
        latest_eps = fund_slice.groupby('ticker')['eps'].last()
    else:
        latest_eps = pd.Series(index=latest_prices.index, dtype=float)
        latest_eps[:] = np.nan
    
    # Compute BTM
    btm = compute_btm(latest_mc, latest_book_value)
    
    # Compute E/P
    ep = compute_ep(latest_prices, latest_eps)
    
    # If both BTM and E/P are empty (no fundamental data), use price-based proxy
    if btm.empty and ep.empty:
        # Use inverse of market cap as a simple value proxy (smaller cap = higher value)
        # This is a very simplified approach when fundamental data is unavailable
        value_proxy = 1.0 / latest_mc
        value_proxy = value_proxy.replace([np.inf, -np.inf], np.nan).dropna()
        if value_proxy.empty:
            return pd.Series(dtype=float, name='value_factor')
        # Standardize
        if FACTOR_CONFIG.standardization_method == "rank":
            value_score = value_proxy.rank(pct=True) - 0.5
        else:
            value_score = pd.Series(
                zscore(value_proxy.dropna(), nan_policy='omit'),
                index=value_proxy.index
            ).fillna(0)
        return value_score.rename('value_factor')
    
    # Align all components
    # If one is empty, use the other
    if btm.empty:
        common_tickers = ep.index
        btm_aligned = pd.Series(index=common_tickers, dtype=float)
        btm_aligned[:] = 0.0  # Neutral value when missing
        ep_aligned = ep.loc[common_tickers]
    elif ep.empty:
        common_tickers = btm.index
        btm_aligned = btm.loc[common_tickers]
        ep_aligned = pd.Series(index=common_tickers, dtype=float)
        ep_aligned[:] = 0.0  # Neutral value when missing
    else:
        common_tickers = btm.index.intersection(ep.index)
        btm_aligned = btm.loc[common_tickers]
        ep_aligned = ep.loc[common_tickers]
    
    if len(common_tickers) == 0:
        return pd.Series(dtype=float, name='value_factor')
    
    # Winsorize if enabled
    if FACTOR_CONFIG.winsorize_enabled:
        btm_aligned = winsorize(
            btm_aligned,
            FACTOR_CONFIG.value_winsorize_pct[0],
            FACTOR_CONFIG.value_winsorize_pct[1]
        )
        ep_aligned = winsorize(
            ep_aligned,
            FACTOR_CONFIG.value_winsorize_pct[0],
            FACTOR_CONFIG.value_winsorize_pct[1]
        )
    
    # Cross-sectional z-score
    if FACTOR_CONFIG.standardization_method == "rank":
        btm_z = btm_aligned.rank(pct=True) - 0.5  # Rank-based, centered at 0
        ep_z = ep_aligned.rank(pct=True) - 0.5
    else:  # zscore
        btm_non_na = btm_aligned.dropna()
        if len(btm_non_na) > 0:
            btm_z_values = zscore(btm_non_na, nan_policy='omit')
            btm_z = pd.Series(btm_z_values, index=btm_non_na.index).reindex(btm_aligned.index).fillna(0)
        else:
            btm_z = pd.Series(0.0, index=btm_aligned.index)
        
        ep_non_na = ep_aligned.dropna()
        if len(ep_non_na) > 0:
            ep_z_values = zscore(ep_non_na, nan_policy='omit')
            ep_z = pd.Series(ep_z_values, index=ep_non_na.index).reindex(ep_aligned.index).fillna(0)
        else:
            ep_z = pd.Series(0.0, index=ep_aligned.index)
    
    # Combine into value score
    value_score = (
        FACTOR_CONFIG.value_btm_weight * btm_z +
        FACTOR_CONFIG.value_ep_weight * ep_z
    ) / (FACTOR_CONFIG.value_btm_weight + FACTOR_CONFIG.value_ep_weight)
    
    # Re-standardize the composite
    if FACTOR_CONFIG.standardization_method == "rank":
        value_score = value_score.rank(pct=True) - 0.5
    else:
        value_non_na = value_score.dropna()
        if len(value_non_na) > 0:
            value_z_values = zscore(value_non_na, nan_policy='omit')
            value_score = pd.Series(value_z_values, index=value_non_na.index).reindex(value_score.index).fillna(0)
        else:
            value_score = pd.Series(0.0, index=value_score.index)
    
    return value_score.rename('value_factor')


def compute_value_factor_panel(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Compute value factor for all rebalancing dates.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex
    fundamental_data : pd.DataFrame
        Fundamental data with (date, ticker) MultiIndex
    market_cap : pd.Series
        Market cap series
    rebalance_dates : pd.DatetimeIndex
        Rebalancing dates
    
    Returns
    -------
    pd.DataFrame
        Value factor scores with (date, ticker) MultiIndex
    """
    logger.info(f"Computing value factor for {len(rebalance_dates)} dates")
    
    all_scores = []
    
    for date in rebalance_dates:
        try:
            scores = compute_value_factor(price_data, fundamental_data, market_cap, date)
            
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
            logger.warning(f"Error computing value factor for {date}: {e}")
            continue
    
    if not all_scores:
        raise ValueError("No value factor scores computed")
    
    result = pd.concat(all_scores, axis=0).sort_index()
    logger.info(f"Computed value factor: {len(result)} records")
    
    return result

