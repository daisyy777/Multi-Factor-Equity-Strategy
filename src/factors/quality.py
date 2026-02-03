"""
Quality factor construction module.

Implements quality-based factors including ROE, leverage, and earnings stability.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.quality")


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series to clip extreme values."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def compute_roe(
    net_income: pd.Series,
    book_value: pd.Series
) -> pd.Series:
    """
    Compute Return on Equity.
    
    ROE = NetIncome / BookValueOfEquity
    
    Parameters
    ----------
    net_income : pd.Series
        Net income with (date, ticker) MultiIndex
    book_value : pd.Series
        Book value of equity with (date, ticker) MultiIndex
    
    Returns
    -------
    pd.Series
        ROE ratio
    """
    # Align indices
    aligned_ni, aligned_bv = net_income.align(book_value, join='inner')
    
    # Compute ratio
    roe = aligned_ni / aligned_bv
    
    # Remove infinite values
    roe = roe.replace([np.inf, -np.inf], np.nan)
    
    return roe.rename('roe')


def compute_leverage(
    total_liabilities: pd.Series,
    total_assets: pd.Series
) -> pd.Series:
    """
    Compute Leverage ratio.
    
    Leverage = TotalLiabilities / TotalAssets
    
    Parameters
    ----------
    total_liabilities : pd.Series
        Total liabilities with (date, ticker) MultiIndex
    total_assets : pd.Series
        Total assets with (date, ticker) MultiIndex
    
    Returns
    -------
    pd.Series
        Leverage ratio
    """
    # Align indices
    aligned_liab, aligned_assets = total_liabilities.align(total_assets, join='inner')
    
    # Compute ratio
    leverage = aligned_liab / aligned_assets
    
    # Remove infinite values
    leverage = leverage.replace([np.inf, -np.inf], np.nan)
    
    return leverage.rename('leverage')


def compute_earnings_stability(
    eps: pd.Series,
    date: pd.Timestamp,
    quarters: int = 8
) -> pd.Series:
    """
    Compute earnings stability (negative of variance of EPS).
    
    Higher stability (lower variance) is better, so we use negative variance.
    
    Parameters
    ----------
    eps : pd.Series
        EPS with (date, ticker) MultiIndex
    date : pd.Timestamp
        Date for which to compute stability
    quarters : int
        Number of quarters to look back
    
    Returns
    -------
    pd.Series
        Earnings stability score (negative variance) indexed by ticker
    """
    # Get data up to this date
    eps_slice = eps.loc[eps.index.get_level_values('date') <= date]
    
    # Resample to quarterly if needed (assuming data is already quarterly)
    eps_df = eps_slice.unstack('ticker')
    
    # Get last N quarters
    if len(eps_df) >= quarters:
        recent_eps = eps_df.iloc[-quarters:]
    else:
        recent_eps = eps_df
    
    # Compute variance for each ticker
    eps_variance = recent_eps.var()
    
    # Use negative variance (lower variance = higher quality)
    stability = -eps_variance
    
    return stability.rename('earnings_stability')


def compute_quality_factor(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    date: pd.Timestamp
) -> pd.Series:
    """
    Compute quality factor z-score for a given date.
    
    Combines ROE, leverage (negative), and earnings stability.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with (date, ticker) MultiIndex
    fundamental_data : pd.DataFrame
        Fundamental data with (date, ticker) MultiIndex
    market_cap : pd.Series
        Market cap series (not used directly but kept for consistency)
    date : pd.Timestamp
        Date for which to compute the factor
    
    Returns
    -------
    pd.Series
        Quality factor z-scores indexed by ticker
    """
    # Get data up to this date
    fund_slice = fundamental_data.loc[fundamental_data.index.get_level_values('date') <= date]
    
    # Get most recent data for each ticker
    latest_net_income = fund_slice.groupby('ticker')['net_income'].last() if 'net_income' in fund_slice.columns else pd.Series(dtype=float)
    latest_eps = fund_slice.groupby('ticker')['eps'].last() if 'eps' in fund_slice.columns else pd.Series(dtype=float)
    
    # Get book value
    if 'book_value_per_share' in fund_slice.columns and 'shares_outstanding' in fund_slice.columns:
        latest_bv_ps = fund_slice.groupby('ticker')['book_value_per_share'].last()
        latest_shares = fund_slice.groupby('ticker')['shares_outstanding'].last()
        latest_book_value = latest_bv_ps * latest_shares
    else:
        latest_book_value = pd.Series(dtype=float)
    
    # Get total assets and liabilities
    latest_assets = fund_slice.groupby('ticker')['total_assets'].last() if 'total_assets' in fund_slice.columns else pd.Series(dtype=float)
    latest_liabilities = fund_slice.groupby('ticker')['total_liabilities'].last() if 'total_liabilities' in fund_slice.columns else pd.Series(dtype=float)
    
    # Compute ROE
    roe = compute_roe(latest_net_income, latest_book_value)
    
    # Compute leverage
    leverage = compute_leverage(latest_liabilities, latest_assets)
    
    # Compute earnings stability
    if 'eps' in fundamental_data.columns:
        eps_series = fundamental_data['eps']
        stability = compute_earnings_stability(eps_series, date, FACTOR_CONFIG.quality_eps_stability_quarters)
    else:
        stability = pd.Series(dtype=float)
    
    # Find common tickers across all components
    components = [roe, leverage, stability]
    components = [c.dropna() for c in components if not c.empty]
    
    if not components:
        # Return empty Series with proper name
        return pd.Series(dtype=float, name='quality_factor')
    
    common_tickers = components[0].index
    for c in components[1:]:
        common_tickers = common_tickers.intersection(c.index)
    
    if len(common_tickers) == 0:
        # Return empty Series with proper name
        return pd.Series(dtype=float, name='quality_factor')
    
    # Align all components - ensure we have valid indices
    roe_aligned = roe.loc[common_tickers] if not roe.empty and len(common_tickers) > 0 else pd.Series(index=common_tickers, dtype=float)
    leverage_aligned = leverage.loc[common_tickers] if not leverage.empty and len(common_tickers) > 0 else pd.Series(index=common_tickers, dtype=float)
    stability_aligned = stability.loc[common_tickers] if not stability.empty and len(common_tickers) > 0 else pd.Series(index=common_tickers, dtype=float)
    
    # Fill NaN with median if needed
    if roe_aligned.isna().any():
        roe_aligned = roe_aligned.fillna(roe_aligned.median())
    if leverage_aligned.isna().any():
        leverage_aligned = leverage_aligned.fillna(leverage_aligned.median())
    if stability_aligned.isna().any():
        stability_aligned = stability_aligned.fillna(stability_aligned.median())
    
    # Winsorize if enabled
    if FACTOR_CONFIG.winsorize_enabled:
        roe_aligned = winsorize(roe_aligned, FACTOR_CONFIG.quality_winsorize_pct[0], FACTOR_CONFIG.quality_winsorize_pct[1])
        leverage_aligned = winsorize(leverage_aligned, FACTOR_CONFIG.quality_winsorize_pct[0], FACTOR_CONFIG.quality_winsorize_pct[1])
        stability_aligned = winsorize(stability_aligned, FACTOR_CONFIG.quality_winsorize_pct[0], FACTOR_CONFIG.quality_winsorize_pct[1])
    
    # Cross-sectional z-score each component
    if FACTOR_CONFIG.standardization_method == "rank":
        roe_z = roe_aligned.rank(pct=True) - 0.5
        leverage_z = leverage_aligned.rank(pct=True) - 0.5
        stability_z = stability_aligned.rank(pct=True) - 0.5
    else:  # zscore
        roe_z = pd.Series(zscore(roe_aligned.dropna(), nan_policy='omit'), index=roe_aligned.index).fillna(0)
        leverage_z = pd.Series(zscore(leverage_aligned.dropna(), nan_policy='omit'), index=leverage_aligned.index).fillna(0)
        stability_z = pd.Series(zscore(stability_aligned.dropna(), nan_policy='omit'), index=stability_aligned.index).fillna(0)
    
    # Combine into quality score
    # ROE and stability are positive contributors, leverage is negative
    quality_score = (
        FACTOR_CONFIG.quality_roe_weight * roe_z +
        FACTOR_CONFIG.quality_leverage_weight * leverage_z +
        FACTOR_CONFIG.quality_stability_weight * stability_z
    ) / (abs(FACTOR_CONFIG.quality_roe_weight) + abs(FACTOR_CONFIG.quality_leverage_weight) + abs(FACTOR_CONFIG.quality_stability_weight))
    
    # Re-standardize the composite
    if FACTOR_CONFIG.standardization_method == "rank":
        quality_score = quality_score.rank(pct=True) - 0.5
    else:
        quality_score = pd.Series(zscore(quality_score.dropna(), nan_policy='omit'), index=quality_score.index).fillna(0)
    
    return quality_score.rename('quality_factor')


def compute_quality_factor_panel(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Compute quality factor for all rebalancing dates.
    
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
        Quality factor scores with (date, ticker) MultiIndex
    """
    logger.info(f"Computing quality factor for {len(rebalance_dates)} dates")
    
    all_scores = []
    
    for date in rebalance_dates:
        try:
            scores = compute_quality_factor(price_data, fundamental_data, market_cap, date)
            
            # Skip if scores is empty
            if scores.empty:
                logger.debug(f"No quality factor data for {date}")
                continue
                
            scores = scores.to_frame()
            scores['date'] = date
            scores = scores.reset_index().set_index(['date', 'ticker'])
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"Error computing quality factor for {date}: {e}")
            continue
    
    if not all_scores:
        logger.warning("No quality factor scores computed for any date. This may be due to missing fundamental data.")
        # Return empty DataFrame with correct structure
        empty_idx = pd.MultiIndex.from_product([rebalance_dates, []], names=['date', 'ticker'])
        return pd.DataFrame(index=empty_idx, columns=['quality_factor'])
    
    result = pd.concat(all_scores, axis=0).sort_index()
    logger.info(f"Computed quality factor: {len(result)} records")
    
    return result


