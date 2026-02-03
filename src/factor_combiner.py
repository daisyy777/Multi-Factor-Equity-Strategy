"""
Factor combination module.

Combines individual factor scores into a composite multi-factor score,
with optional regression-based dynamic weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression

from .utils.config import FACTOR_CONFIG, BACKTEST_CONFIG
from .utils.logging import get_logger

logger = get_logger("factor_combiner")


def compute_regression_weights(
    factor_scores: pd.DataFrame,
    returns: pd.Series,
    lookback_months: int = 24
) -> Dict[str, float]:
    """
    Compute factor weights using rolling regression of next-period returns.
    
    r_{i,t+1} = alpha + Σ_f beta_f * z_{i,f,t} + epsilon
    
    Parameters
    ----------
    factor_scores : pd.DataFrame
        Factor scores with (date, ticker) MultiIndex, columns are factor names
    returns : pd.Series
        Next-period returns with (date, ticker) MultiIndex
    lookback_months : int
        Number of months to use for regression
    
    Returns
    -------
    Dict[str, float]
        Factor weights normalized to sum to 1
    """
    logger.info("Computing regression-based factor weights")
    
    # Align factor scores and returns
    aligned_scores, aligned_returns = factor_scores.align(returns, join='inner')
    
    # Get dates
    dates = aligned_scores.index.get_level_values('date').unique().sort_values()
    
    if len(dates) < lookback_months:
        logger.warning(f"Insufficient history for regression weights. Using equal weights.")
        factor_names = factor_scores.columns.tolist()
        return {name: 1.0 / len(factor_names) for name in factor_names}
    
    # Use recent lookback period
    recent_dates = dates[-lookback_months:]
    
    # Prepare data for regression
    X_list = []
    y_list = []
    
    for date in recent_dates:
        date_scores = aligned_scores.loc[(date, slice(None)), :]
        date_returns = aligned_returns.loc[(date, slice(None))]
        
        # Find common tickers
        common_tickers = date_scores.index.get_level_values('ticker').intersection(
            date_returns.index.get_level_values('ticker')
        )
        
        if len(common_tickers) < 10:  # Need sufficient observations
            continue
        
        X_date = date_scores.loc[(date, common_tickers), :].values
        y_date = date_returns.loc[(date, common_tickers)].values
        
        X_list.append(X_date)
        y_list.append(y_date)
    
    if not X_list:
        logger.warning("No data for regression. Using equal weights.")
        factor_names = factor_scores.columns.tolist()
        return {name: 1.0 / len(factor_names) for name in factor_names}
    
    # Combine all data
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Remove NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 50:  # Need sufficient observations
        logger.warning("Insufficient data for regression. Using equal weights.")
        factor_names = factor_scores.columns.tolist()
        return {name: 1.0 / len(factor_names) for name in factor_names}
    
    # Fit regression
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        # Get factor weights (coefficients)
        weights = model.coef_
        
        # Normalize to sum to 1 (absolute values)
        weights = np.abs(weights)
        weights = weights / weights.sum()
        
        # Create dictionary
        factor_names = factor_scores.columns.tolist()
        weight_dict = {name: float(w) for name, w in zip(factor_names, weights)}
        
        logger.info(f"Regression weights: {weight_dict}")
        return weight_dict
        
    except Exception as e:
        logger.warning(f"Error in regression: {e}. Using equal weights.")
        factor_names = factor_scores.columns.tolist()
        return {name: 1.0 / len(factor_names) for name in factor_names}


def combine_factors(
    factor_scores: pd.DataFrame,
    factor_weights: Optional[Dict[str, float]] = None,
    returns: Optional[pd.Series] = None
) -> pd.Series:
    """
    Combine individual factor scores into a composite score.
    
    Parameters
    ----------
    factor_scores : pd.DataFrame
        Factor scores with (date, ticker) MultiIndex, columns are factor names
    factor_weights : Dict[str, float], optional
        Manual factor weights. If None, uses config or regression-based weights
    returns : pd.Series, optional
        Next-period returns for regression-based weighting
    
    Returns
    -------
    pd.Series
        Composite factor score with (date, ticker) MultiIndex
    """
    # Determine weights
    if factor_weights is not None:
        weights = factor_weights
    elif FACTOR_CONFIG.use_regression_weights and returns is not None:
        weights = compute_regression_weights(factor_scores, returns, FACTOR_CONFIG.regression_lookback_months)
    elif FACTOR_CONFIG.factor_weights is not None:
        weights = FACTOR_CONFIG.factor_weights
    else:
        # Equal weights
        factor_names = factor_scores.columns.tolist()
        weights = {name: 1.0 / len(factor_names) for name in factor_names}
        logger.info(f"Using equal factor weights: {weights}")
    
    # Apply size factor weight if configured
    if 'size_factor' in factor_scores.columns and FACTOR_CONFIG.size_weight == 0:
        # Remove size from equal weighting
        weights_no_size = {k: v for k, v in weights.items() if k != 'size_factor'}
        if weights_no_size:
            total_weight = sum(weights_no_size.values())
            weights = {k: v / total_weight for k, v in weights_no_size.items()}
    
    # Compute weighted sum
    # Use the index from factor_scores (should be (date, ticker) MultiIndex)
    composite = pd.Series(0.0, index=factor_scores.index)
    
    total_weight = 0.0
    for factor_name, weight in weights.items():
        if factor_name in factor_scores.columns:
            # Apply size weight multiplier if size factor
            if factor_name == 'size_factor':
                weight = weight * FACTOR_CONFIG.size_weight
            
            factor_values = factor_scores[factor_name]
            # Only add non-NaN values
            non_na_mask = factor_values.notna()
            composite.loc[non_na_mask] = composite.loc[non_na_mask] + weight * factor_values.loc[non_na_mask]
            total_weight += weight * non_na_mask.astype(float)
    
    # Normalize by total weight (handle cases where some factors are missing)
    composite = composite / total_weight.replace(0, np.nan)
    
    # Set to NaN where no factors were available
    composite[total_weight == 0] = np.nan
    
    return composite.rename('composite_score')


def compute_composite_scores_panel(
    factor_scores: pd.DataFrame,
    factor_weights: Optional[Dict[str, float]] = None,
    returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute composite scores for all dates.
    
    Parameters
    ----------
    factor_scores : pd.DataFrame
        Factor scores with (date, ticker) MultiIndex
    factor_weights : Dict[str, float], optional
        Manual factor weights
    returns : pd.Series, optional
        Returns for regression-based weighting
    
    Returns
    -------
    pd.DataFrame
        Composite scores with (date, ticker) MultiIndex
    """
    logger.info("Computing composite factor scores")
    
    # Group by date and compute composite for each date
    all_composites = []
    dates = factor_scores.index.get_level_values('date').unique()
    
    for date in dates:
        date_factors = factor_scores.loc[(date, slice(None)), :]
        date_returns = None
        if returns is not None:
            # Get next period returns (if available)
            try:
                next_date_idx = dates.get_loc(date) + 1
                if next_date_idx < len(dates):
                    next_date = dates[next_date_idx]
                    date_returns = returns.loc[(next_date, slice(None))]
            except:
                pass
        
        composite = combine_factors(date_factors, factor_weights, date_returns)
        composite = composite.to_frame()
        
        # The composite Series already has (date, ticker) MultiIndex from date_factors
        # Just ensure it's properly structured - no need to add date column
        if not isinstance(composite.index, pd.MultiIndex) or 'date' not in composite.index.names:
            # If index doesn't have date, add it
            composite = composite.reset_index()
            composite['date'] = date
            composite = composite.set_index(['date', 'ticker'])
        # Otherwise, index is already correct (date, ticker) MultiIndex
        all_composites.append(composite)
    
    result = pd.concat(all_composites, axis=0).sort_index()
    logger.info(f"Computed composite scores: {len(result)} records")
    
    return result


