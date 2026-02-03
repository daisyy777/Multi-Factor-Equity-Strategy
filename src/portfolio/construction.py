"""
Portfolio construction module.

Builds long-short portfolios from composite factor scores.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from ..utils.config import BACKTEST_CONFIG
from ..utils.logging import get_logger
from .constraints import apply_constraints

logger = get_logger("portfolio.construction")


def select_long_short_stocks(
    scores: pd.Series,
    long_pct: float,
    short_pct: float
) -> Tuple[List[str], List[str]]:
    """
    Select stocks for long and short legs based on composite scores.
    
    Parameters
    ----------
    scores : pd.Series
        Composite factor scores indexed by ticker
    long_pct : float
        Percentage of stocks to select for long leg (e.g., 0.20 for top 20%)
    short_pct : float
        Percentage of stocks to select for short leg (e.g., 0.20 for bottom 20%)
    
    Returns
    -------
    Tuple[List[str], List[str]]
        (long_tickers, short_tickers)
    """
    # Remove NaN scores
    scores_clean = scores.dropna().sort_values(ascending=False)
    
    if len(scores_clean) == 0:
        logger.warning("All scores are NaN, cannot select stocks")
        return [], []
    
    # CRITICAL: Verify scores index contains ticker strings
    if len(scores_clean) > 0:
        sample_idx = scores_clean.index[0]
        if not isinstance(sample_idx, str):
            logger.error(f"CRITICAL: scores index is not strings in select_long_short_stocks, sample: {scores_clean.index[:3].tolist()}, type: {type(sample_idx)}")
            # This is a critical error - scores should be indexed by ticker names
            # Return empty to avoid creating invalid portfolio
            return [], []
    
    n_stocks = len(scores_clean)
    n_long = max(1, int(n_stocks * long_pct))
    n_short = max(1, int(n_stocks * short_pct))
    
    # Select top for long, bottom for short
    long_tickers = scores_clean.head(n_long).index.tolist()
    short_tickers = scores_clean.tail(n_short).index.tolist()
    
    # Convert index to list if it's not already
    if long_tickers and not isinstance(long_tickers[0], str):
        long_tickers = [str(t) for t in long_tickers]
    if short_tickers and not isinstance(short_tickers[0], str):
        short_tickers = [str(t) for t in short_tickers]
    
    logger.debug(f"Selected {len(long_tickers)} long and {len(short_tickers)} short stocks")
    
    return long_tickers, short_tickers


def compute_equal_weights(
    long_tickers: List[str],
    short_tickers: List[str]
) -> pd.Series:
    """
    Compute equal weights for long and short legs.
    
    Long leg sums to +1, short leg sums to -1.
    
    Parameters
    ----------
    long_tickers : List[str]
        Tickers for long leg
    short_tickers : List[str]
        Tickers for short leg
    
    Returns
    -------
    pd.Series
        Weights indexed by ticker (long positive, short negative)
    """
    weights = pd.Series(0.0, index=long_tickers + short_tickers)
    
    if long_tickers:
        weights.loc[long_tickers] = 1.0 / len(long_tickers)
    
    if short_tickers:
        weights.loc[short_tickers] = -1.0 / len(short_tickers)
    
    # Verify dollar-neutrality: long sum should be +1, short sum should be -1
    long_sum = weights[weights > 0].sum()
    short_sum = weights[weights < 0].sum()
    net_exposure = weights.sum()
    
    if abs(net_exposure) > 0.01:  # Allow small rounding errors
        logger.warning(f"Portfolio not dollar-neutral: net_exposure={net_exposure:.4f}, long_sum={long_sum:.4f}, short_sum={short_sum:.4f}")
    
    return weights


def compute_score_proportional_weights(
    scores: pd.Series,
    long_tickers: List[str],
    short_tickers: List[str],
    max_weight: Optional[float] = None
) -> pd.Series:
    """
    Compute score-proportional weights for long and short legs.
    
    Weights within each leg are proportional to scores, normalized to sum to ±1.
    
    Parameters
    ----------
    scores : pd.Series
        Composite factor scores indexed by ticker
    long_tickers : List[str]
        Tickers for long leg
    short_tickers : List[str]
        Tickers for short leg
    max_weight : float, optional
        Maximum weight per stock (default from config)
    
    Returns
    -------
    pd.Series
        Weights indexed by ticker (long positive, short negative)
    """
    if max_weight is None:
        max_weight = BACKTEST_CONFIG.max_weight
    
    weights = pd.Series(0.0, index=long_tickers + short_tickers)
    
    # Long leg: positive scores, normalize to sum to +1
    if long_tickers:
        long_scores = scores.loc[long_tickers]
        # Make scores positive and normalize
        long_weights = long_scores - long_scores.min() + 0.01  # Add small constant
        long_weights = long_weights / long_weights.sum()
        
        # Clip at max_weight
        long_weights = long_weights.clip(upper=max_weight)
        # Renormalize after clipping
        long_weights = long_weights / long_weights.sum()
        
        weights.loc[long_tickers] = long_weights
    
    # Short leg: negative scores (inverted), normalize to sum to -1
    if short_tickers:
        short_scores = scores.loc[short_tickers]
        # Make scores positive (invert) and normalize
        short_weights = -(short_scores - short_scores.max() - 0.01)  # Invert and add constant
        short_weights = short_weights / abs(short_weights.sum())
        
        # Clip at max_weight
        short_weights = short_weights.clip(lower=-max_weight)
        # Renormalize after clipping
        short_weights = short_weights / abs(short_weights.sum())
        
        weights.loc[short_tickers] = short_weights
    
    return weights


def construct_portfolio(
    scores: pd.Series,
    eligible_tickers: Optional[List[str]] = None,
    previous_weights: Optional[pd.Series] = None,
    sector_info: Optional[pd.Series] = None,
    market_cap: Optional[pd.Series] = None
) -> pd.Series:
    """
    Construct target portfolio weights from composite scores.
    
    Parameters
    ----------
    scores : pd.Series
        Composite factor scores indexed by ticker
    eligible_tickers : List[str], optional
        List of eligible tickers (if None, uses all in scores)
    previous_weights : pd.Series, optional
        Previous period weights (for turnover calculation)
    sector_info : pd.Series, optional
        Sector classification indexed by ticker
    market_cap : pd.Series, optional
        Market cap for each ticker (for beta-neutral if needed)
    
    Returns
    -------
    pd.Series
        Target portfolio weights indexed by ticker
    """
    # Filter to eligible tickers
    if eligible_tickers is not None:
        scores = scores.loc[scores.index.intersection(eligible_tickers)]
    
    if scores.empty:
        logger.warning("No scores available for portfolio construction")
        return pd.Series(dtype=float)
    
    # Select long/short stocks
    long_tickers, short_tickers = select_long_short_stocks(
        scores,
        BACKTEST_CONFIG.long_pct,
        BACKTEST_CONFIG.short_pct
    )
    
    if not long_tickers and not short_tickers:
        logger.warning("No stocks selected for portfolio")
        return pd.Series(dtype=float)
    
    # Verify ticker lists contain strings
    if long_tickers and not isinstance(long_tickers[0], str):
        logger.error(f"long_tickers contains non-strings: {long_tickers[:3]}")
        long_tickers = [str(t) for t in long_tickers]
    if short_tickers and not isinstance(short_tickers[0], str):
        logger.error(f"short_tickers contains non-strings: {short_tickers[:3]}")
        short_tickers = [str(t) for t in short_tickers]
    
    # Compute initial weights
    if BACKTEST_CONFIG.weighting_scheme == "score_proportional":
        weights = compute_score_proportional_weights(scores, long_tickers, short_tickers)
    else:  # equal
        weights = compute_equal_weights(long_tickers, short_tickers)
    
    # Verify weights index contains strings
    if len(weights) > 0 and not isinstance(weights.index[0], str):
        logger.error(f"weights index is not strings, sample: {weights.index[:3].tolist()}, converting")
        # Convert index to string
        weights.index = weights.index.astype(str)
        weights.index.name = 'ticker'
    
    # Apply constraints
    weights = apply_constraints(
        weights,
        sector_info=sector_info,
        market_cap=market_cap
    )
    
    # Final check: ensure index is strings
    if len(weights) > 0 and not isinstance(weights.index[0], str):
        logger.error(f"weights index still not strings after constraints, sample: {weights.index[:3].tolist()}")
        weights.index = weights.index.astype(str)
        weights.index.name = 'ticker'
    
    return weights


def construct_portfolio_panel(
    composite_scores: pd.DataFrame,
    eligible_tickers: Optional[pd.DataFrame] = None,
    sector_info: Optional[pd.DataFrame] = None,
    market_cap: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Construct target portfolio weights for all rebalancing dates.
    
    Parameters
    ----------
    composite_scores : pd.DataFrame
        Composite scores with (date, ticker) MultiIndex
    eligible_tickers : pd.DataFrame, optional
        Eligible tickers by date (boolean DataFrame)
    sector_info : pd.DataFrame, optional
        Sector classification with (date, ticker) MultiIndex
    market_cap : pd.Series, optional
        Market cap series
    
    Returns
    -------
    pd.DataFrame
        Target portfolio weights with (date, ticker) MultiIndex
    """
    logger.info("Constructing portfolios for all rebalancing dates")
    
    all_weights = []
    dates = composite_scores.index.get_level_values('date').unique()
    
    previous_weights = None
    
    for date in dates:
        # Extract scores for this date
        try:
            date_slice = composite_scores.loc[(date, slice(None)), :]
            
            # Get the composite_score column (or first column)
            if isinstance(date_slice, pd.DataFrame):
                if 'composite_score' in date_slice.columns:
                    date_scores = date_slice['composite_score']
                else:
                    # Get first column
                    date_scores = date_slice.iloc[:, 0]
            else:
                date_scores = date_slice
        except (KeyError, IndexError) as e:
            logger.debug(f"Error extracting scores for {date}: {e}")
            continue
        
        # Ensure date_scores is indexed by ticker only (not MultiIndex)
        # The index should be (date, ticker) MultiIndex, we need to extract ticker level
        if isinstance(date_scores.index, pd.MultiIndex):
            # Use xs to extract ticker level directly
            # date_scores already has MultiIndex (date, ticker), we just need ticker
            if 'ticker' in date_scores.index.names:
                # Get ticker level values directly
                ticker_values = date_scores.index.get_level_values('ticker')
                date_scores.index = ticker_values
                date_scores.index.name = 'ticker'
            else:
                # Fallback: assume level 1 is ticker
                ticker_values = date_scores.index.get_level_values(1)
                date_scores.index = ticker_values
                date_scores.index.name = 'ticker'
        elif date_scores.index.name != 'ticker':
            # If not MultiIndex but index name is not 'ticker', try to rename
            date_scores.index.name = 'ticker'
        
        # Verify index contains strings (ticker names) - critical check
        if len(date_scores) > 0:
            sample_idx = date_scores.index[0]
            if not isinstance(sample_idx, str):
                # This is the problem! Index is not ticker names
                logger.error(f"CRITICAL: date_scores index is not strings on {date}, sample: {date_scores.index[:5].tolist()}, type: {type(sample_idx)}")
                # Try to get ticker names from the original composite_scores MultiIndex
                try:
                    # Get the full MultiIndex slice for this date
                    date_full_slice = composite_scores.loc[(date, slice(None)), :]
                    # Extract ticker level from the MultiIndex
                    if isinstance(date_full_slice.index, pd.MultiIndex) and 'ticker' in date_full_slice.index.names:
                        ticker_level = date_full_slice.index.get_level_values('ticker')
                        # Align with date_scores (they should have same length and order)
                        if len(ticker_level) == len(date_scores):
                            date_scores.index = ticker_level
                            date_scores.index.name = 'ticker'
                            logger.info(f"Fixed ticker index on {date}, now has {len(date_scores)} tickers")
                        else:
                            logger.error(f"Ticker level length mismatch on {date}: {len(ticker_level)} vs {len(date_scores)}")
                            continue
                    else:
                        logger.error(f"Cannot extract ticker from composite_scores on {date}")
                        continue
                except Exception as e:
                    logger.error(f"Error fixing ticker index on {date}: {e}")
                    continue
        
        # Debug: check if scores are all NaN or empty
        if date_scores.empty:
            logger.warning(f"Empty scores for {date}")
            continue
        if date_scores.isna().all():
            logger.warning(f"All scores are NaN for {date}, skipping")
            continue
        
        # Remove NaN scores before passing to construct_portfolio
        date_scores_clean = date_scores.dropna()
        if date_scores_clean.empty:
            logger.warning(f"No valid scores after dropping NaN for {date}")
            continue
        
        # CRITICAL: Verify index contains ticker strings before constructing portfolio
        if len(date_scores_clean) > 0:
            sample_idx = date_scores_clean.index[0]
            if not isinstance(sample_idx, str):
                logger.error(f"CRITICAL on {date}: date_scores_clean index is not strings, sample: {date_scores_clean.index[:5].tolist()}, type: {type(sample_idx)}")
                # Try to fix by getting ticker names from composite_scores
                try:
                    date_full = composite_scores.loc[(date, slice(None)), :]
                    if isinstance(date_full.index, pd.MultiIndex) and 'ticker' in date_full.index.names:
                        ticker_names = date_full.index.get_level_values('ticker')
                        # Align with date_scores_clean
                        if len(ticker_names) >= len(date_scores_clean):
                            # Use the ticker names corresponding to non-NaN scores
                            non_na_mask = date_scores.notna()
                            ticker_names_clean = ticker_names[non_na_mask]
                            if len(ticker_names_clean) == len(date_scores_clean):
                                date_scores_clean.index = ticker_names_clean
                                date_scores_clean.index.name = 'ticker'
                                logger.info(f"Fixed ticker index on {date}, now has {len(date_scores_clean)} tickers with string names")
                            else:
                                logger.error(f"Ticker name alignment failed on {date}")
                                continue
                        else:
                            logger.error(f"Ticker names length mismatch on {date}")
                            continue
                    else:
                        logger.error(f"Cannot extract ticker names from composite_scores on {date}")
                        continue
                except Exception as e:
                    logger.error(f"Error fixing ticker index on {date}: {e}")
                    continue
        
        # Log some info about scores for first few dates
        if len([d for d in dates if d <= date]) <= 5:
            logger.info(f"Date {date}: {len(date_scores_clean)} valid scores, range [{date_scores_clean.min():.4f}, {date_scores_clean.max():.4f}], sample tickers: {date_scores_clean.index[:3].tolist()}")
        
        # Get eligible tickers for this date
        eligible = None
        if eligible_tickers is not None:
            try:
                if isinstance(eligible_tickers, pd.DataFrame):
                    eligible = eligible_tickers.loc[(date, slice(None))].index.get_level_values('ticker').tolist()
                else:
                    eligible = eligible_tickers
            except (KeyError, AttributeError):
                eligible = None
        
        # Get sector info for this date
        sector = None
        if sector_info is not None:
            sector = sector_info.loc[(date, slice(None)), 'sector']
        
        # Get market cap for this date
        mc = None
        if market_cap is not None:
            mc_date = market_cap.loc[(market_cap.index.get_level_values('date') <= date)]
            mc = mc_date.groupby('ticker').last()
        
        # Construct portfolio (use cleaned scores)
        weights = construct_portfolio(
            date_scores_clean,
            eligible_tickers=eligible,
            previous_weights=previous_weights,
            sector_info=sector,
            market_cap=mc
        )
        
        if weights.empty:
            logger.debug(f"No weights for {date}, skipping")
            continue
            
        # Verify weights index contains ticker strings before converting to DataFrame
        if len(weights) > 0:
            sample_idx = weights.index[0]
            if not isinstance(sample_idx, str):
                logger.error(f"CRITICAL on {date}: weights index is not strings before to_frame, sample: {weights.index[:3].tolist()}, type: {type(sample_idx)}")
                # This should not happen if construct_portfolio is working correctly
                # But if it does, try to fix it
                weights.index = weights.index.astype(str)
                weights.index.name = 'ticker'
        
        weights = weights.to_frame()
        # Rename the weight column to 'weight' for clarity
        if weights.columns[0] != 'weight':
            weights = weights.rename(columns={weights.columns[0]: 'weight'})
        
        # CRITICAL: Before reset_index, verify the index contains ticker names
        if len(weights) > 0:
            sample_idx = weights.index[0]
            if not isinstance(sample_idx, str):
                logger.error(f"CRITICAL on {date}: weights DataFrame index is not strings, sample: {weights.index[:3].tolist()}")
                # Try to get ticker names from date_scores_clean if available
                if 'date_scores_clean' in locals() and len(date_scores_clean) > 0:
                    # Map numeric indices to ticker names
                    ticker_map = {i: ticker for i, ticker in enumerate(date_scores_clean.index)}
                    weights.index = weights.index.map(ticker_map).fillna(weights.index)
                    weights.index.name = 'ticker'
                else:
                    logger.error(f"Cannot fix weights index on {date}")
                    continue
        
        weights['date'] = date
        
        # Reset index - the index should be ticker names (strings)
        if weights.index.name != 'ticker':
            weights.index.name = 'ticker'
        weights = weights.reset_index()
        
        # Ensure 'ticker' column exists and contains strings
        if 'ticker' not in weights.columns:
            # If reset_index created 'index' column, rename it
            if 'index' in weights.columns:
                weights = weights.rename(columns={'index': 'ticker'})
            else:
                # Create ticker column from index values
                weights['ticker'] = weights.index.astype(str)
                weights = weights.reset_index(drop=True)
        
        # Final verification: ticker column should contain strings
        if len(weights) > 0 and not isinstance(weights['ticker'].iloc[0], str):
            logger.error(f"CRITICAL on {date}: ticker column is not strings after reset_index, sample: {weights['ticker'].head(3).tolist()}")
            weights['ticker'] = weights['ticker'].astype(str)
        
        weights = weights.set_index(['date', 'ticker'])
        all_weights.append(weights)
        
        # Update previous weights for next iteration
        previous_weights = weights.iloc[:, 0] if len(weights.columns) > 0 else None
    
    if not all_weights:
        logger.warning("No portfolio weights generated for any date. This may be due to missing factor scores.")
        # Return empty DataFrame with correct structure
        empty_idx = pd.MultiIndex.from_product([dates, []], names=['date', 'ticker'])
        return pd.DataFrame(index=empty_idx, columns=['weight'])
    
    result = pd.concat(all_weights, axis=0).sort_index()
    logger.info(f"Constructed portfolios: {len(result)} weight records")
    
    return result


