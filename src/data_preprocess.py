"""
Data preprocessing module for multi-factor equity strategy.

This module handles cleaning, alignment, and preprocessing of market and
fundamental data before factor construction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import timedelta

from .utils.config import DATA_CONFIG, BACKTEST_CONFIG
from .utils.logging import get_logger

logger = get_logger("data_preprocess")


def filter_universe(
    price_data: pd.DataFrame,
    min_price: Optional[float] = None,
    min_history_months: Optional[int] = None,
    min_volume: Optional[float] = None
) -> List[str]:
    """
    Filter universe based on data quality and liquidity criteria.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        MultiIndex DataFrame with (date, ticker) index
    min_price : float, optional
        Minimum stock price (default from config)
    min_history_months : int, optional
        Minimum history required in months (default from config)
    min_volume : float, optional
        Minimum average daily volume (default from config)
    
    Returns
    -------
    List[str]
        List of eligible tickers
    """
    if min_price is None:
        min_price = BACKTEST_CONFIG.min_price
    if min_history_months is None:
        min_history_months = BACKTEST_CONFIG.min_history_months
    if min_volume is None:
        min_volume = DATA_CONFIG.min_volume
    
    logger.info(f"Filtering universe: min_price={min_price}, min_history={min_history_months} months")
    
    eligible_tickers = set(price_data.index.get_level_values('ticker').unique())
    
    # Filter by minimum price
    if min_price > 0:
        recent_prices = price_data['adj_close'].groupby('ticker').last()
        price_filter = recent_prices >= min_price
        eligible_tickers = eligible_tickers.intersection(set(price_filter[price_filter].index))
        logger.info(f"After price filter: {len(eligible_tickers)} tickers")
    
    # Filter by minimum history
    if min_history_months > 0:
        ticker_counts = price_data.groupby('ticker').size()
        start_date = price_data.index.get_level_values('date').min()
        min_required_days = min_history_months * 21  # Approximate trading days per month
        
        history_filter = ticker_counts >= min_required_days
        eligible_tickers = eligible_tickers.intersection(set(history_filter[history_filter].index))
        logger.info(f"After history filter: {len(eligible_tickers)} tickers")
    
    # Filter by minimum volume (average over recent period)
    if min_volume > 0:
        recent_data = price_data.loc[price_data.index.get_level_values('date') >= 
                                    price_data.index.get_level_values('date').max() - timedelta(days=90)]
        avg_volumes = recent_data.groupby('ticker')['volume'].mean()
        volume_filter = avg_volumes >= min_volume
        eligible_tickers = eligible_tickers.intersection(set(volume_filter[volume_filter].index))
        logger.info(f"After volume filter: {len(eligible_tickers)} tickers")
    
    eligible_list = sorted(list(eligible_tickers))
    logger.info(f"Final eligible universe: {len(eligible_list)} tickers")
    
    return eligible_list


def compute_market_cap(price_data: pd.DataFrame, fundamental_data: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Compute market capitalization for each (date, ticker).
    
    Parameters
    ----------
    price_data : pd.DataFrame
        MultiIndex DataFrame with (date, ticker) index, must have 'adj_close'
    fundamental_data : pd.DataFrame, optional
        MultiIndex DataFrame with (date, ticker) index, if provided and has 'shares_outstanding',
        uses that; otherwise estimates from price
    
    Returns
    -------
    pd.Series
        Market cap series with (date, ticker) MultiIndex
    """
    prices = price_data['adj_close']
    
    if fundamental_data is not None and 'shares_outstanding' in fundamental_data.columns:
        # Use actual shares outstanding if available
        shares = fundamental_data['shares_outstanding']
        # Forward fill shares outstanding
        shares = shares.groupby('ticker').ffill()
        # Align with price dates
        shares_aligned = shares.reindex(prices.index, method='ffill')
        market_cap = prices * shares_aligned
    else:
        # Estimate market cap using price (simplified - assumes $100M minimum)
        # This is a rough proxy; in production, use actual shares outstanding
        logger.warning("Estimating market cap from price only (no shares data).")
        market_cap = prices * 1e8  # Rough estimate
    
    return market_cap.rename('market_cap')


def clean_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data: handle missing values, outliers, etc.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        MultiIndex DataFrame with (date, ticker) index
    
    Returns
    -------
    pd.DataFrame
        Cleaned price data
    """
    logger.info("Cleaning price data")
    
    df = price_data.copy()
    
    # Remove rows with missing adj_close
    df = df.dropna(subset=['adj_close'])
    
    # Remove extreme outliers (e.g., prices > 100x median for that ticker)
    for ticker in df.index.get_level_values('ticker').unique():
        ticker_prices = df.loc[(slice(None), ticker), 'adj_close']
        median_price = ticker_prices.median()
        if median_price > 0:
            outlier_mask = (ticker_prices > median_price * 100) | (ticker_prices < median_price * 0.01)
            if outlier_mask.any():
                df = df.drop(df.loc[(slice(None), ticker)].index[outlier_mask])
    
    # Ensure volume is non-negative
    if 'volume' in df.columns:
        df['volume'] = df['volume'].clip(lower=0)
    
    logger.info(f"Cleaned price data: {len(df)} records")
    return df


def align_fundamentals(
    fundamental_data: pd.DataFrame,
    price_data_index: pd.DatetimeIndex,
    lag_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Align fundamental data with trading calendar and apply reporting lag.
    
    Parameters
    ----------
    fundamental_data : pd.DataFrame
        MultiIndex DataFrame with (date, ticker) index
    price_data_index : pd.DatetimeIndex
        Trading dates from price data
    lag_days : int, optional
        Days to lag fundamentals after reporting date (default from config)
    
    Returns
    -------
    pd.DataFrame
        Aligned fundamental data indexed by trading dates
    """
    if lag_days is None:
        lag_days = DATA_CONFIG.fundamental_lag_days
    
    logger.info(f"Aligning fundamentals with {lag_days} day lag")
    
    # Check if fundamental_data is empty or has no tickers
    if fundamental_data.empty:
        logger.warning("Fundamental data is empty")
        empty_idx = pd.MultiIndex.from_product([price_data_index, []], names=['date', 'ticker'])
        return pd.DataFrame(index=empty_idx)
    
    # Forward fill fundamentals to all trading dates
    aligned_data = []
    
    for ticker in fundamental_data.index.get_level_values('ticker').unique():
        # Use xs() method for more reliable MultiIndex slicing
        try:
            ticker_fundamentals = fundamental_data.xs(ticker, level='ticker')
        except KeyError:
            logger.warning(f"Ticker {ticker} not found in fundamental data")
            continue
        
        if ticker_fundamentals.empty:
            continue
        
        # Create series for each fundamental metric
        for col in ticker_fundamentals.columns:
            fund_series = ticker_fundamentals[col].dropna()
            
            if fund_series.empty:
                continue
            
            # Shift reporting dates by lag_days
            fund_series.index = fund_series.index + timedelta(days=lag_days)
            
            # Reindex to trading dates and forward fill
            aligned_series = fund_series.reindex(price_data_index, method='ffill')
            aligned_series.name = col
            aligned_series = aligned_series.to_frame()
            aligned_series['ticker'] = ticker
            aligned_series['date'] = aligned_series.index
            aligned_series = aligned_series.set_index(['date', 'ticker'])
            
            aligned_data.append(aligned_series)
    
    if not aligned_data:
        logger.warning("No aligned fundamental data generated")
        # Return empty DataFrame with correct structure
        empty_idx = pd.MultiIndex.from_product([price_data_index, []], names=['date', 'ticker'])
        return pd.DataFrame(index=empty_idx, columns=fundamental_data.columns)
    
    # Combine all aligned data
    result = pd.concat(aligned_data, axis=0)
    result = result.groupby(['date', 'ticker']).first()  # In case of duplicates
    result = result.sort_index()
    
    logger.info(f"Aligned fundamentals: {len(result)} records")
    return result


def preprocess_data(
    price_data: pd.DataFrame,
    fundamental_data: Optional[pd.DataFrame] = None,
    eligible_tickers: Optional[List[str]] = None
) -> tuple:
    """
    Main preprocessing function: clean, filter, and align all data.
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Raw price data
    fundamental_data : pd.DataFrame, optional
        Raw fundamental data
    eligible_tickers : List[str], optional
        Pre-filtered list of eligible tickers
    
    Returns
    -------
    tuple
        (clean_price_data, clean_fundamental_data, market_cap_series, eligible_tickers)
    """
    logger.info("Starting data preprocessing")
    
    # Clean price data
    clean_prices = clean_price_data(price_data)
    
    # Filter universe if not pre-filtered
    if eligible_tickers is None:
        eligible_tickers = filter_universe(clean_prices)
    
    # Filter price data to eligible tickers
    clean_prices = clean_prices.loc[(slice(None), eligible_tickers), :]
    
    # Compute market cap
    market_cap = compute_market_cap(clean_prices, fundamental_data)
    clean_prices['market_cap'] = market_cap
    
    # Align fundamentals if provided
    clean_fundamentals = None
    if fundamental_data is not None and len(eligible_tickers) > 0:
        # Filter fundamental data to eligible tickers using isin for MultiIndex
        fundamental_mask = fundamental_data.index.get_level_values('ticker').isin(eligible_tickers)
        filtered_fundamentals = fundamental_data[fundamental_mask]
        
        trading_dates = clean_prices.index.get_level_values('date').unique()
        clean_fundamentals = align_fundamentals(
            filtered_fundamentals,
            trading_dates
        )
    
    logger.info("Data preprocessing complete")
    
    return clean_prices, clean_fundamentals, market_cap, eligible_tickers


