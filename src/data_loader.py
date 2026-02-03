"""
Data loading module for multi-factor equity strategy.

This module handles downloading and loading market data (prices, volumes)
and fundamental data from various sources (Yahoo Finance, CSV files, etc.).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta

from .utils.config import RAW_DATA_DIR, METADATA_DIR, DATA_CONFIG
from .utils.logging import get_logger

logger = get_logger("data_loader")


def load_universe(universe_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load universe of stocks (e.g., S&P 500 constituents).
    
    Parameters
    ----------
    universe_file : str, optional
        Name of the universe file in data/metadata/. If None, uses config.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['ticker', 'name', 'sector', 'industry'] (if available)
    """
    if universe_file is None:
        universe_file = DATA_CONFIG.universe_file
    
    universe_path = METADATA_DIR / universe_file
    
    if universe_path.exists():
        logger.info(f"Loading universe from {universe_path}")
        df = pd.read_csv(universe_path)
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'symbol' in df.columns:
            df = df.rename(columns={'symbol': 'ticker'})
        return df
    else:
        logger.warning(f"Universe file not found at {universe_path}. Creating sample S&P 500 list.")
        # Create a sample list of S&P 500 tickers (top 100 by market cap)
        # In production, this should be loaded from a proper data source
        sample_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'V', 'UNH', 'JNJ', 'WMT', 'XOM', 'JPM', 'PG', 'MA', 'CVX', 'LLY',
            'HD', 'ABBV', 'MRK', 'PEP', 'COST', 'AVGO', 'ADBE', 'CSCO', 'MCD',
            'TMO', 'ACN', 'DIS', 'DHR', 'VZ', 'NFLX', 'NKE', 'PM', 'TXN',
            'NEE', 'CMCSA', 'LIN', 'WFC', 'RTX', 'HON', 'BMY', 'UPS', 'QCOM',
            'COP', 'ORCL', 'IBM', 'LOW', 'UNP', 'MS', 'GS', 'AMGN', 'BA',
            'GE', 'SPGI', 'CAT', 'PLD', 'INTU', 'AMAT', 'ELV', 'AXP', 'DE',
            'BKNG', 'ADI', 'SBUX', 'C', 'MDT', 'ISRG', 'TJX', 'GILD', 'ZTS',
            'REGN', 'BLK', 'FI', 'ADP', 'PANW', 'SNPS', 'CDNS', 'CRWD', 'FTNT',
            'KLAC', 'ANET', 'MCHP', 'NXPI', 'LRCX', 'CTAS', 'APH', 'ODFL',
            'CME', 'FAST', 'TTD', 'FDS', 'POOL', 'TSCO', 'MNDY', 'ARES', 'DXCM'
        ]
        df = pd.DataFrame({'ticker': sample_tickers})
        # Save for future use
        df.to_csv(universe_path, index=False)
        return df


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache: bool = True
) -> pd.DataFrame:
    """
    Download daily price data for a list of tickers.
    
    Parameters
    ----------
    tickers : List[str]
        List of stock tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    cache : bool
        Whether to cache the data to disk
    
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (date, ticker) as index and columns:
        ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    logger.info(f"Downloading price data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Check cache first
    cache_file = RAW_DATA_DIR / f"prices_{start_date}_{end_date}.parquet"
    if cache and cache_file.exists():
        logger.info(f"Loading cached price data from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            logger.warning(f"Error loading cache: {e}. Re-downloading data.")
    
    # Download data using yf.download for multiple tickers at once
    try:
        # yf.download returns MultiIndex columns: (Field, Ticker)
        # Set auto_adjust=False to get 'Adj Close' column (default is True in newer versions)
        df_raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if df_raw.empty:
            raise ValueError("No data downloaded for any ticker")
        
        # Handle single ticker case (columns are not MultiIndex when only one ticker)
        if not isinstance(df_raw.columns, pd.MultiIndex):
            # Single ticker: convert to MultiIndex format
            if len(tickers) == 1:
                df_raw.columns = pd.MultiIndex.from_product([df_raw.columns, tickers])
            else:
                # This shouldn't happen, but handle it gracefully
                logger.warning("Expected MultiIndex columns for multiple tickers, but got single-level columns")
                raise ValueError("Unexpected column structure from yfinance download")
        
        # Convert MultiIndex columns to long format
        # Stack ticker level to get (date, ticker) MultiIndex
        all_data = []
        failed_tickers = []
        
        # Column mapping from yfinance to our format
        col_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        for ticker in tickers:
            try:
                # Extract columns for this ticker using xs
                # Check if ticker exists in columns
                available_tickers = df_raw.columns.get_level_values(1).unique()
                if ticker not in available_tickers:
                    logger.warning(f"Ticker {ticker} not found in downloaded data")
                    failed_tickers.append(ticker)
                    continue
                
                # Extract data for this ticker using xs (default drop_level=True removes the ticker level)
                try:
                    ticker_cols = df_raw.xs(ticker, level=1, axis=1)
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error extracting data for {ticker}: {e}")
                    failed_tickers.append(ticker)
                    continue
                
                if ticker_cols.empty:
                    logger.warning(f"No data for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                # Check for required 'Adj Close' column
                if 'Adj Close' not in ticker_cols.columns:
                    logger.warning(f"'Adj Close' not found for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue
                
                # Reset index to get date as column
                ticker_data = ticker_cols.reset_index()
                
                # Rename date column (yfinance DatetimeIndex becomes 'Date' column after reset_index)
                if 'Date' in ticker_data.columns:
                    ticker_data = ticker_data.rename(columns={'Date': 'date'})
                else:
                    # Fallback: use index if it's still datetime
                    ticker_data['date'] = pd.to_datetime(ticker_cols.index)
                
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                ticker_data['ticker'] = ticker
                
                # Select and rename columns
                selected_cols = ['date', 'ticker']
                for old_col in col_mapping.keys():
                    if old_col in ticker_data.columns:
                        selected_cols.append(old_col)
                
                ticker_data = ticker_data[selected_cols]
                ticker_data = ticker_data.rename(columns=col_mapping)
                
                # Ensure adj_close exists (critical column)
                if 'adj_close' not in ticker_data.columns:
                    logger.warning(f"'adj_close' column missing for {ticker}, using 'close' as fallback")
                    if 'close' in ticker_data.columns:
                        ticker_data['adj_close'] = ticker_data['close']
                    else:
                        logger.warning(f"No price data available for {ticker}, skipping")
                        failed_tickers.append(ticker)
                        continue
                
                all_data.append(ticker_data)
                logger.debug(f"Downloaded data for {ticker}")
                
            except KeyError as e:
                logger.warning(f"Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data downloaded for any ticker")
        
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index(['date', 'ticker']).sort_index()
        
        logger.info(f"Downloaded data for {len(tickers) - len(failed_tickers)}/{len(tickers)} tickers")
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers[:10]}")
            
    except Exception as e:
        logger.error(f"Error in batch download: {e}. Falling back to individual downloads.")
        # Fallback to individual downloads if batch fails
        all_data = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                df = df.reset_index()
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['Date'])
                df = df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                df.columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                
                all_data.append(df)
                logger.debug(f"Downloaded data for {ticker}")
                
            except Exception as ex:
                logger.warning(f"Error downloading {ticker}: {ex}")
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data downloaded for any ticker")
        
        result = pd.concat(all_data, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index(['date', 'ticker']).sort_index()
        
        logger.info(f"Downloaded data for {len(tickers) - len(failed_tickers)}/{len(tickers)} tickers")
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers[:10]}")
    
    # Cache if requested
    if cache:
        try:
            result.to_parquet(cache_file)
            logger.info(f"Cached price data to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache data: {e}")
    
    return result


def download_fundamental_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache: bool = True
) -> pd.DataFrame:
    """
    Download fundamental data for a list of tickers.
    
    Note: This is a simplified implementation. In production, you would use
    a proper fundamental data provider (e.g., Compustat, FactSet, or Alpha Vantage).
    
    Parameters
    ----------
    tickers : List[str]
        List of stock tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    cache : bool
        Whether to cache the data to disk
    
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (date, ticker) as index and fundamental columns
    """
    logger.info(f"Downloading fundamental data for {len(tickers)} tickers")
    logger.warning("Using simplified fundamental data. In production, use a proper data provider.")
    
    # Check cache
    cache_file = RAW_DATA_DIR / f"fundamentals_{start_date}_{end_date}.parquet"
    if cache and cache_file.exists():
        logger.info(f"Loading cached fundamental data from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            logger.warning(f"Error loading cache: {e}. Re-downloading data.")
    
    # For demonstration, we'll create synthetic fundamental data
    # In production, replace this with actual API calls to a fundamental data provider
    
    all_data = []
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')  # Quarterly
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract available fundamental metrics
            book_value = info.get('bookValue', np.nan)
            shares_outstanding = info.get('sharesOutstanding', np.nan)
            total_assets = info.get('totalAssets', np.nan)
            total_debt = info.get('totalDebt', np.nan)
            net_income = info.get('netIncomeToCommon', np.nan)
            operating_cashflow = info.get('operatingCashflow', np.nan)
            
            # Create quarterly records (simplified - actual data would have varying report dates)
            for date in dates:
                row = {
                    'date': date,
                    'ticker': ticker,
                    'book_value_per_share': book_value if not np.isnan(book_value) else np.nan,
                    'shares_outstanding': shares_outstanding if not np.isnan(shares_outstanding) else np.nan,
                    'total_assets': total_assets if not np.isnan(total_assets) else np.nan,
                    'total_liabilities': total_debt if not np.isnan(total_debt) else np.nan,
                    'net_income': net_income if not np.isnan(net_income) else np.nan,
                    'operating_cashflow': operating_cashflow if not np.isnan(operating_cashflow) else np.nan,
                    'eps': info.get('trailingEps', np.nan) if not np.isnan(info.get('trailingEps', np.nan)) else np.nan,
                }
                all_data.append(row)
                
        except Exception as e:
            logger.debug(f"Could not get fundamental data for {ticker}: {e}")
            continue
    
    if not all_data:
        logger.warning("No fundamental data downloaded. Creating empty DataFrame with expected structure.")
        dates_list = []
        tickers_list = []
        for date in dates:
            for ticker in tickers:
                dates_list.append(date)
                tickers_list.append(ticker)
        
        result = pd.DataFrame({
            'date': dates_list,
            'ticker': tickers_list,
            'book_value_per_share': np.nan,
            'shares_outstanding': np.nan,
            'total_assets': np.nan,
            'total_liabilities': np.nan,
            'net_income': np.nan,
            'operating_cashflow': np.nan,
            'eps': np.nan,
        })
    else:
        result = pd.DataFrame(all_data)
    
    result['date'] = pd.to_datetime(result['date'])
    result = result.set_index(['date', 'ticker']).sort_index()
    
    # Cache if requested
    if cache:
        try:
            result.to_parquet(cache_file)
            logger.info(f"Cached fundamental data to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache data: {e}")
    
    return result


def load_price_data_from_csv(file_path: Path) -> pd.DataFrame:
    """Load price data from a CSV file."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'ticker']).sort_index()
    return df


def load_fundamental_data_from_csv(file_path: Path) -> pd.DataFrame:
    """Load fundamental data from a CSV file."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'ticker']).sort_index()
    return df

