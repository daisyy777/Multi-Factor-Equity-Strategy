"""
Data loading module for multi-factor equity strategy.

This module handles downloading and loading market data (prices, volumes)
and fundamental data from various sources.

Key design principle (look-ahead bias prevention):
  Fundamental data is timestamped as:
    available_date = quarter_end_date + report_lag_days
  so the backtest never sees data before it was publicly available.
"""

import io
import urllib.request
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
    Load universe of stocks (static fallback list).

    For a bias-free backtest prefer load_historical_sp500_universe().
    """
    if universe_file is None:
        universe_file = DATA_CONFIG.universe_file

    universe_path = METADATA_DIR / universe_file

    if universe_path.exists():
        logger.info(f"Loading universe from {universe_path}")
        df = pd.read_csv(universe_path)
        df.columns = df.columns.str.lower()
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        return df

    logger.warning(f"Universe file not found at {universe_path}. Using static sample list.")
    sample_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "V", "UNH", "JNJ", "WMT", "XOM", "JPM", "PG", "MA", "CVX", "LLY",
        "HD", "ABBV", "MRK", "PEP", "COST", "AVGO", "ADBE", "CSCO", "MCD",
        "TMO", "ACN", "DIS", "DHR", "VZ", "NFLX", "NKE", "PM", "TXN",
        "NEE", "CMCSA", "LIN", "WFC", "RTX", "HON", "BMY", "UPS", "QCOM",
        "COP", "ORCL", "IBM", "LOW", "UNP", "MS", "GS", "AMGN", "BA",
        "GE", "SPGI", "CAT", "PLD", "INTU", "AMAT", "ELV", "AXP", "DE",
        "BKNG", "ADI", "SBUX", "C", "MDT", "ISRG", "TJX", "GILD", "ZTS",
        "REGN", "BLK", "FI", "ADP", "PANW", "SNPS", "CDNS", "CRWD", "FTNT",
        "KLAC", "ANET", "MCHP", "NXPI", "LRCX", "CTAS", "APH", "ODFL",
        "CME", "FAST", "TTD", "FDS", "POOL", "TSCO", "ARES", "DXCM",
    ]
    df = pd.DataFrame({"ticker": sample_tickers})
    df.to_csv(universe_path, index=False)
    return df


def load_historical_sp500_universe(
    date: Optional[pd.Timestamp] = None,
    fallback_to_static: bool = True,
) -> pd.DataFrame:
    """
    Load historical S&P 500 constituents to eliminate survivorship bias.

    Downloads the public dataset maintained at:
        https://github.com/fja05680/sp500

    Each row in that CSV corresponds to a snapshot of the index at a given
    date, with a comma-separated list of tickers in the ``tickers`` column.

    Parameters
    ----------
    date : pd.Timestamp, optional
        If provided, returns constituents as of that date.
        If None, returns the full history DataFrame.
    fallback_to_static : bool
        Fall back to the static list (with survivorship bias) when the
        download fails rather than raising.

    Returns
    -------
    pd.DataFrame
        Columns: ['ticker'] when *date* is specified;
        otherwise the raw history DataFrame with a 'tickers' column.
    """
    cache_file = METADATA_DIR / "sp500_historical.csv"

    def _parse_snapshot(hist_df: pd.DataFrame, snap_date: pd.Timestamp) -> pd.DataFrame:
        available = hist_df.index[hist_df.index <= snap_date]
        if len(available) == 0:
            logger.warning(f"No S&P 500 snapshot available on or before {snap_date}")
            return pd.DataFrame({"ticker": []})
        tickers_str = hist_df.loc[available.max(), "tickers"]
        tickers = [t.strip() for t in str(tickers_str).split(",") if t.strip()]
        return pd.DataFrame({"ticker": tickers})

    # --- Try cache ---
    if cache_file.exists():
        try:
            hist_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.debug(f"Loaded historical universe from cache ({len(hist_df)} snapshots)")
            if date is not None:
                return _parse_snapshot(hist_df, date)
            return hist_df
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    # --- Try download ---
    url = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes(04-11-2023).csv"
    )
    try:
        logger.info("Downloading historical S&P 500 components from GitHub...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        hist_df = pd.read_csv(io.StringIO(content), index_col=0, parse_dates=True)
        hist_df.to_csv(cache_file)
        logger.info(f"Cached historical S&P 500 data: {len(hist_df)} snapshots")
        if date is not None:
            return _parse_snapshot(hist_df, date)
        return hist_df
    except Exception as e:
        logger.warning(f"Could not download historical S&P 500 data: {e}")
        if fallback_to_static:
            logger.warning(
                "Falling back to static ticker list. "
                "Results will contain survivorship bias."
            )
            return load_universe()
        raise


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily adjusted price data for a list of tickers.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) with columns:
        ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    logger.info(
        f"Downloading price data for {len(tickers)} tickers "
        f"from {start_date} to {end_date}"
    )

    cache_file = RAW_DATA_DIR / f"prices_{start_date}_{end_date}.parquet"
    if cache and cache_file.exists():
        logger.info(f"Loading cached price data from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Re-downloading.")

    col_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    try:
        df_raw = yf.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
        )
        if df_raw.empty:
            raise ValueError("No data downloaded")

        if not isinstance(df_raw.columns, pd.MultiIndex):
            if len(tickers) == 1:
                df_raw.columns = pd.MultiIndex.from_product([df_raw.columns, tickers])
            else:
                raise ValueError("Unexpected column structure from yfinance")

        all_data = []
        failed_tickers = []
        available_tickers = df_raw.columns.get_level_values(1).unique()

        for ticker in tickers:
            if ticker not in available_tickers:
                failed_tickers.append(ticker)
                continue
            try:
                ticker_cols = df_raw.xs(ticker, level=1, axis=1)
                if ticker_cols.empty or "Adj Close" not in ticker_cols.columns:
                    failed_tickers.append(ticker)
                    continue
                td = ticker_cols.reset_index().rename(columns={"Date": "date"})
                td["date"] = pd.to_datetime(td["date"])
                td["ticker"] = ticker
                selected = ["date", "ticker"] + [
                    c for c in col_mapping if c in td.columns
                ]
                td = td[selected].rename(columns=col_mapping)
                if "adj_close" not in td.columns and "close" in td.columns:
                    td["adj_close"] = td["close"]
                all_data.append(td)
            except Exception as ex:
                logger.warning(f"Error processing {ticker}: {ex}")
                failed_tickers.append(ticker)

        if not all_data:
            raise ValueError("No data for any ticker")

        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index(["date", "ticker"]).sort_index()

    except Exception as e:
        logger.error(f"Batch download failed: {e}. Falling back to individual downloads.")
        all_data = []
        failed_tickers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date, end=end_date, auto_adjust=False
                )
                if df.empty:
                    failed_tickers.append(ticker)
                    continue
                df = df.reset_index()
                df["ticker"] = ticker
                df["date"] = pd.to_datetime(df["Date"])
                df = df[["date", "ticker", "Open", "High", "Low", "Close",
                         "Adj Close", "Volume"]]
                df.columns = ["date", "ticker", "open", "high", "low", "close",
                               "adj_close", "volume"]
                all_data.append(df)
            except Exception as ex:
                logger.warning(f"Error downloading {ticker}: {ex}")
                failed_tickers.append(ticker)

        if not all_data:
            raise ValueError("No data downloaded for any ticker")
        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index(["date", "ticker"]).sort_index()

    logger.info(
        f"Downloaded price data for "
        f"{len(tickers) - len(failed_tickers)}/{len(tickers)} tickers"
    )
    if failed_tickers:
        logger.warning(f"Failed tickers (price): {failed_tickers[:10]}")

    if cache:
        try:
            result.to_parquet(cache_file)
            logger.info(f"Cached price data to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache price data: {e}")

    return result


def download_fundamental_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache: bool = True,
    report_lag_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download *historical* fundamental data using yfinance quarterly statements.

    Design (look-ahead bias prevention)
    ------------------------------------
    Each data record is timestamped as::

        available_date = quarter_end_date + report_lag_days

    SEC 10-Q filings are due 40 days after quarter end for large accelerated
    filers and 45 days for accelerated filers.  Using 60 days provides a
    conservative safety margin and matches common practice in academic
    factor research.

    This replaces the previous implementation that used ``yf.Ticker().info``
    (a *current* snapshot) and broadcast that single value across all
    historical dates, which caused severe look-ahead bias.

    Parameters
    ----------
    tickers : list of str
    start_date : str  (YYYY-MM-DD)
    end_date   : str  (YYYY-MM-DD)
    cache : bool
    report_lag_days : int, optional
        Days after quarter end before data is tradeable.
        Defaults to ``DATA_CONFIG.fundamental_report_lag_days`` (60).

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker).  Columns include:
        ``book_value_per_share``, ``shares_outstanding``, ``book_value``,
        ``total_assets``, ``total_liabilities``, ``net_income``,
        ``operating_cashflow``, ``eps``.
    """
    if report_lag_days is None:
        report_lag_days = DATA_CONFIG.fundamental_report_lag_days

    logger.info(
        f"Downloading historical fundamental data for {len(tickers)} tickers "
        f"(report_lag={report_lag_days} days)"
    )

    # Cache key includes lag so changing the lag invalidates stale cache.
    cache_file = (
        RAW_DATA_DIR
        / f"fundamentals_hist_{start_date}_{end_date}_lag{report_lag_days}.parquet"
    )
    if cache and cache_file.exists():
        logger.info(f"Loading cached fundamental data from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Re-downloading.")

    all_data: list = []
    failed_tickers: list = []

    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            logger.info(f"Downloading fundamentals: {i}/{len(tickers)} tickers")
        try:
            stock = yf.Ticker(ticker)

            # quarterly_income_stmt / quarterly_balance_sheet:
            #   index  = metric names
            #   columns = quarter-end dates (most recent first)
            income = stock.quarterly_income_stmt
            balance = stock.quarterly_balance_sheet

            income_ok = income is not None and not income.empty
            balance_ok = balance is not None and not balance.empty

            if not income_ok and not balance_ok:
                logger.debug(f"No quarterly statements for {ticker}")
                failed_tickers.append(ticker)
                continue

            # Collect all quarter-end dates present in either statement.
            all_dates: set = set()
            if income_ok:
                all_dates.update(pd.to_datetime(income.columns).tolist())
            if balance_ok:
                all_dates.update(pd.to_datetime(balance.columns).tolist())

            for q_date in sorted(all_dates):
                q_date = pd.Timestamp(q_date)
                available_date = q_date + pd.Timedelta(days=report_lag_days)

                # Skip quarters outside the requested date window.
                if available_date < pd.Timestamp(start_date):
                    continue
                if q_date > pd.Timestamp(end_date):
                    continue

                row: dict = {"date": available_date, "ticker": ticker}

                # ── Income statement ─────────────────────────────────────
                if income_ok and q_date in income.columns:
                    col = income[q_date]

                    for name in [
                        "Net Income",
                        "Net Income Common Stockholders",
                        "Net Income Including Noncontrolling Interests",
                    ]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["net_income"] = float(col[name])
                            break

                    for name in ["Basic EPS", "Diluted EPS"]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["eps"] = float(col[name])
                            break

                    for name in ["Operating Income", "EBIT"]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["operating_cashflow"] = float(col[name])
                            break

                # ── Balance sheet ─────────────────────────────────────────
                if balance_ok and q_date in balance.columns:
                    col = balance[q_date]

                    # Total stockholders' equity (book value)
                    for name in [
                        "Stockholders Equity",
                        "Common Stock Equity",
                        "Total Equity Gross Minority Interest",
                    ]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["book_value"] = float(col[name])
                            break

                    for name in ["Total Assets"]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["total_assets"] = float(col[name])
                            break

                    for name in [
                        "Total Liabilities Net Minority Interest",
                        "Total Liabilities",
                    ]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["total_liabilities"] = float(col[name])
                            break

                    for name in [
                        "Share Issued",
                        "Ordinary Shares Number",
                        "Common Stock",
                    ]:
                        if name in col.index and pd.notna(col.get(name)):
                            row["shares_outstanding"] = float(col[name])
                            break

                # Derive per-share book value when possible.
                if (
                    "book_value" in row
                    and "shares_outstanding" in row
                    and row["shares_outstanding"] > 0
                ):
                    row["book_value_per_share"] = (
                        row["book_value"] / row["shares_outstanding"]
                    )

                all_data.append(row)

        except Exception as e:
            logger.debug(f"Error downloading fundamentals for {ticker}: {e}")
            failed_tickers.append(ticker)

    if failed_tickers:
        logger.warning(
            f"Could not get fundamentals for {len(failed_tickers)} tickers: "
            f"{failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}"
        )

    if not all_data:
        logger.error(
            "No historical fundamental data downloaded. "
            "Value and quality factors will be degraded."
        )
        result = pd.DataFrame(
            columns=[
                "book_value_per_share",
                "shares_outstanding",
                "book_value",
                "total_assets",
                "total_liabilities",
                "net_income",
                "operating_cashflow",
                "eps",
            ]
        )
        result.index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return result

    result = pd.DataFrame(all_data)
    result["date"] = pd.to_datetime(result["date"])
    result = result.drop_duplicates(subset=["date", "ticker"])
    result = result.sort_values(["ticker", "date"])
    result = result.set_index(["date", "ticker"]).sort_index()

    logger.info(
        f"Downloaded historical fundamental data: {len(result)} records "
        f"for {result.index.get_level_values('ticker').nunique()} tickers"
    )

    if cache:
        try:
            result.to_parquet(cache_file)
            logger.info(f"Cached fundamental data to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache fundamental data: {e}")

    return result


def load_price_data_from_csv(file_path: Path) -> pd.DataFrame:
    """Load price data from a CSV file."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()


def load_fundamental_data_from_csv(file_path: Path) -> pd.DataFrame:
    """Load fundamental data from a CSV file."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()
