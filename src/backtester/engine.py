"""
Backtest engine module.

Event-driven daily backtest engine for multi-factor equity strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

from ..utils.config import BACKTEST_CONFIG
from ..utils.logging import get_logger

logger = get_logger("backtester.engine")


def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute returns from prices."""
    return prices.pct_change()


def get_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    frequency: str = "monthly",
    day: int = -1
) -> pd.DatetimeIndex:
    """
    Get rebalancing dates based on frequency.
    
    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
        All trading dates
    frequency : str
        "monthly" or "quarterly"
    day : int
        Day of month (-1 for last day, or specific day 1-31)
    
    Returns
    -------
    pd.DatetimeIndex
        Rebalancing dates
    """
    if frequency == "monthly":
        period = "M"
    elif frequency == "quarterly":
        period = "Q"
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Group by period
    date_df = pd.DataFrame({'date': trading_dates})
    date_df['period'] = date_df['date'].dt.to_period(period)
    
    if day == -1:
        # Last trading day of each period
        rebalance_dates = date_df.groupby('period')['date'].last().values
    else:
        # Specific day of month
        date_df['day_of_month'] = date_df['date'].dt.day
        date_df_filtered = date_df[date_df['day_of_month'] <= day]
        rebalance_dates = date_df_filtered.groupby('period')['date'].last().values
    
    return pd.DatetimeIndex(rebalance_dates).intersection(trading_dates)


class BacktestEngine:
    """
    Event-driven backtest engine.
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission_rate: float = None,
        slippage_rate: float = None
    ):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        initial_capital : float, optional
            Initial capital (default from config)
        commission_rate : float, optional
            Commission rate per side (default from config)
        slippage_rate : float, optional
            Slippage rate (default from config)
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG.initial_capital
        self.commission_rate = commission_rate or BACKTEST_CONFIG.commission_rate
        self.slippage_rate = slippage_rate or BACKTEST_CONFIG.slippage_rate
        
        # State
        self.positions = pd.Series(dtype=float)  # ticker -> shares
        self.cash = self.initial_capital
        self.nav_series = []
        self.return_series = []
        self.long_value_series = []
        self.short_value_series = []
        self.turnover_series = []
        self.trade_log = []
        
    def _compute_portfolio_value(self, prices: pd.Series, date: pd.Timestamp) -> Dict[str, float]:
        """
        Compute current portfolio value.
        
        Returns
        -------
        Dict with 'total', 'long', 'short', 'cash'
        """
        # Get current prices for positions
        if self.positions.empty:
            return {
                'total': self.cash,
                'long': 0.0,
                'short': 0.0,
                'cash': self.cash,
                'market_value': 0.0
            }
        
        position_tickers = self.positions.index.tolist()
        try:
            current_prices = prices.loc[(date, position_tickers)]
            # If current_prices has MultiIndex, drop date level
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel('date')
        except (KeyError, IndexError):
            # Some tickers might not have prices on this date
            available_tickers = [t for t in position_tickers if (date, t) in prices.index]
            if not available_tickers:
                return {
                    'total': self.cash,
                    'long': 0.0,
                    'short': 0.0,
                    'cash': self.cash,
                    'market_value': 0.0
                }
            current_prices = prices.loc[(date, available_tickers)]
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel('date')
            # Align positions with available prices
            self.positions = self.positions.loc[available_tickers]
        
        # Align indices for multiplication
        aligned_positions, aligned_prices = self.positions.align(current_prices, join='inner', fill_value=0.0)
        
        # Compute market value
        market_value = (aligned_positions * aligned_prices).sum()
        
        # Separate long and short
        long_mask = aligned_positions > 0
        short_mask = aligned_positions < 0
        long_value = (aligned_positions[long_mask] * aligned_prices[long_mask]).sum() if long_mask.any() else 0.0
        short_value = abs((aligned_positions[short_mask] * aligned_prices[short_mask]).sum()) if short_mask.any() else 0.0
        
        total_value = self.cash + market_value
        
        return {
            'total': total_value,
            'long': long_value,
            'short': short_value,
            'cash': self.cash,
            'market_value': market_value
        }
    
    def _execute_trades(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Execute trades to move from current positions to target weights.
        
        Returns
        -------
        Dict with 'costs', 'turnover', 'notional_traded'
        """
        # Get current portfolio value
        portfolio_values = self._compute_portfolio_value(prices, date)
        total_nav = portfolio_values['total']
        
        if total_nav <= 0:
            logger.warning(f"Portfolio value is non-positive on {date}")
            return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
        
        # Get prices for all target_weights tickers
        target_tickers = target_weights.index.tolist()
        is_first_rebalance = len(self.trade_log) == 0
        if is_first_rebalance:
            logger.info(f"Getting prices for {len(target_tickers)} target tickers on {date}")
        
        try:
            # Get prices for target tickers
            # Try exact date first
            current_prices = prices.loc[(date, target_tickers)]
            # If current_prices has MultiIndex, drop date level
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel('date')
            if is_first_rebalance:
                logger.info(f"Got prices for {len(current_prices)} tickers, price range: [{current_prices.min():.2f}, {current_prices.max():.2f}]")
        except (KeyError, IndexError) as e:
            # If exact date not found, try to find the closest trading date before or on this date
            if is_first_rebalance:
                logger.info(f"Exact date {date} not found in prices, looking for closest trading date")
            # Get all dates up to and including this date
            available_dates = prices.index.get_level_values('date').unique()
            available_dates = available_dates[available_dates <= date]
            if len(available_dates) == 0:
                if is_first_rebalance:
                    logger.warning(f"No trading dates available before or on {date}")
                logger.warning(f"No prices available for any target ticker on {date}")
                return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
            else:
                # Use the most recent date
                closest_date = available_dates.max()
                if is_first_rebalance:
                    logger.info(f"Using closest trading date: {closest_date}")
                try:
                    current_prices = prices.loc[(closest_date, target_tickers)]
                    if isinstance(current_prices.index, pd.MultiIndex):
                        current_prices = current_prices.droplevel('date')
                    if is_first_rebalance:
                        logger.info(f"Got prices for {len(current_prices)} tickers from {closest_date}, price range: [{current_prices.min():.2f}, {current_prices.max():.2f}]")
                except (KeyError, IndexError) as e2:
                    # Still failed, check available tickers
                    if is_first_rebalance:
                        logger.warning(f"Price lookup failed on {closest_date}: {e2}, checking available tickers")
                    available_tickers = [t for t in target_tickers if (closest_date, t) in prices.index]
                    if is_first_rebalance:
                        logger.info(f"Found {len(available_tickers)} available tickers out of {len(target_tickers)} on {closest_date}")
                        if len(available_tickers) == 0:
                            # Debug: check what tickers are in prices for this date
                            prices_on_date = prices.loc[closest_date] if closest_date in prices.index.get_level_values('date') else pd.Series()
                            if len(prices_on_date) > 0:
                                logger.info(f"  Prices available for {len(prices_on_date)} tickers on {closest_date}, sample: {prices_on_date.index.tolist()[:5]}")
                                logger.info(f"  Target tickers sample: {target_tickers[:5]}")
                            else:
                                logger.warning(f"  No prices at all on {closest_date}")
                    if not available_tickers:
                        logger.warning(f"No prices available for any target ticker on {date} (checked {closest_date})")
                        return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
                    current_prices = prices.loc[(closest_date, available_tickers)]
                    if isinstance(current_prices.index, pd.MultiIndex):
                        current_prices = current_prices.droplevel('date')
                    # Filter target_weights to available tickers
                    target_weights = target_weights.loc[target_weights.index.intersection(available_tickers)]
                    if target_weights.empty:
                        logger.warning(f"No target weights for available tickers on {date}")
                        return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
            # Some tickers might not have prices - filter to available ones
            if is_first_rebalance:
                logger.warning(f"Price lookup failed: {e}, checking available tickers")
            available_tickers = [t for t in target_tickers if (date, t) in prices.index]
            if is_first_rebalance:
                logger.info(f"Found {len(available_tickers)} available tickers out of {len(target_tickers)}")
            if not available_tickers:
                logger.warning(f"No prices available for any target ticker on {date}")
                return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
            current_prices = prices.loc[(date, available_tickers)]
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel('date')
            # Filter target_weights to available tickers
            target_weights = target_weights.loc[target_weights.index.intersection(available_tickers)]
            if target_weights.empty:
                logger.warning(f"No target weights for available tickers on {date}")
                return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
        
        # Also get prices for current positions (if any)
        if not self.positions.empty:
            position_tickers = self.positions.index.tolist()
            try:
                position_prices = prices.loc[(date, position_tickers)]
                if isinstance(position_prices.index, pd.MultiIndex):
                    position_prices = position_prices.droplevel('date')
                # Combine with target prices
                current_prices = pd.concat([current_prices, position_prices]).drop_duplicates()
            except (KeyError, IndexError):
                pass  # Use only target prices
        
        # Align positions and prices for calculation
        aligned_positions, aligned_prices = self.positions.align(current_prices, join='outer', fill_value=0.0)
        
        # Current weights (in dollar terms)
        current_market_values = aligned_positions * aligned_prices
        current_weights = current_market_values / total_nav if total_nav > 0 else pd.Series(0.0, index=aligned_positions.index)
        current_weights = current_weights.fillna(0.0)
        
        # Align target_weights with current_prices (only for target tickers)
        # Use inner join to only keep tickers that have both weights and prices
        target_weights_aligned, prices_for_shares = target_weights.align(current_prices, join='inner', fill_value=0.0)
        
        if is_first_rebalance:
            logger.info(f"After alignment: {len(target_weights_aligned)} tickers with both weights and prices")
        
        # Filter out tickers with zero prices
        valid_price_mask = prices_for_shares > 1e-10
        if not valid_price_mask.any():
            logger.warning(f"No valid prices for target weights on {date}. Price range: [{prices_for_shares.min():.2f}, {prices_for_shares.max():.2f}]")
            return {'costs': 0.0, 'turnover': 0.0, 'notional_traded': 0.0}
        
        target_weights_aligned = target_weights_aligned[valid_price_mask]
        prices_for_shares = prices_for_shares[valid_price_mask]
        
        if is_first_rebalance:
            logger.info(f"After filtering valid prices: {len(target_weights_aligned)} tickers, weight sum: {target_weights_aligned.abs().sum():.4f}, price range: [{prices_for_shares.min():.2f}, {prices_for_shares.max():.2f}]")
        
        # Target market values
        target_market_values = target_weights_aligned * total_nav
        
        # Compute current market values for target tickers (align with target_market_values)
        current_market_values_aligned = aligned_positions.loc[target_weights_aligned.index] * aligned_prices.loc[target_weights_aligned.index]
        current_market_values_aligned = current_market_values_aligned.fillna(0.0)
        
        # Compute trades
        trades = target_market_values - current_market_values_aligned
        
        # Compute turnover: 0.5 * sum(|trades|) / NAV
        # This measures the one-way turnover (buy + sell) relative to portfolio size
        turnover = 0.5 * abs(trades).sum() / total_nav if total_nav > 0 else 0.0
        notional_traded = abs(trades).sum()
        
        # Log turnover for debugging (first rebalance only)
        if len(self.trade_log) == 0:
            logger.info(f"🔄 REBALANCING on {date}: Turnover={turnover:.2%}, Notional=${notional_traded:,.0f}, NAV=${total_nav:,.0f}")
        
        # Compute transaction costs
        costs = notional_traded * (self.commission_rate + self.slippage_rate)
        
        # Update positions and cash
        # Calculate new shares: target_market_value / price
        # prices_for_shares and target_market_values are already aligned
        new_shares = target_market_values / prices_for_shares
        new_shares = new_shares.replace([np.inf, -np.inf], 0.0)
        
        if is_first_rebalance:
            logger.info(f"Calculated new_shares: {len(new_shares)} tickers, non-zero: {(new_shares.abs() > 1e-10).sum()}, range: [{new_shares.min():.4f}, {new_shares.max():.4f}]")
            logger.info(f"  Sample: target_market_values range: [{target_market_values.min():.2f}, {target_market_values.max():.2f}], prices range: [{prices_for_shares.min():.2f}, {prices_for_shares.max():.2f}]")
        
        # Remove positions with zero shares (use small threshold)
        self.positions = new_shares[new_shares.abs() > 1e-10]
        
        if is_first_rebalance:
            logger.info(f"Final positions: {len(self.positions)} tickers")
        
        # Update cash: subtract costs and net trade value
        # For long-short: trades.sum() should be ~0 (long + short cancel out)
        net_trade_value = trades.sum()
        self.cash = self.cash - costs - net_trade_value
        
        # Log first rebalance for debugging
        is_first_rebalance = len(self.trade_log) == 0
        logger.info(f"After rebalance on {date}: positions={len(self.positions)}, cash=${self.cash:.2f}, net_trade=${net_trade_value:.2f}, costs=${costs:.2f}, valid_prices={valid_price_mask.sum()}")
        if len(self.positions) > 0:
            logger.info(f"  Sample positions (first 3): {dict(list(self.positions.head(3).items()))}")
            logger.info(f"  Total shares: long={self.positions[self.positions > 0].sum():.2f}, short={abs(self.positions[self.positions < 0].sum()):.2f}")
        elif is_first_rebalance:
            logger.warning(f"  No positions created! target_market_values range: [{target_market_values.min():.2f}, {target_market_values.max():.2f}], prices range: [{prices_for_shares.min():.2f}, {prices_for_shares.max():.2f}], valid_price_mask sum: {valid_price_mask.sum()}")
        
        # Log trades
        for ticker, trade_value in trades.items():
            if abs(trade_value) > 0.01:  # Minimum trade size
                self.trade_log.append({
                    'date': date,
                    'ticker': ticker,
                    'trade_value': trade_value,
                    'shares': new_shares.get(ticker, 0.0) - self.positions.get(ticker, 0.0)
                })
        
        return {
            'costs': costs,
            'turnover': turnover,
            'notional_traded': notional_traded
        }
    
    def run(
        self,
        price_data: pd.DataFrame,
        target_weights: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Run backtest.
        
        Parameters
        ----------
        price_data : pd.DataFrame
            Price data with (date, ticker) MultiIndex, must have 'adj_close'
        target_weights : pd.DataFrame
            Target weights with (date, ticker) MultiIndex
        rebalance_dates : pd.DatetimeIndex
            Rebalancing dates
        
        Returns
        -------
        pd.DataFrame
            Backtest results with columns: ['nav', 'returns', 'long_value', 'short_value', 'turnover']
        """
        logger.info(f"Starting backtest from {price_data.index.get_level_values('date').min()} to {price_data.index.get_level_values('date').max()}")
        
        prices = price_data['adj_close']
        trading_dates = price_data.index.get_level_values('date').unique().sort_values()
        
        # Initialize
        self.positions = pd.Series(dtype=float)
        self.cash = self.initial_capital
        self.nav_series = []
        self.return_series = []
        self.long_value_series = []
        self.short_value_series = []
        self.turnover_series = []
        self.trade_log = []
        
        previous_nav = self.initial_capital
        
        # FIXED: Convert rebalance_dates to normalized string format for robust matching
        rebalance_strs = set([pd.Timestamp(d).normalize().strftime('%Y-%m-%d') for d in rebalance_dates])
        
        # Daily loop
        for date in trading_dates:
            # Update portfolio value (before rebalancing)
            portfolio_values = self._compute_portfolio_value(prices, date)
            current_nav = portfolio_values['total']
            
            # Compute returns
            daily_return = (current_nav - previous_nav) / previous_nav if previous_nav > 0 else 0.0
            
            # FIXED: Normalize date to string format for matching
            date_str = pd.Timestamp(date).normalize().strftime('%Y-%m-%d')
            
            # Log first few days and around rebalance dates for debugging
            if len(self.nav_series) < 5 or date_str in rebalance_strs:
                logger.info(f"Day {date_str}: NAV=${current_nav:.2f}, cash=${self.cash:.2f}, positions={len(self.positions)}, market_value=${portfolio_values['market_value']:.2f}")
            
            # Rebalance if needed (BEFORE recording metrics, so turnover is recorded on the same day)
            daily_turnover = 0.0
            if date_str in rebalance_strs:
                # Get target weights for this date
                try:
                    date_weights = target_weights.loc[(date, slice(None)), :]
                    if not date_weights.empty:
                        # Get the first column (weight column)
                        if isinstance(date_weights, pd.DataFrame):
                            # Try to get 'weight' column, or first column
                            if 'weight' in date_weights.columns:
                                date_target_weights = date_weights['weight']
                            else:
                                date_target_weights = date_weights.iloc[:, 0]
                        else:
                            date_target_weights = date_weights
                        
                        # Ensure index is ticker only (not MultiIndex)
                        if isinstance(date_target_weights.index, pd.MultiIndex):
                            date_target_weights = date_target_weights.droplevel('date')
                        
                        # Execute trades
                        logger.info(f"🔄 REBALANCING on {date_str}: {len(date_target_weights)} positions, total weight: {date_target_weights.abs().sum():.4f}")
                        try:
                            trade_info = self._execute_trades(date_target_weights, prices, date)
                            daily_turnover = trade_info['turnover']
                            logger.info(f"Trade executed: turnover={daily_turnover:.2%}, costs=${trade_info['costs']:.2f}, notional=${trade_info['notional_traded']:,.0f}")
                            
                            # Log NAV after rebalance
                            portfolio_values_after = self._compute_portfolio_value(prices, date)
                            logger.info(f"NAV after rebalance on {date}: ${portfolio_values_after['total']:.2f}, positions={len(self.positions)}")
                        except Exception as e:
                            logger.error(f"Error executing trades on {date}: {e}", exc_info=True)
                            daily_turnover = 0.0
                    else:
                        logger.debug(f"No target weights for rebalance date {date}")
                        daily_turnover = 0.0
                except (KeyError, IndexError) as e:
                    logger.debug(f"No target weights available for rebalance date {date}: {e}")
                    daily_turnover = 0.0
            
            # Record metrics (AFTER rebalancing, so turnover is recorded on the same day)
            self.nav_series.append(current_nav)
            self.return_series.append(daily_return)
            self.long_value_series.append(portfolio_values['long'])
            self.short_value_series.append(portfolio_values['short'])
            self.turnover_series.append(daily_turnover)
            
            # Debug: log turnover on rebalance dates
            if daily_turnover > 0:
                logger.debug(f"Recorded turnover {daily_turnover:.2%} for {date_str}, series_len={len(self.turnover_series)}")
            
            previous_nav = current_nav
        
        # Verify series lengths match
        if len(self.turnover_series) != len(trading_dates):
            logger.warning(f"Turnover series length mismatch: {len(self.turnover_series)} vs {len(trading_dates)}")
            # Pad or truncate to match
            if len(self.turnover_series) < len(trading_dates):
                self.turnover_series.extend([0.0] * (len(trading_dates) - len(self.turnover_series)))
            else:
                self.turnover_series = self.turnover_series[:len(trading_dates)]
        
        # Debug: check turnover values
        non_zero_turnover = [t for t in self.turnover_series if t > 0]
        if len(non_zero_turnover) > 0:
            logger.info(f"Turnover stats: {len(non_zero_turnover)} rebalance days, avg={sum(non_zero_turnover)/len(non_zero_turnover):.2%}, max={max(non_zero_turnover):.2%}")
            logger.info(f"Sample turnover values: {self.turnover_series[:10]}")
        else:
            logger.warning("No non-zero turnover values found!")
            logger.warning(f"Turnover series length: {len(self.turnover_series)}, first 10 values: {self.turnover_series[:10]}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'nav': self.nav_series,
            'returns': self.return_series,
            'long_value': self.long_value_series,
            'short_value': self.short_value_series,
            'turnover': self.turnover_series
        }, index=trading_dates)
        
        # Verify turnover was saved correctly
        if 'turnover' in results.columns:
            saved_non_zero = (results['turnover'] > 0).sum()
            logger.info(f"Saved turnover: {saved_non_zero} non-zero values out of {len(results)} total")
            if saved_non_zero == 0 and len(non_zero_turnover) > 0:
                logger.error(f"ERROR: Turnover values not saved correctly! Series had {len(non_zero_turnover)} non-zero, but DataFrame has {saved_non_zero}")
        
        logger.info(f"Backtest complete. Final NAV: ${results['nav'].iloc[-1]:,.2f}")
        
        return results
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)


