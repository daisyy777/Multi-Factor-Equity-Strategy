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
    day: int = -1,
) -> pd.DatetimeIndex:
    """
    Get rebalancing dates based on frequency.

    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
    frequency : str  "monthly" or "quarterly"
    day : int  -1 = last trading day of period
    """
    if frequency == "monthly":
        period = "M"
    elif frequency == "quarterly":
        period = "Q"
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    date_df = pd.DataFrame({"date": trading_dates})
    date_df["period"] = date_df["date"].dt.to_period(period)

    if day == -1:
        rebalance_dates = date_df.groupby("period")["date"].last().values
    else:
        date_df["day_of_month"] = date_df["date"].dt.day
        date_df_filtered = date_df[date_df["day_of_month"] <= day]
        rebalance_dates = date_df_filtered.groupby("period")["date"].last().values

    return pd.DatetimeIndex(rebalance_dates).intersection(trading_dates)


class BacktestEngine:
    """Event-driven backtest engine."""

    def __init__(
        self,
        initial_capital: float = None,
        commission_rate: float = None,
        slippage_rate: float = None,
    ):
        self.initial_capital = initial_capital or BACKTEST_CONFIG.initial_capital
        self.commission_rate = commission_rate or BACKTEST_CONFIG.commission_rate
        self.slippage_rate = slippage_rate or BACKTEST_CONFIG.slippage_rate

        self.positions = pd.Series(dtype=float)
        self.cash = self.initial_capital
        self.nav_series: list = []
        self.return_series: list = []
        self.long_value_series: list = []
        self.short_value_series: list = []
        self.turnover_series: list = []
        self.trade_log: list = []

    def _compute_portfolio_value(
        self, prices: pd.Series, date: pd.Timestamp
    ) -> Dict[str, float]:
        """Compute current portfolio value."""
        if self.positions.empty:
            return {
                "total": self.cash,
                "long": 0.0,
                "short": 0.0,
                "cash": self.cash,
                "market_value": 0.0,
            }

        position_tickers = self.positions.index.tolist()
        try:
            current_prices = prices.loc[(date, position_tickers)]
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel("date")
        except (KeyError, IndexError):
            available = [
                t for t in position_tickers if (date, t) in prices.index
            ]
            if not available:
                return {
                    "total": self.cash,
                    "long": 0.0,
                    "short": 0.0,
                    "cash": self.cash,
                    "market_value": 0.0,
                }
            current_prices = prices.loc[(date, available)]
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel("date")
            self.positions = self.positions.loc[available]

        aligned_pos, aligned_px = self.positions.align(
            current_prices, join="inner", fill_value=0.0
        )
        market_value = (aligned_pos * aligned_px).sum()

        long_mask = aligned_pos > 0
        short_mask = aligned_pos < 0
        long_value = (
            (aligned_pos[long_mask] * aligned_px[long_mask]).sum()
            if long_mask.any()
            else 0.0
        )
        short_value = (
            abs((aligned_pos[short_mask] * aligned_px[short_mask]).sum())
            if short_mask.any()
            else 0.0
        )

        return {
            "total": self.cash + market_value,
            "long": long_value,
            "short": short_value,
            "cash": self.cash,
            "market_value": market_value,
        }

    def _execute_trades(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Execute trades to move from current positions to target weights.

        Returns
        -------
        Dict with keys: 'costs', 'turnover', 'notional_traded'
        """
        # Determine once whether this is the first rebalance (used for
        # debug logging throughout the function).
        is_first_rebalance = len(self.trade_log) == 0

        portfolio_values = self._compute_portfolio_value(prices, date)
        total_nav = portfolio_values["total"]

        if total_nav <= 0:
            logger.warning(f"Portfolio value is non-positive on {date}")
            return {"costs": 0.0, "turnover": 0.0, "notional_traded": 0.0}

        target_tickers = target_weights.index.tolist()
        if is_first_rebalance:
            logger.info(
                f"Getting prices for {len(target_tickers)} target tickers on {date}"
            )

        # ── Fetch prices for target tickers ──────────────────────────────
        try:
            current_prices = prices.loc[(date, target_tickers)]
            if isinstance(current_prices.index, pd.MultiIndex):
                current_prices = current_prices.droplevel("date")
            if is_first_rebalance:
                logger.info(
                    f"Got prices for {len(current_prices)} tickers, "
                    f"range: [{current_prices.min():.2f}, {current_prices.max():.2f}]"
                )
        except (KeyError, IndexError):
            available_dates = prices.index.get_level_values("date").unique()
            available_dates = available_dates[available_dates <= date]
            if len(available_dates) == 0:
                logger.warning(f"No trading dates on or before {date}")
                return {"costs": 0.0, "turnover": 0.0, "notional_traded": 0.0}

            closest_date = available_dates.max()
            if is_first_rebalance:
                logger.info(f"Using closest trading date: {closest_date}")
            try:
                current_prices = prices.loc[(closest_date, target_tickers)]
                if isinstance(current_prices.index, pd.MultiIndex):
                    current_prices = current_prices.droplevel("date")
            except (KeyError, IndexError):
                available = [
                    t for t in target_tickers
                    if (closest_date, t) in prices.index
                ]
                if not available:
                    logger.warning(
                        f"No prices for any target ticker on {date}"
                    )
                    return {"costs": 0.0, "turnover": 0.0, "notional_traded": 0.0}
                current_prices = prices.loc[(closest_date, available)]
                if isinstance(current_prices.index, pd.MultiIndex):
                    current_prices = current_prices.droplevel("date")
                target_weights = target_weights.loc[
                    target_weights.index.intersection(available)
                ]
                if target_weights.empty:
                    return {"costs": 0.0, "turnover": 0.0, "notional_traded": 0.0}

        # Include prices for existing positions too
        if not self.positions.empty:
            try:
                pos_prices = prices.loc[(date, self.positions.index.tolist())]
                if isinstance(pos_prices.index, pd.MultiIndex):
                    pos_prices = pos_prices.droplevel("date")
                current_prices = pd.concat([current_prices, pos_prices]).drop_duplicates()
            except (KeyError, IndexError):
                pass

        aligned_pos, aligned_px = self.positions.align(
            current_prices, join="outer", fill_value=0.0
        )
        current_mv = aligned_pos * aligned_px
        current_weights = (
            current_mv / total_nav if total_nav > 0
            else pd.Series(0.0, index=aligned_pos.index)
        ).fillna(0.0)

        target_w_aligned, prices_for_shares = target_weights.align(
            current_prices, join="inner", fill_value=0.0
        )

        if is_first_rebalance:
            logger.info(
                f"After alignment: {len(target_w_aligned)} tickers "
                f"with both weights and prices"
            )

        valid_px_mask = prices_for_shares > 1e-10
        if not valid_px_mask.any():
            logger.warning(f"No valid prices for target weights on {date}")
            return {"costs": 0.0, "turnover": 0.0, "notional_traded": 0.0}

        target_w_aligned = target_w_aligned[valid_px_mask]
        prices_for_shares = prices_for_shares[valid_px_mask]

        target_mv = target_w_aligned * total_nav
        current_mv_aligned = (
            aligned_pos.reindex(target_w_aligned.index, fill_value=0.0)
            * aligned_px.reindex(target_w_aligned.index, fill_value=0.0)
        )

        trades = target_mv - current_mv_aligned
        turnover = 0.5 * trades.abs().sum() / total_nav if total_nav > 0 else 0.0
        notional_traded = trades.abs().sum()

        if is_first_rebalance:
            logger.info(
                f"REBALANCING on {date}: Turnover={turnover:.2%}, "
                f"Notional=${notional_traded:,.0f}, NAV=${total_nav:,.0f}"
            )

        costs = notional_traded * (self.commission_rate + self.slippage_rate)

        new_shares = target_mv / prices_for_shares
        new_shares = new_shares.replace([np.inf, -np.inf], 0.0)
        self.positions = new_shares[new_shares.abs() > 1e-10]

        if is_first_rebalance:
            logger.info(f"Final positions: {len(self.positions)} tickers")

        net_trade_value = trades.sum()
        self.cash = self.cash - costs - net_trade_value

        # NOTE: is_first_rebalance is NOT reassigned here (that was a bug
        # in the previous version where a duplicate assignment appeared
        # after this point, overwriting the correct value).
        logger.info(
            f"After rebalance on {date}: positions={len(self.positions)}, "
            f"cash=${self.cash:.2f}, net_trade=${net_trade_value:.2f}, "
            f"costs=${costs:.2f}, valid_prices={valid_px_mask.sum()}"
        )
        if len(self.positions) > 0:
            logger.info(
                f"  Sample positions (first 3): "
                f"{dict(list(self.positions.head(3).items()))}"
            )
            logger.info(
                f"  Total shares: long={self.positions[self.positions > 0].sum():.2f}, "
                f"short={abs(self.positions[self.positions < 0].sum()):.2f}"
            )
        elif is_first_rebalance:
            logger.warning(
                f"  No positions created! "
                f"target_mv range: [{target_mv.min():.2f}, {target_mv.max():.2f}], "
                f"prices range: [{prices_for_shares.min():.2f}, "
                f"{prices_for_shares.max():.2f}]"
            )

        for ticker, trade_value in trades.items():
            if abs(trade_value) > 0.01:
                self.trade_log.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "trade_value": trade_value,
                        "shares": new_shares.get(ticker, 0.0)
                        - self.positions.get(ticker, 0.0),
                    }
                )

        return {
            "costs": costs,
            "turnover": turnover,
            "notional_traded": notional_traded,
        }

    def run(
        self,
        price_data: pd.DataFrame,
        target_weights: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Parameters
        ----------
        price_data : pd.DataFrame
            (date, ticker) MultiIndex, must have 'adj_close' column.
        target_weights : pd.DataFrame
            (date, ticker) MultiIndex with a 'weight' column.
        rebalance_dates : pd.DatetimeIndex

        Returns
        -------
        pd.DataFrame
            Columns: ['nav', 'returns', 'long_value', 'short_value', 'turnover']
        """
        logger.info(
            f"Starting backtest from "
            f"{price_data.index.get_level_values('date').min()} to "
            f"{price_data.index.get_level_values('date').max()}"
        )

        prices = price_data["adj_close"]
        trading_dates = (
            price_data.index.get_level_values("date").unique().sort_values()
        )

        # Reset state
        self.positions = pd.Series(dtype=float)
        self.cash = self.initial_capital
        self.nav_series = []
        self.return_series = []
        self.long_value_series = []
        self.short_value_series = []
        self.turnover_series = []
        self.trade_log = []

        previous_nav = self.initial_capital

        # Normalise rebalance dates to YYYY-MM-DD strings for O(1) lookup.
        rebalance_strs = {
            pd.Timestamp(d).normalize().strftime("%Y-%m-%d")
            for d in rebalance_dates
        }

        for date in trading_dates:
            portfolio_values = self._compute_portfolio_value(prices, date)
            current_nav = portfolio_values["total"]

            daily_return = (
                (current_nav - previous_nav) / previous_nav
                if previous_nav > 0
                else 0.0
            )

            date_str = pd.Timestamp(date).normalize().strftime("%Y-%m-%d")

            if len(self.nav_series) < 5 or date_str in rebalance_strs:
                logger.info(
                    f"Day {date_str}: NAV=${current_nav:.2f}, "
                    f"cash=${self.cash:.2f}, positions={len(self.positions)}, "
                    f"market_value=${portfolio_values['market_value']:.2f}"
                )

            daily_turnover = 0.0
            if date_str in rebalance_strs:
                try:
                    date_weights = target_weights.loc[(date, slice(None)), :]
                    if not date_weights.empty:
                        if isinstance(date_weights, pd.DataFrame):
                            if "weight" in date_weights.columns:
                                date_target_weights = date_weights["weight"]
                            else:
                                date_target_weights = date_weights.iloc[:, 0]
                        else:
                            date_target_weights = date_weights

                        if isinstance(date_target_weights.index, pd.MultiIndex):
                            date_target_weights = date_target_weights.droplevel(
                                "date"
                            )

                        logger.info(
                            f"REBALANCING on {date_str}: "
                            f"{len(date_target_weights)} positions, "
                            f"total weight: {date_target_weights.abs().sum():.4f}"
                        )
                        try:
                            trade_info = self._execute_trades(
                                date_target_weights, prices, date
                            )
                            daily_turnover = trade_info["turnover"]
                            logger.info(
                                f"Trade executed: turnover={daily_turnover:.2%}, "
                                f"costs=${trade_info['costs']:.2f}, "
                                f"notional=${trade_info['notional_traded']:,.0f}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error executing trades on {date}: {e}",
                                exc_info=True,
                            )
                except (KeyError, IndexError):
                    pass

            self.nav_series.append(current_nav)
            self.return_series.append(daily_return)
            self.long_value_series.append(portfolio_values["long"])
            self.short_value_series.append(portfolio_values["short"])
            self.turnover_series.append(daily_turnover)

            previous_nav = current_nav

        # Pad / truncate to match trading_dates length
        n = len(trading_dates)
        if len(self.turnover_series) < n:
            self.turnover_series.extend([0.0] * (n - len(self.turnover_series)))
        else:
            self.turnover_series = self.turnover_series[:n]

        non_zero = [t for t in self.turnover_series if t > 0]
        if non_zero:
            logger.info(
                f"Turnover stats: {len(non_zero)} rebalance days, "
                f"avg={sum(non_zero)/len(non_zero):.2%}, max={max(non_zero):.2%}"
            )
        else:
            logger.warning("No non-zero turnover values recorded.")

        results = pd.DataFrame(
            {
                "nav": self.nav_series,
                "returns": self.return_series,
                "long_value": self.long_value_series,
                "short_value": self.short_value_series,
                "turnover": self.turnover_series,
            },
            index=trading_dates,
        )

        logger.info(
            f"Backtest complete. Final NAV: ${results['nav'].iloc[-1]:,.2f}"
        )
        return results

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
