"""
Value factor construction module.

Implements value-based factors including Book-to-Market (BTM) and
Earnings-to-Price (E/P) ratios.

Note on data: prefers the ``book_value`` column (total stockholders' equity)
over the derived ``book_value_per_share * shares_outstanding`` product
because total equity from the balance sheet is more accurate.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.value")


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def compute_btm(
    market_cap: pd.Series,
    book_value: pd.Series,
) -> pd.Series:
    """
    Book-to-Market = BookValueOfEquity / MarketCap.

    Parameters
    ----------
    market_cap : pd.Series  (ticker index)
    book_value : pd.Series  (ticker index)
    """
    aligned_mc, aligned_bv = market_cap.align(book_value, join="inner")
    btm = aligned_bv / aligned_mc
    btm = btm.replace([np.inf, -np.inf], np.nan)
    btm = btm[btm > 0]
    return btm.rename("btm")


def compute_ep(
    price: pd.Series,
    eps: pd.Series,
) -> pd.Series:
    """
    Earnings-to-Price = EarningsPerShare / PricePerShare.
    """
    aligned_price, aligned_eps = price.align(eps, join="inner")
    ep = aligned_eps / aligned_price
    ep = ep.replace([np.inf, -np.inf], np.nan)
    return ep.rename("ep")


def compute_value_factor(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    date: pd.Timestamp,
) -> pd.Series:
    """
    Compute value factor z-score for *date*.

    Combines BTM and E/P into a composite value score.
    Uses ``book_value`` (total equity) when available; falls back to
    ``book_value_per_share * shares_outstanding``.
    """
    price_slice = price_data.loc[price_data.index.get_level_values("date") <= date]
    fund_slice = fundamental_data.loc[
        fundamental_data.index.get_level_values("date") <= date
    ]
    mc_slice = market_cap.loc[market_cap.index.get_level_values("date") <= date]

    latest_prices = price_slice.groupby("ticker")["adj_close"].last()
    latest_mc = mc_slice.groupby("ticker").last()

    # ── Book value (prefer total equity over per-share × shares) ──────────
    if "book_value" in fund_slice.columns:
        latest_book_value = fund_slice.groupby("ticker")["book_value"].last()
    elif (
        "book_value_per_share" in fund_slice.columns
        and "shares_outstanding" in fund_slice.columns
    ):
        bv_ps = fund_slice.groupby("ticker")["book_value_per_share"].last()
        shares = fund_slice.groupby("ticker")["shares_outstanding"].last()
        latest_book_value = bv_ps * shares
    else:
        logger.warning("No book value data available.")
        latest_book_value = pd.Series(index=latest_mc.index, dtype=float)

    # ── EPS ───────────────────────────────────────────────────────────────
    if "eps" in fund_slice.columns:
        latest_eps = fund_slice.groupby("ticker")["eps"].last()
    else:
        latest_eps = pd.Series(index=latest_prices.index, dtype=float)

    btm = compute_btm(latest_mc, latest_book_value)
    ep = compute_ep(latest_prices, latest_eps)

    # Fallback when no fundamental data at all
    if btm.empty and ep.empty:
        value_proxy = 1.0 / latest_mc
        value_proxy = value_proxy.replace([np.inf, -np.inf], np.nan).dropna()
        if value_proxy.empty:
            return pd.Series(dtype=float, name="value_factor")
        if FACTOR_CONFIG.standardization_method == "rank":
            value_score = value_proxy.rank(pct=True) - 0.5
        else:
            value_score = pd.Series(
                zscore(value_proxy.dropna(), nan_policy="omit"),
                index=value_proxy.index,
            ).fillna(0)
        return value_score.rename("value_factor")

    # ── Align components ──────────────────────────────────────────────────
    if btm.empty:
        common_tickers = ep.index
        btm_aligned = pd.Series(0.0, index=common_tickers)
        ep_aligned = ep.loc[common_tickers]
    elif ep.empty:
        common_tickers = btm.index
        btm_aligned = btm.loc[common_tickers]
        ep_aligned = pd.Series(0.0, index=common_tickers)
    else:
        common_tickers = btm.index.intersection(ep.index)
        btm_aligned = btm.loc[common_tickers]
        ep_aligned = ep.loc[common_tickers]

    if len(common_tickers) == 0:
        return pd.Series(dtype=float, name="value_factor")

    if FACTOR_CONFIG.winsorize_enabled:
        btm_aligned = winsorize(
            btm_aligned,
            FACTOR_CONFIG.value_winsorize_pct[0],
            FACTOR_CONFIG.value_winsorize_pct[1],
        )
        ep_aligned = winsorize(
            ep_aligned,
            FACTOR_CONFIG.value_winsorize_pct[0],
            FACTOR_CONFIG.value_winsorize_pct[1],
        )

    # Cross-sectional standardisation
    def _zscore(s: pd.Series) -> pd.Series:
        non_na = s.dropna()
        if len(non_na) < 2:
            return pd.Series(0.0, index=s.index)
        z = pd.Series(
            zscore(non_na, nan_policy="omit"), index=non_na.index
        ).reindex(s.index).fillna(0)
        return z

    if FACTOR_CONFIG.standardization_method == "rank":
        btm_z = btm_aligned.rank(pct=True) - 0.5
        ep_z = ep_aligned.rank(pct=True) - 0.5
    else:
        btm_z = _zscore(btm_aligned)
        ep_z = _zscore(ep_aligned)

    value_score = (
        FACTOR_CONFIG.value_btm_weight * btm_z
        + FACTOR_CONFIG.value_ep_weight * ep_z
    ) / (FACTOR_CONFIG.value_btm_weight + FACTOR_CONFIG.value_ep_weight)

    if FACTOR_CONFIG.standardization_method == "rank":
        value_score = value_score.rank(pct=True) - 0.5
    else:
        value_score = _zscore(value_score)

    return value_score.rename("value_factor")


def compute_value_factor_panel(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute value factor for all rebalancing dates."""
    logger.info(f"Computing value factor for {len(rebalance_dates)} dates")
    all_scores = []

    for date in rebalance_dates:
        try:
            scores = compute_value_factor(
                price_data, fundamental_data, market_cap, date
            )
            if scores.empty:
                continue
            scores = scores.to_frame()
            scores["date"] = date
            scores = scores.reset_index()
            if "index" in scores.columns and "ticker" not in scores.columns:
                scores = scores.rename(columns={"index": "ticker"})
            scores = scores.set_index(["date", "ticker"])
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"Error computing value factor for {date}: {e}")

    if not all_scores:
        raise ValueError("No value factor scores computed")

    result = pd.concat(all_scores).sort_index()
    logger.info(f"Computed value factor: {len(result)} records")
    return result
