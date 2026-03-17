"""
Quality factor construction module.

Implements quality-based factors including ROE, leverage, and earnings stability.
Prefers ``book_value`` (total equity) over the per-share × shares product.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

from ..utils.config import FACTOR_CONFIG
from ..utils.logging import get_logger

logger = get_logger("factors.quality")


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def compute_roe(net_income: pd.Series, book_value: pd.Series) -> pd.Series:
    """Return on Equity = NetIncome / BookValueOfEquity."""
    aligned_ni, aligned_bv = net_income.align(book_value, join="inner")
    roe = aligned_ni / aligned_bv
    return roe.replace([np.inf, -np.inf], np.nan).rename("roe")


def compute_leverage(
    total_liabilities: pd.Series, total_assets: pd.Series
) -> pd.Series:
    """Leverage = TotalLiabilities / TotalAssets."""
    aligned_liab, aligned_assets = total_liabilities.align(total_assets, join="inner")
    leverage = aligned_liab / aligned_assets
    return leverage.replace([np.inf, -np.inf], np.nan).rename("leverage")


def compute_earnings_stability(
    eps: pd.Series, date: pd.Timestamp, quarters: int = 8
) -> pd.Series:
    """
    Earnings stability = negative variance of EPS over the last *quarters* quarters.
    Lower variance → higher quality → higher score.
    """
    eps_slice = eps.loc[eps.index.get_level_values("date") <= date]
    eps_df = eps_slice.unstack("ticker")
    if len(eps_df) >= quarters:
        recent_eps = eps_df.iloc[-quarters:]
    else:
        recent_eps = eps_df
    return (-recent_eps.var()).rename("earnings_stability")


def compute_quality_factor(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    date: pd.Timestamp,
) -> pd.Series:
    """
    Compute quality factor z-score for *date*.

    Combines ROE, leverage (negative), and earnings stability.
    Uses total ``book_value`` when available.
    """
    fund_slice = fundamental_data.loc[
        fundamental_data.index.get_level_values("date") <= date
    ]

    latest_net_income = (
        fund_slice.groupby("ticker")["net_income"].last()
        if "net_income" in fund_slice.columns
        else pd.Series(dtype=float)
    )

    # ── Book value (prefer total equity) ──────────────────────────────────
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
        latest_book_value = pd.Series(dtype=float)

    latest_assets = (
        fund_slice.groupby("ticker")["total_assets"].last()
        if "total_assets" in fund_slice.columns
        else pd.Series(dtype=float)
    )
    latest_liabilities = (
        fund_slice.groupby("ticker")["total_liabilities"].last()
        if "total_liabilities" in fund_slice.columns
        else pd.Series(dtype=float)
    )

    roe = compute_roe(latest_net_income, latest_book_value)
    leverage = compute_leverage(latest_liabilities, latest_assets)

    if "eps" in fundamental_data.columns:
        stability = compute_earnings_stability(
            fundamental_data["eps"], date, FACTOR_CONFIG.quality_eps_stability_quarters
        )
    else:
        stability = pd.Series(dtype=float)

    components = [c.dropna() for c in [roe, leverage, stability] if not c.empty]
    if not components:
        return pd.Series(dtype=float, name="quality_factor")

    common_tickers = components[0].index
    for c in components[1:]:
        common_tickers = common_tickers.intersection(c.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float, name="quality_factor")

    def _align(s: pd.Series) -> pd.Series:
        if s.empty:
            return pd.Series(index=common_tickers, dtype=float)
        return s.reindex(common_tickers).fillna(s.median())

    roe_a = _align(roe)
    leverage_a = _align(leverage)
    stability_a = _align(stability)

    if FACTOR_CONFIG.winsorize_enabled:
        lo, hi = FACTOR_CONFIG.quality_winsorize_pct
        roe_a = winsorize(roe_a, lo, hi)
        leverage_a = winsorize(leverage_a, lo, hi)
        stability_a = winsorize(stability_a, lo, hi)

    def _std(s: pd.Series) -> pd.Series:
        if FACTOR_CONFIG.standardization_method == "rank":
            return s.rank(pct=True) - 0.5
        non_na = s.dropna()
        if len(non_na) < 2:
            return pd.Series(0.0, index=s.index)
        return (
            pd.Series(zscore(non_na, nan_policy="omit"), index=non_na.index)
            .reindex(s.index)
            .fillna(0)
        )

    roe_z = _std(roe_a)
    leverage_z = _std(leverage_a)
    stability_z = _std(stability_a)

    w_roe = abs(FACTOR_CONFIG.quality_roe_weight)
    w_lev = abs(FACTOR_CONFIG.quality_leverage_weight)
    w_sta = abs(FACTOR_CONFIG.quality_stability_weight)

    quality_score = (
        FACTOR_CONFIG.quality_roe_weight * roe_z
        + FACTOR_CONFIG.quality_leverage_weight * leverage_z
        + FACTOR_CONFIG.quality_stability_weight * stability_z
    ) / (w_roe + w_lev + w_sta)

    quality_score = _std(quality_score)
    return quality_score.rename("quality_factor")


def compute_quality_factor_panel(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    market_cap: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute quality factor for all rebalancing dates."""
    logger.info(f"Computing quality factor for {len(rebalance_dates)} dates")
    all_scores = []

    for date in rebalance_dates:
        try:
            scores = compute_quality_factor(
                price_data, fundamental_data, market_cap, date
            )
            if scores.empty:
                logger.debug(f"No quality factor data for {date}")
                continue
            scores = scores.to_frame()
            scores["date"] = date
            scores = scores.reset_index().set_index(["date", "ticker"])
            all_scores.append(scores)
        except Exception as e:
            logger.warning(f"Error computing quality factor for {date}: {e}")

    if not all_scores:
        logger.warning("No quality factor scores computed for any date.")
        empty_idx = pd.MultiIndex.from_product(
            [rebalance_dates, []], names=["date", "ticker"]
        )
        return pd.DataFrame(index=empty_idx, columns=["quality_factor"])

    result = pd.concat(all_scores).sort_index()
    logger.info(f"Computed quality factor: {len(result)} records")
    return result
