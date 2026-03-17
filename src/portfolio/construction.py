"""
Portfolio construction module.

Builds long-short portfolios from composite factor scores, with an optional
turnover constraint that caps one-way turnover per rebalance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from ..utils.config import BACKTEST_CONFIG
from ..utils.logging import get_logger
from .constraints import apply_constraints

logger = get_logger("portfolio.construction")


# ---------------------------------------------------------------------------
# Turnover constraint
# ---------------------------------------------------------------------------

def apply_turnover_constraint(
    new_weights: pd.Series,
    previous_weights: pd.Series,
    max_turnover: float,
) -> pd.Series:
    """
    Scale trades so that one-way turnover does not exceed *max_turnover*.

    One-way turnover is defined as::

        0.5 * sum(|new_weights - prev_weights|)

    If the proposed rebalance is within the limit the weights are returned
    unchanged.  Otherwise all trades are scaled back proportionally so the
    limit is exactly met.

    Parameters
    ----------
    new_weights : pd.Series
        Target weights (ticker index).
    previous_weights : pd.Series
        Weights from the previous period (ticker index).  May be None or
        empty on the first rebalance.
    max_turnover : float
        Maximum allowed one-way turnover (e.g. 0.30 = 30 %).
        Set to 1.0 (or higher) to disable.

    Returns
    -------
    pd.Series
        Constrained target weights.
    """
    if previous_weights is None or previous_weights.empty:
        return new_weights

    if max_turnover >= 1.0:
        return new_weights

    # Align on the union of tickers.
    all_tickers = new_weights.index.union(previous_weights.index)
    new_w = new_weights.reindex(all_tickers, fill_value=0.0)
    prev_w = previous_weights.reindex(all_tickers, fill_value=0.0)

    trades = new_w - prev_w
    one_way_turnover = trades.abs().sum() / 2.0

    if one_way_turnover <= max_turnover + 1e-9:
        return new_weights  # Already within limit

    # Scale trades proportionally.
    scale = max_turnover / one_way_turnover
    constrained_w = prev_w + trades * scale

    logger.debug(
        f"Turnover constraint applied: proposed={one_way_turnover:.2%}, "
        f"limit={max_turnover:.2%}, scale={scale:.4f}"
    )

    # Drop near-zero positions to keep the portfolio sparse.
    return constrained_w[constrained_w.abs() > 1e-6]


# ---------------------------------------------------------------------------
# Stock selection
# ---------------------------------------------------------------------------

def select_long_short_stocks(
    scores: pd.Series,
    long_pct: float,
    short_pct: float,
) -> Tuple[List[str], List[str]]:
    """
    Select stocks for long / short legs based on composite scores.
    """
    scores_clean = scores.dropna().sort_values(ascending=False)
    if scores_clean.empty:
        return [], []

    if len(scores_clean) > 0 and not isinstance(scores_clean.index[0], str):
        logger.error(
            "scores index is not strings — cannot construct portfolio. "
            f"Sample: {scores_clean.index[:3].tolist()}"
        )
        return [], []

    n = len(scores_clean)
    n_long = max(1, int(n * long_pct))
    n_short = max(1, int(n * short_pct))

    return (
        scores_clean.head(n_long).index.tolist(),
        scores_clean.tail(n_short).index.tolist(),
    )


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_equal_weights(
    long_tickers: List[str],
    short_tickers: List[str],
) -> pd.Series:
    """
    Equal-weight long/short portfolio.
    Long leg sums to +1, short leg sums to -1.
    """
    weights = pd.Series(0.0, index=long_tickers + short_tickers)
    if long_tickers:
        weights.loc[long_tickers] = 1.0 / len(long_tickers)
    if short_tickers:
        weights.loc[short_tickers] = -1.0 / len(short_tickers)
    return weights


def compute_score_proportional_weights(
    scores: pd.Series,
    long_tickers: List[str],
    short_tickers: List[str],
    max_weight: Optional[float] = None,
) -> pd.Series:
    """Score-proportional weights, normalised to \u00b11 per leg."""
    if max_weight is None:
        max_weight = BACKTEST_CONFIG.max_weight

    weights = pd.Series(0.0, index=long_tickers + short_tickers)

    if long_tickers:
        long_scores = scores.loc[long_tickers]
        long_w = (long_scores - long_scores.min() + 0.01)
        long_w = (long_w / long_w.sum()).clip(upper=max_weight)
        weights.loc[long_tickers] = long_w / long_w.sum()

    if short_tickers:
        short_scores = scores.loc[short_tickers]
        short_w = -(short_scores - short_scores.max() - 0.01)
        short_w = (short_w / short_w.abs().sum()).clip(lower=-max_weight)
        weights.loc[short_tickers] = short_w / short_w.abs().sum() * -1

    return weights


# ---------------------------------------------------------------------------
# Single-date portfolio construction
# ---------------------------------------------------------------------------

def construct_portfolio(
    scores: pd.Series,
    eligible_tickers: Optional[List[str]] = None,
    previous_weights: Optional[pd.Series] = None,
    sector_info: Optional[pd.Series] = None,
    market_cap: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Construct target portfolio weights from composite scores for one date.
    """
    if eligible_tickers is not None:
        scores = scores.loc[scores.index.intersection(eligible_tickers)]
    if scores.empty:
        return pd.Series(dtype=float)

    long_tickers, short_tickers = select_long_short_stocks(
        scores, BACKTEST_CONFIG.long_pct, BACKTEST_CONFIG.short_pct
    )
    if not long_tickers and not short_tickers:
        return pd.Series(dtype=float)

    if BACKTEST_CONFIG.weighting_scheme == "score_proportional":
        weights = compute_score_proportional_weights(scores, long_tickers, short_tickers)
    else:
        weights = compute_equal_weights(long_tickers, short_tickers)

    # Ensure string index
    if len(weights) > 0 and not isinstance(weights.index[0], str):
        weights.index = weights.index.astype(str)
    weights.index.name = "ticker"

    weights = apply_constraints(
        weights, sector_info=sector_info, market_cap=market_cap
    )

    if len(weights) > 0 and not isinstance(weights.index[0], str):
        weights.index = weights.index.astype(str)
    weights.index.name = "ticker"

    return weights


# ---------------------------------------------------------------------------
# Panel portfolio construction
# ---------------------------------------------------------------------------

def construct_portfolio_panel(
    composite_scores: pd.DataFrame,
    eligible_tickers: Optional[pd.DataFrame] = None,
    sector_info: Optional[pd.DataFrame] = None,
    market_cap: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Construct target portfolio weights for all rebalancing dates.

    Applies the turnover constraint (from BACKTEST_CONFIG.max_turnover_per_rebalance)
    when previous weights are available.
    """
    logger.info("Constructing portfolios for all rebalancing dates")

    max_turnover = BACKTEST_CONFIG.max_turnover_per_rebalance
    all_weights: list = []
    dates = composite_scores.index.get_level_values("date").unique()
    previous_weights: Optional[pd.Series] = None

    for date in dates:
        # ── Extract scores for this date ────────────────────────────────
        try:
            date_slice = composite_scores.loc[(date, slice(None)), :]
            if isinstance(date_slice, pd.DataFrame):
                date_scores = (
                    date_slice["composite_score"]
                    if "composite_score" in date_slice.columns
                    else date_slice.iloc[:, 0]
                )
            else:
                date_scores = date_slice
        except (KeyError, IndexError) as e:
            logger.debug(f"Error extracting scores for {date}: {e}")
            continue

        # Ensure ticker-only index
        if isinstance(date_scores.index, pd.MultiIndex):
            ticker_vals = date_scores.index.get_level_values(
                "ticker" if "ticker" in date_scores.index.names else 1
            )
            date_scores = date_scores.copy()
            date_scores.index = ticker_vals
            date_scores.index.name = "ticker"

        if date_scores.empty or date_scores.isna().all():
            continue

        date_scores_clean = date_scores.dropna()
        if date_scores_clean.empty:
            continue

        # Fix non-string index (defensive)
        if len(date_scores_clean) > 0 and not isinstance(
            date_scores_clean.index[0], str
        ):
            try:
                full = composite_scores.loc[(date, slice(None)), :]
                if isinstance(full.index, pd.MultiIndex) and "ticker" in full.index.names:
                    ticker_names = full.index.get_level_values("ticker")
                    non_na_mask = date_scores.notna()
                    t_clean = ticker_names[non_na_mask]
                    if len(t_clean) == len(date_scores_clean):
                        date_scores_clean.index = t_clean
                        date_scores_clean.index.name = "ticker"
                    else:
                        continue
                else:
                    continue
            except Exception as e:
                logger.error(f"Cannot fix ticker index on {date}: {e}")
                continue

        if len([d for d in dates if d <= date]) <= 5:
            logger.info(
                f"Date {date}: {len(date_scores_clean)} valid scores, "
                f"range [{date_scores_clean.min():.4f}, {date_scores_clean.max():.4f}], "
                f"sample tickers: {date_scores_clean.index[:3].tolist()}"
            )

        # ── Eligible tickers ─────────────────────────────────────────────
        eligible = None
        if eligible_tickers is not None:
            try:
                eligible = (
                    eligible_tickers.loc[(date, slice(None))]
                    .index.get_level_values("ticker")
                    .tolist()
                )
            except (KeyError, AttributeError):
                eligible = eligible_tickers

        # ── Sector / market-cap helpers ────────────────────────────────
        sector = None
        if sector_info is not None:
            sector = sector_info.loc[(date, slice(None)), "sector"]

        mc = None
        if market_cap is not None:
            mc_slice = market_cap.loc[
                market_cap.index.get_level_values("date") <= date
            ]
            mc = mc_slice.groupby("ticker").last()

        # ── Construct unconstrained weights ──────────────────────────────
        weights = construct_portfolio(
            date_scores_clean,
            eligible_tickers=eligible,
            previous_weights=previous_weights,
            sector_info=sector,
            market_cap=mc,
        )
        if weights.empty:
            continue

        # ── Apply turnover constraint ───────────────────────────────────
        if previous_weights is not None and not previous_weights.empty:
            weights = apply_turnover_constraint(
                weights, previous_weights, max_turnover
            )

        if weights.empty:
            continue

        # Ensure string index
        if len(weights) > 0 and not isinstance(weights.index[0], str):
            weights.index = weights.index.astype(str)
        weights.index.name = "ticker"

        weights_df = weights.to_frame().rename(columns={weights.name or 0: "weight"})
        weights_df["date"] = date
        weights_df = weights_df.reset_index().set_index(["date", "ticker"])
        all_weights.append(weights_df)

        previous_weights = weights.copy()

    if not all_weights:
        logger.warning("No portfolio weights generated for any date.")
        empty_idx = pd.MultiIndex.from_product(
            [dates, []], names=["date", "ticker"]
        )
        return pd.DataFrame(index=empty_idx, columns=["weight"])

    result = pd.concat(all_weights).sort_index()
    logger.info(f"Constructed portfolios: {len(result)} weight records")
    return result
