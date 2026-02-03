"""
Portfolio constraints module.

Applies various constraints to portfolio weights: sector limits, single-name limits,
beta neutrality, etc.
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..utils.config import BACKTEST_CONFIG
from ..utils.logging import get_logger

logger = get_logger("portfolio.constraints")


def apply_single_name_constraints(
    weights: pd.Series,
    max_weight: Optional[float] = None
) -> pd.Series:
    """
    Apply single-name position limits.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    max_weight : float, optional
        Maximum absolute weight per stock (default from config)
    
    Returns
    -------
    pd.Series
        Constrained weights
    """
    if max_weight is None:
        max_weight = BACKTEST_CONFIG.max_weight
    
    # Clip weights
    weights_constrained = weights.clip(lower=-max_weight, upper=max_weight)
    
    # Check if clipping occurred
    if not weights_constrained.equals(weights):
        n_clipped = (weights_constrained != weights).sum()
        logger.debug(f"Clipped {n_clipped} positions to max_weight={max_weight}")
    
    return weights_constrained


def apply_sector_constraints(
    weights: pd.Series,
    sector_info: pd.Series,
    max_sector_exposure: Optional[float] = None
) -> pd.Series:
    """
    Apply sector exposure constraints.
    
    Limits net sector exposure (long - short) to within ±max_sector_exposure.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    sector_info : pd.Series
        Sector classification indexed by ticker
    max_sector_exposure : float, optional
        Maximum absolute net sector exposure (default from config)
    
    Returns
    -------
    pd.Series
        Constrained weights
    """
    if max_sector_exposure is None:
        max_sector_exposure = BACKTEST_CONFIG.max_sector_exposure
    
    if sector_info is None or sector_info.empty:
        logger.debug("No sector information provided, skipping sector constraints")
        return weights
    
    # Align weights and sector info
    aligned_weights, aligned_sectors = weights.align(sector_info, join='inner')
    
    if aligned_weights.empty:
        return weights
    
    # Compute sector exposures
    sector_exposures = aligned_weights.groupby(aligned_sectors).sum()
    
    # Check for violations
    violations = sector_exposures[abs(sector_exposures) > max_sector_exposure]
    
    if len(violations) > 0:
        logger.debug(f"Sector exposure violations: {violations.to_dict()}")
        
        # Scale down weights in violating sectors
        scale_factors = pd.Series(1.0, index=aligned_weights.index)
        
        for sector, exposure in violations.items():
            sector_mask = aligned_sectors == sector
            scale = max_sector_exposure / abs(exposure)
            scale_factors.loc[sector_mask] = scale
        
        aligned_weights = aligned_weights * scale_factors
        logger.debug(f"Scaled weights in {len(violations)} sectors")
    
    # Merge back with original weights (for tickers not in sector_info)
    result = weights.copy()
    result.loc[aligned_weights.index] = aligned_weights
    
    return result


def compute_portfolio_beta(
    weights: pd.Series,
    market_cap: pd.Series,
    betas: Optional[pd.Series] = None
) -> float:
    """
    Compute portfolio beta to market.
    
    If betas are provided, use them directly. Otherwise, use market cap as proxy
    (larger stocks tend to have beta closer to 1).
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    market_cap : pd.Series
        Market cap indexed by ticker
    betas : pd.Series, optional
        Stock betas indexed by ticker
    
    Returns
    -------
    float
        Portfolio beta
    """
    # Align weights with market cap or betas
    if betas is not None:
        aligned_weights, aligned_betas = weights.align(betas, join='inner')
        portfolio_beta = (aligned_weights * aligned_betas).sum()
    else:
        # Use market cap as proxy (simplified - in production, use actual beta)
        aligned_weights, aligned_mc = weights.align(market_cap, join='inner')
        # Normalize market cap to get rough beta proxy
        mc_normalized = aligned_mc / aligned_mc.median()
        mc_normalized = mc_normalized.clip(0.5, 2.0)  # Cap between 0.5 and 2.0
        portfolio_beta = (aligned_weights * mc_normalized).sum()
    
    return portfolio_beta


def apply_beta_neutrality(
    weights: pd.Series,
    market_cap: pd.Series,
    betas: Optional[pd.Series] = None,
    target_beta: float = 0.0
) -> pd.Series:
    """
    Adjust portfolio weights to achieve beta neutrality.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    market_cap : pd.Series
        Market cap indexed by ticker
    betas : pd.Series, optional
        Stock betas indexed by ticker
    target_beta : float
        Target portfolio beta (default 0.0 for beta-neutral)
    
    Returns
    -------
    pd.Series
        Adjusted weights
    """
    if not BACKTEST_CONFIG.beta_neutral:
        return weights
    
    current_beta = compute_portfolio_beta(weights, market_cap, betas)
    
    if abs(current_beta - target_beta) < 0.01:  # Already close enough
        return weights
    
    # Simple adjustment: scale long/short legs
    # This is simplified; in production, use more sophisticated optimization
    long_weights = weights[weights > 0].sum()
    short_weights = abs(weights[weights < 0].sum())
    
    if long_weights > 0 and short_weights > 0:
        # Adjust ratio to target beta
        # This is a heuristic; proper implementation would use optimization
        beta_diff = current_beta - target_beta
        
        if beta_diff > 0:
            # Reduce long leg or increase short leg
            scale = 1.0 - (beta_diff / (2 * long_weights))
            weights[weights > 0] = weights[weights > 0] * scale
        else:
            # Increase long leg or reduce short leg
            scale = 1.0 + (abs(beta_diff) / (2 * short_weights))
            weights[weights < 0] = weights[weights < 0] * scale
        
        logger.debug(f"Adjusted weights for beta neutrality: {current_beta:.3f} -> {compute_portfolio_beta(weights, market_cap, betas):.3f}")
    
    return weights


def apply_constraints(
    weights: pd.Series,
    sector_info: Optional[pd.Series] = None,
    market_cap: Optional[pd.Series] = None,
    betas: Optional[pd.Series] = None
) -> pd.Series:
    """
    Apply all constraints to portfolio weights.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    sector_info : pd.Series, optional
        Sector classification indexed by ticker
    market_cap : pd.Series, optional
        Market cap indexed by ticker
    betas : pd.Series, optional
        Stock betas indexed by ticker
    
    Returns
    -------
    pd.Series
        Fully constrained weights
    """
    result = weights.copy()
    
    # Apply constraints in order
    
    # 1. Single-name limits
    result = apply_single_name_constraints(result)
    
    # 2. Sector constraints
    if sector_info is not None:
        result = apply_sector_constraints(result, sector_info)
    
    # 3. Beta neutrality
    if BACKTEST_CONFIG.beta_neutral and market_cap is not None:
        result = apply_beta_neutrality(result, market_cap, betas)
    
    # Renormalize to maintain dollar neutrality (approximately)
    long_sum = result[result > 0].sum()
    short_sum = abs(result[result < 0].sum())
    
    if long_sum > 0 and short_sum > 0:
        # Normalize both legs independently
        result[result > 0] = result[result > 0] / long_sum
        result[result < 0] = result[result < 0] / short_sum
    
    return result


