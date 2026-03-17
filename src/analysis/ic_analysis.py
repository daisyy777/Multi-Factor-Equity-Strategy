"""
Information Coefficient (IC) analysis module.

The IC measures how well a factor score *predicts* the next-period
cross-sectional return.  It is defined as the Spearman rank correlation
between factor z-scores at time *t* and forward returns over the
next rebalance period.

    IC_t = SpearmanCorr(factor_scores_t, forward_returns_{t+1})

Key outputs
-----------
- IC time series per factor
- IC mean, standard deviation, t-stat
- Information Coefficient Information Ratio (ICIR = mean(IC) / std(IC))
- Cumulative IC chart

Rule of thumb (Grinold & Kahn):
  |IC mean| > 0.05  → useful signal
  |IC mean| > 0.02  → marginal signal
  |IC mean| < 0.02  → very weak / likely noise
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger("analysis.ic")


# ---------------------------------------------------------------------------
# Core IC computation
# ---------------------------------------------------------------------------

def compute_ic_series(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    factor_name: str = "factor",
) -> pd.Series:
    """
    Compute IC time series for a single factor.

    Parameters
    ----------
    factor_scores : pd.Series
        (date, ticker) MultiIndex.  Values are cross-sectionally standardised
        factor scores.
    forward_returns : pd.Series
        (date, ticker) MultiIndex.  Values are *forward* returns — i.e. the
        return from the current rebalance date to the next one.
    factor_name : str
        Label used in the returned Series name.

    Returns
    -------
    pd.Series
        IC per rebalance date, indexed by date.
    """
    dates = factor_scores.index.get_level_values("date").unique().sort_values()
    ic_values: list = []
    ic_dates: list = []

    for date in dates:
        try:
            f_scores = factor_scores.loc[(date, slice(None))]
            f_rets = forward_returns.loc[(date, slice(None))]
        except (KeyError, IndexError):
            continue

        # Align on ticker
        if isinstance(f_scores.index, pd.MultiIndex):
            f_scores.index = f_scores.index.get_level_values("ticker")
        if isinstance(f_rets.index, pd.MultiIndex):
            f_rets.index = f_rets.index.get_level_values("ticker")

        common = f_scores.index.intersection(f_rets.index)
        if len(common) < 10:  # Need enough stocks for meaningful correlation
            continue

        x = f_scores.loc[common].values
        y = f_rets.loc[common].values

        # Drop NaNs
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if mask.sum() < 10:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic, _ = stats.spearmanr(x[mask], y[mask])

        if not np.isnan(ic):
            ic_values.append(ic)
            ic_dates.append(date)

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name=factor_name)


def compute_forward_returns(
    price_data: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute forward returns between consecutive rebalance dates.

    ``forward_return_t`` = total return from ``rebalance_dates[t]`` to
    ``rebalance_dates[t+1]``.

    Parameters
    ----------
    price_data : pd.DataFrame
        (date, ticker) MultiIndex with ``adj_close`` column.
    rebalance_dates : pd.DatetimeIndex

    Returns
    -------
    pd.Series
        (date, ticker) MultiIndex.  Date is the *start* of the holding period.
    """
    prices = price_data["adj_close"].unstack("ticker")
    all_results: list = []

    for i in range(len(rebalance_dates) - 1):
        t0 = rebalance_dates[i]
        t1 = rebalance_dates[i + 1]

        # Get closest available prices
        available = prices.index
        t0_avail = available[available <= t0]
        t1_avail = available[available <= t1]
        if len(t0_avail) == 0 or len(t1_avail) == 0:
            continue

        p0 = prices.loc[t0_avail.max()]
        p1 = prices.loc[t1_avail.max()]

        fwd_ret = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan)
        fwd_ret = fwd_ret.dropna()

        fwd_df = fwd_ret.to_frame(name="forward_return")
        fwd_df["date"] = t0
        fwd_df = fwd_df.reset_index().set_index(["date", "ticker"])["forward_return"]
        all_results.append(fwd_df)

    if not all_results:
        empty = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
            name="forward_return",
        )
        return empty

    return pd.concat(all_results).sort_index()


def compute_ic_panel(
    factor_scores: pd.DataFrame,
    price_data: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute IC time series for every factor column in *factor_scores*.

    Parameters
    ----------
    factor_scores : pd.DataFrame
        (date, ticker) MultiIndex.  Each column is one factor.
    price_data : pd.DataFrame
        (date, ticker) MultiIndex with ``adj_close``.
    rebalance_dates : pd.DatetimeIndex

    Returns
    -------
    pd.DataFrame
        IC per factor, indexed by rebalance date.
    """
    logger.info(
        f"Computing IC for {len(factor_scores.columns)} factors "
        f"over {len(rebalance_dates)} rebalance dates"
    )

    fwd_returns = compute_forward_returns(price_data, rebalance_dates)
    ic_dict: dict = {}

    for factor_name in factor_scores.columns:
        f_series = factor_scores[factor_name].dropna()
        ic = compute_ic_series(f_series, fwd_returns, factor_name=factor_name)
        ic_dict[factor_name] = ic
        logger.info(
            f"  {factor_name}: IC mean={ic.mean():.4f}, "
            f"IC std={ic.std():.4f}, ICIR={_icir(ic):.2f}, "
            f"n={len(ic)}"
        )

    return pd.DataFrame(ic_dict)


def _icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC).  Returns 0 on degenerate input."""
    if ic_series.empty or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def summarise_ic(ic_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the IC panel into a table of key statistics.

    Returns
    -------
    pd.DataFrame
        Rows = factors.  Columns = [IC_mean, IC_std, ICIR, IC_tstat,
        pct_positive, n_periods].
    """
    rows: list = []
    for col in ic_panel.columns:
        s = ic_panel[col].dropna()
        if s.empty:
            continue
        n = len(s)
        mean_ic = s.mean()
        std_ic = s.std()
        icir = _icir(s)
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 1 else 0.0
        pct_pos = (s > 0).mean()

        signal_strength = (
            "strong" if abs(mean_ic) > 0.05
            else "marginal" if abs(mean_ic) > 0.02
            else "weak"
        )

        rows.append(
            {
                "factor": col,
                "IC_mean": round(mean_ic, 4),
                "IC_std": round(std_ic, 4),
                "ICIR": round(icir, 2),
                "IC_tstat": round(t_stat, 2),
                "pct_positive": round(pct_pos, 2),
                "n_periods": n,
                "signal": signal_strength,
            }
        )

    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ic_tearsheet(
    ic_panel: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Produce a multi-panel IC tear-sheet.

    Panels
    ------
    1. IC time series per factor
    2. Cumulative IC per factor
    3. IC distribution (histogram)
    4. IC summary table

    Parameters
    ----------
    ic_panel : pd.DataFrame
        Output of :func:`compute_ic_panel`.
    output_path : str, optional
        If given, save the figure to this path.
    show : bool
        If True, call ``plt.show()``.
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available — skipping IC tear-sheet.")
        return

    n_factors = len(ic_panel.columns)
    if n_factors == 0 or ic_panel.empty:
        logger.warning("IC panel is empty — skipping IC tear-sheet.")
        return

    fig = plt.figure(figsize=(14, 4 * n_factors + 4))
    gs = gridspec.GridSpec(n_factors + 1, 2, figure=fig)
    colors = plt.cm.tab10.colors

    for i, factor in enumerate(ic_panel.columns):
        ic = ic_panel[factor].dropna()
        color = colors[i % len(colors)]
        mean_ic = ic.mean()
        icir_val = _icir(ic)

        # IC time series
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.bar(ic.index, ic.values, color=color, alpha=0.6, width=20)
        ax_ts.axhline(0, color="black", linewidth=0.8)
        ax_ts.axhline(mean_ic, color=color, linewidth=1.5, linestyle="--",
                      label=f"mean={mean_ic:.3f}")
        ax_ts.set_title(f"{factor} — IC time series (ICIR={icir_val:.2f})",
                        fontsize=10)
        ax_ts.legend(fontsize=8)
        ax_ts.set_ylabel("IC")

        # Cumulative IC
        ax_cum = fig.add_subplot(gs[i, 1])
        cum_ic = ic.cumsum()
        ax_cum.plot(cum_ic.index, cum_ic.values, color=color, linewidth=1.5)
        ax_cum.axhline(0, color="black", linewidth=0.8)
        ax_cum.set_title(f"{factor} — Cumulative IC", fontsize=10)
        ax_cum.set_ylabel("Cumulative IC")

    # Summary table in last row
    ax_table = fig.add_subplot(gs[n_factors, :])
    ax_table.axis("off")
    summary = summarise_ic(ic_panel)
    if not summary.empty:
        tbl = ax_table.table(
            cellText=summary.values,
            rowLabels=summary.index,
            colLabels=summary.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
    ax_table.set_title("IC Summary", fontsize=10, pad=12)

    fig.suptitle("Factor IC Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"IC tear-sheet saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)
