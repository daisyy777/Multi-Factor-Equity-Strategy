"""Quick script to check composite scores"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.utils.config import BACKTEST_CONFIG
from src.data_loader import load_universe, download_price_data
from src.data_preprocess import preprocess_data
from src.factors import compute_value_factor_panel, compute_momentum_factor_panel
from src.factor_combiner import compute_composite_scores_panel
from src.backtester import get_rebalance_dates

print('Loading data...')
universe_df = load_universe()
tickers = universe_df['ticker'].tolist()
data_start = pd.Timestamp(BACKTEST_CONFIG.start_date) - pd.DateOffset(months=15)
price_data = download_price_data(tickers, data_start.strftime('%Y-%m-%d'), BACKTEST_CONFIG.end_date, cache=True)
clean_prices, _, market_cap, _ = preprocess_data(price_data, None)
clean_prices = clean_prices.loc[clean_prices.index.get_level_values('date') >= pd.Timestamp(BACKTEST_CONFIG.start_date)]
trading_dates = clean_prices.index.get_level_values('date').unique().sort_values()
rebalance_dates = get_rebalance_dates(trading_dates, BACKTEST_CONFIG.rebalance_frequency, BACKTEST_CONFIG.rebalance_day)

print('Computing factors...')
value_factor = compute_value_factor_panel(clean_prices, clean_prices[[]], market_cap, rebalance_dates)
momentum_factor = compute_momentum_factor_panel(clean_prices, rebalance_dates)

print(f'Value factor shape: {value_factor.shape}, NaN count: {value_factor.isna().sum().sum()}')
print(f'Momentum factor shape: {momentum_factor.shape}, NaN count: {momentum_factor.isna().sum().sum()}')

all_factors = pd.concat({'value_factor': value_factor, 'momentum_factor': momentum_factor}, axis=1)
all_factors.columns = all_factors.columns.droplevel(1) if isinstance(all_factors.columns, pd.MultiIndex) else all_factors.columns

print(f'All factors shape: {all_factors.shape}')
print(f'All factors columns: {all_factors.columns.tolist()}')
print(f'All factors NaN count: {all_factors.isna().sum().sum()}')

# Check first few dates
print('\nFirst 5 dates factor data:')
for i, date in enumerate(rebalance_dates[:5]):
    date_factors = all_factors.loc[(date, slice(None)), :]
    print(f'{date}: value NaN={date_factors["value_factor"].isna().sum()}, momentum NaN={date_factors["momentum_factor"].isna().sum()}')

print('\nComputing composite scores...')
composite_scores = compute_composite_scores_panel(all_factors, returns=None)

print(f'Composite scores shape: {composite_scores.shape}')
print(f'Composite scores columns: {composite_scores.columns.tolist()}')

dates = composite_scores.index.get_level_values('date').unique()
print(f'\nChecking NaN patterns across {len(dates)} dates:')
print('First 10 dates:')
for i, date in enumerate(dates[:10]):
    date_data = composite_scores.loc[(date, slice(None)), :]
    if isinstance(date_data, pd.DataFrame):
        scores = date_data.iloc[:, 0]
    else:
        scores = date_data
    valid = scores.notna().sum()
    nan_count = scores.isna().sum()
    if valid > 0:
        print(f'{date}: {valid} valid, {nan_count} NaN, range: [{scores.min():.4f}, {scores.max():.4f}]')
    else:
        print(f'{date}: {valid} valid, {nan_count} NaN, ALL NaN')

print('\nLast 10 dates:')
for i, date in enumerate(dates[-10:]):
    date_data = composite_scores.loc[(date, slice(None)), :]
    if isinstance(date_data, pd.DataFrame):
        scores = date_data.iloc[:, 0]
    else:
        scores = date_data
    valid = scores.notna().sum()
    nan_count = scores.isna().sum()
    if valid > 0:
        print(f'{date}: {valid} valid, {nan_count} NaN, range: [{scores.min():.4f}, {scores.max():.4f}]')
    else:
        print(f'{date}: {valid} valid, {nan_count} NaN, ALL NaN')

# Find first date with all NaN
print('\nFinding first date with all NaN:')
for date in dates:
    date_data = composite_scores.loc[(date, slice(None)), :]
    if isinstance(date_data, pd.DataFrame):
        scores = date_data.iloc[:, 0]
    else:
        scores = date_data
    if scores.isna().all():
        print(f'First all-NaN date: {date}')
        # Check factors for this date
        date_factors = all_factors.loc[(date, slice(None)), :]
        print(f'  Value factor NaN: {date_factors["value_factor"].isna().sum()}/{len(date_factors)}')
        print(f'  Momentum factor NaN: {date_factors["momentum_factor"].isna().sum()}/{len(date_factors)}')
        break
