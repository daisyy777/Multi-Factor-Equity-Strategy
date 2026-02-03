import pandas as pd
import json

df = pd.read_parquet('results/backtest_results.parquet')
print('=== TURNOVER FIX VERIFICATION ===')
print(f'Total days: {len(df)}')
print(f'Non-zero turnover: {(df["turnover"] > 0).sum()}')
print(f'Max turnover: {df["turnover"].max()}')
if (df["turnover"] > 0).any():
    non_zero = df[df["turnover"] > 0]["turnover"]
    print(f'Mean turnover (non-zero): {non_zero.mean():.2%}')
    print(f'Max turnover: {non_zero.max():.2%}')
    print(f'\nFirst 10 rebalance dates with turnover:')
    print(df[df["turnover"] > 0][["turnover"]].head(10))
else:
    print('WARNING: No non-zero turnover found!')

metrics = json.load(open('results/metrics.json'))
print(f'\n=== METRICS ===')
print(f'Annual Turnover: {metrics.get("annual_turnover", 0)*100:.2f}%')
print(f'Long Leg Return: {metrics.get("long_leg_annualized_return", 0)*100:.2f}%')
print(f'Short Leg Return: {metrics.get("short_leg_annualized_return", 0)*100:.2f}%')
