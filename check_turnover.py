import pandas as pd
import json

df = pd.read_parquet('results/backtest_results.parquet')
print('Turnover column stats:')
print(f'Length: {len(df)}')
print(f'Non-zero count: {(df["turnover"] > 0).sum()}')
print(f'Max: {df["turnover"].max()}')
print(f'Mean: {df["turnover"].mean()}')
print(f'First 10 values:', df["turnover"].head(10).tolist())
print(f'Last 10 values:', df["turnover"].tail(10).tolist())

# Check rebalance dates
non_zero = df[df["turnover"] > 0]
if len(non_zero) > 0:
    print(f'\nNon-zero turnover dates: {len(non_zero)}')
    print(non_zero[["turnover"]].head(10))

metrics = json.load(open('results/metrics.json'))
print(f'\nMetrics:')
print(f'Annual Turnover: {metrics["annual_turnover"]*100:.2f}%')
print(f'Long Leg Return: {metrics["long_leg_annualized_return"]*100:.2f}%')
print(f'Short Leg Return: {metrics["short_leg_annualized_return"]*100:.2f}%')
