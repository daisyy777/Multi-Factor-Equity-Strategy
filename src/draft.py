import yfinance as yf
df = yf.download("AAPL", start="2016-10-01", end="2023-12-31", auto_adjust=False)
print(df.head())
print(df.columns)

