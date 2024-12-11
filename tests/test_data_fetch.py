import pytest
from utils.data_fetch import fetch_stock_data

ticker = "AAPL"

print(f"Fetching data fors {ticker}")

data = fetch_stock_data(ticker)

print(data.head())
print(data.info())