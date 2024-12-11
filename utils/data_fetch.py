import yfinance as yf
import pandas as pd
import numpy as np

'''
	Function to fetch the data from a given stock as well as a date range
'''
def fetch_stock_data(ticker, start_date='2021-01-01', end_date='2024-12-10'):

	# downloads data
	data = yf.download(ticker, start_date, end_date)

	if 'Close' not in data.columns:
		raise ValueError(f"Ticker `{ticker}` does not container 'Close' data.")

	# Calculates the Log Returns
	data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

	# Remove missing vals
	data = data.dropna()

	return data
