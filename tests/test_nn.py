from utils.data_fetch import fetch_stock_data
from models.nn_model import NN_model

ticker = "AAPL"

print(f"Fetching data fors {ticker}")

data = fetch_stock_data(ticker)

NN_model(data)