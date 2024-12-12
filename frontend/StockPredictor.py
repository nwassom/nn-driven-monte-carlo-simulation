import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.nn_model import NN_model
from utils.data_fetch import fetch_stock_data

def predict_stock_price(ticker, time_period):
    
    data = fetch_stock_data(ticker)

    model, scaler = NN_model(data)

    last_data = data[-time_period:].values
    input_shape = model.input_shape[1:]
    last_data_reshaped = last_data.reshape(1, -1, input_shape[1])

    predicted_price = model.predict(last_data_reshaped)

    print(f"PP: {predicted_price}")
    print(f"Scaler: {scaler}")

    # Key Change: Reshape predicted_price before inverse_transform
    predicted_price = predicted_price.reshape(predicted_price.shape[0], -1) # Reshape to (n_samples, n_features)

    predicted_price = scaler.inverse_transform(predicted_price)

    print(predicted_price)
    
    return predicted_price[0][0]

# Create the main window
class StockPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Price Predictor")

        # Create input fields
        ticker_label = tk.Label(self, text="Stock Ticker:")
        self.ticker_entry = tk.Entry(self)
        time_period_label = tk.Label(self, text="Time Period (days):")
        self.time_period_entry = tk.Entry(self)

        # Create a button to trigger prediction
        predict_button = tk.Button(self, text="Predict", command=self.predict)

        # Create a label to display results
        self.result_label = tk.Label(self, text="")

        # Layout the widgets
        ticker_label.grid(row=0, column=0)
        self.ticker_entry.grid(row=0, column=1)
        time_period_label.grid(row=1, column=0)
        self.time_period_entry.grid(row=1, column=1)
        predict_button.grid(row=2, columnspan=2)
        self.result_label.grid(row=3, columnspan=2)

    def predict(self):
        ticker = self.ticker_entry.get()
        time_period = int(self.time_period_entry.get())
        predictedprice = predict_stock_price(ticker, time_period)
        self.result_label.config(text=f"Predicted price: {predicted_price}")

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()