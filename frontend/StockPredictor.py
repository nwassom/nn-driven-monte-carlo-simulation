import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.nn_model import NN_model
from utils.data_fetch import fetch_stock_data

def predict_stock_price(ticker, time_period):
    # Fetch and preprocess data
    data = fetch_stock_data(ticker)

    # Train and predict using the NN model
    model, scaler = NN_model(data)

    # Prepare data for prediction
    last_data = data[-time_period:].values.reshape(1, time_period, data.shape[1])

    print("Shape of last_data:", last_data.shape)
    print("Expected input shape:", model.input_shape)

    # Predict the future price
    predicted_price = model.predict(last_data)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Display the result
    result_label.config(text=f"Predicted price: {predicted_price[0][0]}")

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
        predict_stock_price(ticker, time_period)

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()