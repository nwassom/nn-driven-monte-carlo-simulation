import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.nn_model import NN_model
from utils.data_fetch import fetch_stock_data
from features import get_features

def predict_stock_price(ticker, time_period):
    raw_data = fetch_stock_data(ticker)

    engineered_data, scaler = get_features(raw_data)
    features_count = engineered_data.shape[1]

    model = NN_model(engineered_data, scaler)

    # Take the last 'time_period' rows from the selected features
    last_data = engineered_data[-time_period:]

    # Debugging logs to verify the data
    print(f"Shape of last_data (before reshape): {last_data.shape}")

    # Check if last_data has enough values to reshape into (1, timesteps, features_count)
    expected_shape = time_period * features_count  # This should be time_period * features_count
    if last_data.size != expected_shape:
        raise ValueError(f"Last data size ({last_data.size}) does not match the expected size ({expected_shape}) "
                         f"for {time_period} timesteps and {features_count} features.\n"
                         f"Last_data shape: {last_data.shape}, Time Period: {time_period}")

    # Reshape the data for LSTM model input
    timesteps = time_period  # Use the exact number of timesteps you have
    last_data_reshaped = last_data.reshape(1, timesteps, features_count)  # Reshape into (1, timesteps, features_count)

    # Predict the stock price using the model
    predicted_price = model.predict(last_data_reshaped)

    # Print predicted price for debugging
    print(f"Predicted price (before inverse scaling): {predicted_price}")

    # Reshape predicted_price before inverse_transform
    predicted_price = predicted_price.reshape(predicted_price.shape[0], -1)  # (n_samples, n_features)

    padded_prediction = np.zeros((1, scaler.n_features_in_))
    padded_prediction[:, 0] = predicted_price[:, 0]

    inverse_transformed = scaler.inverse_transform(padded_prediction)
    # Reverse the scaling transformation

    predicted_price = inverse_transformed[:, 0]

    # Print the final predicted price after inverse scaling
    print(f"Predicted price (after inverse scaling): {predicted_price}")

    return predicted_price[0] # Return the predicted price (first element)


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
        predicted_price = predict_stock_price(ticker, time_period)
        self.result_label.config(text=f"Predicted price: {predicted_price}")

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()