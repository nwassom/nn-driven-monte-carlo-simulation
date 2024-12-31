import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import mplfinance as mpf 

from models.nn_model import NN_model
from utils.data_fetch import fetch_stock_data
from features import get_features

'''
    1. Need to reorganize the architecture of this app and files.

    2. Need to implement a way to test the accuracy of prediction using last days stock as the true value vs our prediction

'''
def predict_stock_price(raw_data, time_period):

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
        self.tk.call('tk', 'scaling', 1.0)
        self.title("Stock Price Predictor")

        isChecked = tk.IntVar()

        # Create input fields
        ticker_label = tk.Label(self, text="Stock Ticker:")
        self.ticker_entry = tk.Entry(self)
        time_period_label = tk.Label(self, text="Time Period (days):")
        self.time_period_entry = tk.Entry(self)
        check_label = tk.Label(self, text="Test Accuracy")
        self.test_accuracy_check = tk.Checkbutton(self, text="Enable accuracy check", variable=isChecked, onvalue=1, offvalue=0)

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
        check_label.grid(row=3, columnspan= 2)
        self.test_accuracy_check.grid(row=3, column = 1)

    def predict(self):
        ticker = self.ticker_entry.get()
        time_period = int(self.time_period_entry.get())

        raw_data = fetch_stock_data(ticker)

        predicted_price = predict_stock_price(raw_data, time_period)
        self.result_label.config(text=f"Predicted price: {predicted_price}")

        self.visualize_stock_data(raw_data, predicted_price)

    
    def visualize_stock_data(self, raw_data, predicted_price):
        """
        Visualizes the closing price of the stock data with the predicted price highlighted.

        Args:
            raw_data: A pandas DataFrame containing the stock data.
            predicted_price: The predicted price for the next time period.
        """

        # Extract closing prices
        closing_prices = raw_data['Close']

        # Create the main figure and plot area
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Plot the closing prices with a blue line
        line, = ax.plot(closing_prices.index, closing_prices, label='Closing Price', color='blue')

        # Create a vertical line at the last data point with the predicted price
        x_last = closing_prices.index[-1]
        ax.axvline(x=x_last, color='green', linestyle='dashed', label='Prediction')

        # Plot the predicted price as a green dot
        ax.scatter(x_last, predicted_price, marker='o', color='green', label='Predicted Price')

        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.set_title('Stock Price Visualization')

        # Add legend
        ax.legend()

        # Use Seaborn for stylistic enhancements (optional)
        sns.despine(ax=ax)  # Remove grid lines and top/right spines

        # Enable zooming and panning
        fig.canvas.mpl_connect('scroll_event', lambda event: ax.set_xlim(ax.get_xlim()[0] + event.step * 0.1, ax.get_xlim()[1] + event.step * 0.1))

        # Create an annotation to display date and price
        annotation = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)

        def on_motion(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                annotation.xy = (x, y)
                annotation.set_text(f"Date: {x:.2f}\nPrice: {y:.2f}")
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_motion)

        # Display the plot using tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=4, columnspan=2)  # Place the canvas on the grid


if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()