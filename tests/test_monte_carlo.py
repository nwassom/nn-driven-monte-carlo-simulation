import pytest
import pandas as pd
import numpy as np
from utils import monte_carlo_simulation, fetch_stock_data

# Fetch the data
ticker = 'AAPL'
data = fetch_stock_data(ticker)

def test_monte_carlo_shape():
    num_simulations = 1000
    forecast_days = 252

    # Run Monte Carlo simulation
    simulations = monte_carlo_simulation(data, num_simulations, forecast_days)

    # Assert the correct shape of the simulations
    assert simulations.shape == (num_simulations, forecast_days + 1), \
        f"Expected shape ({num_simulations}, {forecast_days + 1}), but got {simulations.shape}"

def test_monte_carlo_last_price():
    num_simulations = 1000
    forecast_days = 252
    initial_price = data['Close'].iloc[-1].values[0]

    # Run Monte Carlo simulation
    simulations = monte_carlo_simulation(data, num_simulations, forecast_days)

    # Check that final prices are within a reasonable range of the initial price
    final_prices = simulations[:, -1]
    assert np.allclose(final_prices, initial_price, rtol=1.0), \
        f"Last prices deviate too much from the initial price. " \
        f"Initial price: {initial_price}, Final prices: {final_prices[:5]}..."
