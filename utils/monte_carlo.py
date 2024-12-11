import numpy as np

def monte_carlo_simulation(data, num_simulations=1000, forecast_days=252):

	# Gets log returns, calculates its mean, and volatility
	log_returns = data['Log Returns']
	mean_returns = log_returns.mean()
	volatility = log_returns.std()

	# Arr of simulation results
	simulations = []

	for _ in range(num_simulations):

		path = [data['Close'].iloc[-1]]

		for _ in range(forecast_days):
			random_return = np.random.normal(mean_returns, volatility)
			path.append(path[-1] * np.exp(random_return))

		simulations.append(path)


	return np.array(simulations).squeeze()
