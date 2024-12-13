import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

'''
    Prepares the features from the data

    returns the scaled features and the scaler

    needs to be able to save different forms of data to a csv or xml as well

'''
def prepare_features(data):

    # Ensure 'Close' is a Series
    data['Close'] = data['Close'].squeeze()  # Convert single-column DataFrame to Series
    print(f"Type of 'Close' after squeeze: {type(data['Close'])}, Shape: {data['Close'].shape}")

    # Moving Averages
    data['MA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Moving Average
    data['MA_20'] = data['Close'].rolling(window=20).mean()  # 20-day Moving Average
    print(f"Type of 'MA_20': {type(data['MA_20'])}, Shape: {data['MA_20'].shape}")

    # Exponential Moving Average
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()  # 10-day EMA

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))  # RSI calculation

    # Bollinger Bands
    rolling_std = data['Close'].rolling(window=20).std().squeeze()  # Force rolling_std to Series
    print(f"Type of 'rolling_std' after squeeze: {type(rolling_std)}, Shape: {rolling_std.shape}")

    data['BB_Upper'] = data['MA_20'] + 2 * rolling_std  # Upper Band
    data['BB_Lower'] = data['MA_20'] - 2 * rolling_std  # Lower Band

    # Price Changes
    data['Daily_Return'] = data['Close'].pct_change()  # Daily Return
    data['Volatility'] = data['Close'].rolling(window=20).std()  # Rolling Volatility

    # Seasonality Features
    data['Day_of_Week'] = data.index.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
    data['Month'] = data.index.month  # Month of the year

    # Handle missing values (using interpolation as an example)
    data.interpolate(method='linear', inplace=True)

    data.fillna(method='bfill', inplace=True)
    
    # Drop rows with NaN values (force cleanup)
    data.dropna(inplace=True)


    # Select features
    features = ['Close', 'MA_50', 'MA_20', 'EMA_10', 'RSI', 'BB_Upper', 'BB_Lower', 
                'Daily_Return', 'Volatility', 'Day_of_Week', 'Month']
    data = data[features]

    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    scaled_data_df = pd.DataFrame(scaled_data, columns=features)

    # Save to CSV
    scaled_data_df.to_csv('scaled_data.csv', index=False)

    return scaled_data, scaler


def get_features(data):
    return prepare_features(data)