import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

def NN_model(data):

    # Feature Engineering
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Handle missing values (optional, if necessary)
    # data.fillna(method='ffill', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences of data
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data)-seq_length-1):
            X.append(data[i:(i+seq_length), 0])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = 60  # Adjust as needed
    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training, validation, and testing sets
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)

    X_train, X_val_test = X[:train_size], X[train_size:]
    y_train, y_val_test = y[:train_size], y[train_size:]

    X_val, X_test = X_val_test[:val_size], X_val_test[val_size:]
    y_val, y_test = y_val_test[:val_size], y_val_test[val_size:]

    # Reshape data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    return model, scaler