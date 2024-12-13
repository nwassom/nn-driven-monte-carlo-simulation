import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.callbacks import EarlyStopping

def NN_model(scaled_data, scaler):

    # Create sequences of data
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data)-seq_length-1):
            X.append(data[i:(i+seq_length)])
            y.append(data[i+seq_length, 0])  # Assuming 'Close' is the target
        return np.array(X), np.array(y)

    seq_length = 60 
    X, y = create_sequences(scaled_data, seq_length)

    # Shuffle data
    np.random.seed(42) 
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Split data
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    X_train, X_val_test = X[:train_size], X[train_size:]
    y_train, y_val_test = y[:train_size], y[train_size:]
    X_val, X_test = X_val_test[:val_size], X_val_test[val_size:]
    y_val, y_test = y_val_test[:val_size], y_val_test[val_size:]

    # Reshape data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2])) 
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True, recurrent_dropout=0.2)) 
    model.add(LSTM(units=50, recurrent_dropout=0.2)) 
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
             validation_data=(X_val, y_val), callbacks=[early_stop]) 

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    return model