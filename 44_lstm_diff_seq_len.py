import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Simple data (time series)
data = np.array([i for i in range(1,101)]) / 100.0

def create_dataset(seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

for seq_len in [3,5]:
    print(f"\nSequence Length: {seq_len}")

    X, y = create_dataset(seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = models.Sequential([
        layers.LSTM(32, input_shape=(seq_len,1)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    loss = model.evaluate(X, y, verbose=0)
    print("Loss:", loss)