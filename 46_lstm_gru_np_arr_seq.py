import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Data
data = np.array([i for i in range(1,101)]) / 100.0

X, y = [], []
for i in range(len(data)-5):
    X.append(data[i:i+5])
    y.append(data[i+5])

X = np.array(X).reshape(-1,5,1)
y = np.array(y)

# -------- LSTM --------
start = time.time()

model1 = models.Sequential([
    #hyperparameter  layers.LSTM(units, input_shape=(5,1)), units different
    layers.LSTM(32, input_shape=(5,1)),
    layers.Dense(1)
])

model1.compile(optimizer='adam', loss='mse')
model1.fit(X, y, epochs=5, verbose=0)

time_lstm = time.time() - start


# -------- GRU --------
start = time.time()

model2 = models.Sequential([
    layers.GRU(32, input_shape=(5,1)),
    layers.Dense(1)
])

model2.compile(optimizer='adam', loss='mse')
model2.fit(X, y, epochs=5, verbose=0)

time_gru = time.time() - start

print("LSTM Time:", time_lstm)
print("GRU Time:", time_gru)