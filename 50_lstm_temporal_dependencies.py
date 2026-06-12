import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Data
data = np.array([i for i in range(1,20)]) / 20.0

X = []
for i in range(len(data)-5):
    X.append(data[i:i+5])

X = np.array(X).reshape(-1,5,1)

# LSTM returning sequences
lstm = layers.LSTM(4, return_sequences=True)

outputs = lstm(X)

print("Output shape:", outputs.shape)
print("Sample output:\n", outputs[0])