import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

data = np.array([i for i in range(1,20)]) / 20.0

X = []
for i in range(len(data)-5):
    X.append(data[i:i+5])

X = np.array(X).reshape(-1,5,1)

# LSTM returning hidden states
lstm = layers.LSTM(4, return_sequences=True)

outputs = lstm(X)

print("Hidden states shape:", outputs.shape)
print("Sample hidden states:\n", outputs[0])