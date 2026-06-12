import tensorflow as tf
from tensorflow.keras import models, layers 
import numpy as np

data=np.array([
    121,131,141,125,124,125,234,231,212,123,167,141,
    151,211,311,211,111,222,101,102,112,113,231,120,
    121,231,111,123,234,112,115,222,333,221,223,167
    ])

data= data/ max(data)

X ,y= [], []
for i in range(len(data) -5):
    X.append(data[i:i+5])
    y.append(data[i+5])

X=np.array(X)
y=np.array(y)

X=X.reshape(X.shape[0],X.shape[1], 1)

split = int(0.8* len(X))
x_train, x_test= X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = models.Sequential([
    layers.LSTM(32, input_shape=(5,1)),
    layers.Dense(1)   
])

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=2, verbose=0)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("SGD MAE:", acc)





