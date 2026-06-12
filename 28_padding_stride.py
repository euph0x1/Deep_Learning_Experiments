import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1,28,28,1)

# -------- SAME padding, stride 1 --------
model1 = models.Sequential([
    layers.Conv2D(1, (3,3), strides=(1,1), padding='same', input_shape=(28,28,1))
])

out1 = model1.predict(x_train[:1])
print("Same padding shape:", out1.shape)


# -------- VALID padding, stride 2 --------
model2 = models.Sequential([
    layers.Conv2D(1, (3,3), strides=(2,2), padding='valid', input_shape=(28,28,1))
])

out2 = model2.predict(x_train[:1])
print("Valid padding shape:", out2.shape)