import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1,28,28,1)

# Simple CNN
model = models.Sequential([
    layers.Conv2D(4, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Train a bit
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')
model.fit(x_train[:5000], y_train[:5000], epochs=2, verbose=0)

# Get filters (weights)
filters = model.layers[0].get_weights()[0]

# Visualize filters
for i in range(4):
    plt.imshow(filters[:,:,0,i], cmap='gray')
    plt.title(f"Filter {i+1}")
    plt.axis('off')
    plt.show()