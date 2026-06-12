import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load data
(x_train, _), _ = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0

# Reshape
x_train = x_train.reshape(-1, 28, 28, 1)

# -------- Model (FIXED) --------
model = models.Sequential([
    tf.keras.Input(shape=(28,28,1)),   # 🔥 THIS FIXES YOUR ERROR
    layers.Conv2D(4, (3,3), activation='relu')
])

# -------- Feature extraction --------
feature_model = tf.keras.Model(
    inputs=model.inputs,   # note: inputs (not input)
    outputs=model.layers[0].output
)

# Get feature maps
feature_maps = feature_model.predict(x_train[:1])

# -------- Visualization --------
for i in range(4):
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.title(f"Feature Map {i+1}")
    plt.axis('off')
    plt.show()