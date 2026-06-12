import tensorflow as tf
from tensorflow.keras import layers, models, initializers

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# -------- Variance Scaling Initialization --------
print("\nVariance Scaling Initialization")

initializer = initializers.VarianceScaling(
    #scale=2.0,              # like He initialization
    #mode='fan_in',          # based on input units
    #distribution='normal'   # normal distribution
)

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu', kernel_initializer=initializer),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=0)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", acc)