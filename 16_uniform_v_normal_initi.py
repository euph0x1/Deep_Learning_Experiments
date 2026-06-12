import tensorflow as tf
from tensorflow.keras import layers, models
import time
# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

results = {}

# -------- UNIFORM Initialization --------
print("\nUniform Initialization")

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu', kernel_initializer='random_uniform'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=0)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
results['Uniform'] = acc
print("Accuracy:", acc)


# -------- NORMAL Initialization --------
print("\nNormal Initialization")

start = time.time()

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu', kernel_initializer='random_normal'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=0)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

end = time.time()

results['Normal'] = acc
print(loss)
print("Accuracy:", acc)

print("total time:", end -start)

# -------- Final Comparison --------
print("\nFinal Comparison:")
for k, v in results.items():
    print(f"{k} → {v:.4f}")

#kernel_initializer='random_uniform'
#kernel_initializer='random_normal'