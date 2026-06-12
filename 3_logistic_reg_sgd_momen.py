import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=5000)
# Pad sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

# Reduce size (faster)
x_train = x_train[:10000]
y_train = y_train[:10000]

# -------- Model (Logistic Regression) --------
model = models.Sequential([
    layers.Flatten(input_shape=(100,)),   # convert sequence → vector
    layers.Dense(1, activation='sigmoid')
])

# -------- SGD with Momentum --------
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3, verbose=0)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", acc)