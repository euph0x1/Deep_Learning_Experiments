import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=5000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)

# Small subset (fast)
x_train = x_train[:2000]
y_train = y_train[:2000]

# Model
inp = tf.keras.Input(shape=(100,))
x = layers.Embedding(5000, 32)(inp)

enc = layers.LSTM(32, return_sequences=True)(x)
dec = layers.LSTM(32, return_sequences=True)(enc)

attn = layers.Attention()([dec, enc])

out = layers.Dense(1, activation='sigmoid')(attn[:,-1,:])

model = models.Model(inp, out)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, verbose=0)

print("Done")