import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(x_train, _), (x_test, _) = tf.keras.datasets.imdb.load_data(num_words=5000)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

# Use same data as output (Seq2Seq-style)
y_train = x_train
y_test = x_test

# Reduce size for speed
x_train = x_train[:8000]
y_train = y_train[:8000]

# -------- Baseline --------
print("\nBaseline Model")

model = models.Sequential([
    layers.Embedding(5000, 32, input_length=100),
    layers.LSTM(64, return_sequences=True),
    layers.Dense(5000, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, verbose=0)

loss1, acc1 = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Accuracy:", acc1)


# -------- Attention --------
print("\nAttention Model")

inputs = layers.Input(shape=(100,))
emb = layers.Embedding(5000, 32)(inputs)

lstm_out = layers.LSTM(64, return_sequences=True)(emb)

attention = layers.Attention()([lstm_out, lstm_out])

outputs = layers.Dense(5000, activation='softmax')(attention)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, verbose=0)

loss2, acc2 = model.evaluate(x_test, y_test, verbose=0)
print("Attention Accuracy:", acc2)

#------------------------
# -----------------------------
# Fake dataset
# -----------------------------
sequence = [1, 2, 3]

# -----------------------------
# Encoder
# -----------------------------
def encoder(seq):
    state = []
    for x in seq:
        state.append(x)   # just store values
    return state

# -----------------------------
# Decoder
# -----------------------------
def decoder(state):
    output = []
    for x in state:
        output.append(x + 1)   # simple rule
    return output

# -----------------------------
# Run
# -----------------------------
state = encoder(sequence)
result = decoder(state)

print("Input :", sequence)
print("State :", state)
print("Output:", result)