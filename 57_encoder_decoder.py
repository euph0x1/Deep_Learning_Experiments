# -----------------------------
# Step 1: Mapping
# -----------------------------
word_to_num = {
    "cat": 1,
    "dog": 2,
    "bat": 3
}

num_to_word = {}

# reverse mapping using loop
for key in word_to_num:
    value = word_to_num[key]
    num_to_word[value] = key

# -----------------------------
# Step 2: Input
# -----------------------------
word = "cat"

# -----------------------------
# Step 3: Encoder
# -----------------------------
def encoder(w):
    return word_to_num[w]

# -----------------------------
# Step 4: Decoder
# -----------------------------
def decoder(num):
    return num_to_word[num]

# -----------------------------
# Step 5: Run
# -----------------------------
state = encoder(word)
output = decoder(state)

print("Input :", word)
print("State :", state)
print("Output:", output)