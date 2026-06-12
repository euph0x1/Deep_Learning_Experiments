reference = ['this','is','good']
candidate = ['this','is','nice']

# Count matching words
match = 0
for word in candidate:
    if word in reference:
        match += 1

# Precision
precision = match / len(candidate)

print("BLEU Score:", precision)