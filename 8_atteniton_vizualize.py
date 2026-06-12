import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Load model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# Input sentence
text = "the cat sat on the mat"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Get outputs (including attention)
outputs = model(**inputs)
attentions = outputs.attentions   # list of layers

# Take attention from last layer
attn = attentions[-1][0]   # (heads, tokens, tokens)

# Take first head
attn = attn[0].detach().numpy()

# Plot
plt.imshow(attn)
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title("Attention Weights (Last Layer, Head 1)")
plt.colorbar()
plt.show()