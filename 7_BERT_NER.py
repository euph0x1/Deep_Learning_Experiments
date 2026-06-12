# Simple rule-based NER (fake BERT behavior)

text = "Elon Musk founded Tesla in California"

# Tokenize
words = text.split()

# Fake knowledge (like pretrained understanding)
persons = ["Elon", "Musk"]
orgs = ["Tesla", "Google", "Microsoft"]
locations = ["California", "India", "USA"]

# Predict labels
result = []

for word in words:
    if word in persons:
        label = "PERSON"
    elif word in orgs:
        label = "ORG"
    elif word in locations:
        label = "LOC"
    else:
        label = "O"   # Outside (no entity)
    
    result.append((word, label))

# Output
for r in result:
    print(r)

from transformers import pipeline

# Load pretrained NER model
ner = pipeline("ner")

# Test sentence
text = "Elon Musk founded Tesla in California"

# Predict
result = ner(text)

print(result)