import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Dummy attention weights (encoder_len × decoder_len)
attention = np.random.rand(5,5)

plt.imshow(attention, cmap='viridis')
plt.colorbar()
plt.xlabel("Decoder Steps")
plt.ylabel("Encoder Steps")
plt.title("Attention Weights")
plt.show()