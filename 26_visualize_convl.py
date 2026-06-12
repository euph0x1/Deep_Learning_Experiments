import numpy as np
import matplotlib.pyplot as plt

# Simple image
img = np.array([
    [1,2,3,0],
    [0,1,2,3],
    [3,1,0,2],
    [2,3,1,0]
])

# Multiple kernels (filters)
kernels = [
    np.array([[1,0],[0,-1]]),      # edge-like
    np.array([[0,1],[-1,0]]),      # diagonal
    np.ones((2,2)) / 4             # blur
]

feature_maps = []

# Apply each kernel
for k in kernels:
    out = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            out[i,j] = np.sum(img[i:i+2, j:j+2] * k)
    feature_maps.append(out)

# Plot results
for i, fm in enumerate(feature_maps):
    plt.imshow(fm, cmap='gray')
    plt.title(f"Feature Map {i+1}")
    plt.colorbar()
    plt.show()