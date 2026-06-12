import numpy as np

img = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])

kernel = np.array([[1,0],
                   [0,-1]])

out = np.zeros((2,2))

for i in range(2):
    for j in range(2):
        out[i,j] = np.sum(img[i:i+2, j:j+2] * kernel)

print(out)