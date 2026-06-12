import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread("image.png",0)

# Blur kernel (3x3 averaging)
kernel = np.ones((3,3)) / 9


blur = cv2.filter2D(img, -1, kernel)
#blur = cv2.blur(img, (3,3))
# Show
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Blurred")
plt.imshow(blur, cmap='gray')
plt.axis('off')

plt.show()

# Sharpening kernel
kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

sharp = cv2.filter2D(img, -1, kernel)

# Show
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Sharpened")
plt.imshow(sharp, cmap='gray')
plt.axis('off')

plt.show()