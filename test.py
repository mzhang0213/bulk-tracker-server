import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
img = cv2.imread('upload.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract a horizontal line of pixels (e.g., row 100)
line = gray[100, :]  # 1D intensity profile

# First derivative
grad = np.gradient(line.astype(np.float32))

# Second derivative
grad2 = np.gradient(grad)

# Plot to visualize
plt.plot(line, label='Intensity')
plt.plot(grad, label='1st Derivative')
plt.plot(grad2, label='2nd Derivative')
plt.legend()
plt.show()
