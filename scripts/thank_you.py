import numpy as np
import matplotlib.pyplot as plt
# Create a 7x50 array of zeros
arr = np.zeros((7, 50), dtype=int)

# Define THANK YOU
# T
arr[0, 0:5] = 1
arr[1:7, 2] = 1

# H
arr[0:7, 7] = 1
arr[3, 7:11] = 1
arr[0:7, 10] = 1

# A
arr[1:7, 13] = 1
arr[1:7, 17] = 1
arr[0, 14:17] = 1
arr[3, 14:17] = 1

# N
arr[0:7, 20] = 1
arr[1, 21] = 1
arr[2, 22] = 1
arr[3, 23] = 1
arr[4, 24] = 1
arr[0:7, 25] = 1

# K
arr[0:7, 28] = 1
arr[3, 29] = 1
arr[2, 30] = arr[4, 30] = 1
arr[1, 31] = arr[5, 31] = 1
arr[0, 32] = arr[6, 32] = 1

# Y
arr[0:3, 35] = 1
arr[0:3, 39] = 1
arr[2:7, 37] = 1

# O
arr[0:7, 42] = 1
arr[0, 42:46] = 1
arr[6, 42:46] = 1
arr[0:7, 45] = 1

# U
arr[0:6, 48] = 1  # Left vertical line
arr[6, 48:50] = 1  # Bottom horizontal line
arr[0:6, 49] = 1  # Right vertical line

plt.imshow(arr, cmap="gray")
plt.axis("off")
plt.show()