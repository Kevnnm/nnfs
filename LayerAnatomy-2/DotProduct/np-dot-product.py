import numpy as np

inputs = [1, 2, 3, 2.5]

# Three weight sets
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# Three biases
biases = [2, 3, 0.5]

# Compute dot product
# This is the same calculation performed in layer-anatomy.py
output = np.dot(weights, inputs) + biases
print(output)
