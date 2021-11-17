import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Three weight sets for three neurons
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# Three biases for three neurons
biases = [2, 3, 0.5]

# Second set of neurons
weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]
# Three biases for three neurons
biases2 = [-1, 2, -0.5]

# Matrix multiplication to perform dot product in batches
layer1_outputs = np.matmul(inputs, np.array(weights).T) + biases
output = np.matmul(layer1_outputs, np.array(weights2).T) + biases2

print(output)
