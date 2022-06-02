import numpy as np


dvalues = np.array([[1., 1., 1.],  # shape (n_sample, n_neuron) elem: neuron_gradient
                    [2., 2., 2.],
                    [3., 3., 3.]])

inputs = np.array([[1, 2, 3, 2.5],  # shape (n_sample, n_input) elem: input_value
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],  # shape before transpose (n_neuron, n_weight) elem: weight_value
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = [[2, 3, 0.5]]

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Relu derivative
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer
# derivative inputs
dinputs = np.dot(drelu, weights.T)
# derivative weights
dweights = np.dot(inputs.T, drelu)
print(drelu)
# derivative biases
dbiases = np.sum(drelu, axis=0, keepdims=True)
print(dbiases)

weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights, biases)

