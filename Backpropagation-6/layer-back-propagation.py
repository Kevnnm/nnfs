import numpy as np

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],  # shape (n_sample, n_neuron) elem: neuron_gradient
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of weights - one for each neuron
# We have 4 inputs , thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],  # shape before transpose (n_neuron, n_weight) elem: weight_value
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# sum weights of given input
# and multiply by the passed in gradient for this neuron
# weights[0] * gradient[0] is the partial derivative

# dx0 = sum(weights[0] * dvalues[0])
# dx1 = sum(weights[1] * dvalues[0])
# dx2 = sum(weights[2] * dvalues[0])
# dx3 = sum(weights[3] * dvalues[0])

# OR

dinputs = np.dot(dvalues, weights.T)  # n_neuron_gradient . n_neuron

print('dinputs: \n', dinputs)

# calculating the gradient of a neuron with respect to inputs
inputs = np.array([[1, 2, 3, 2.5],  # shape (n_sample, n_input) elem: input_value
                   [2., 5., 01., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

print('inputs.T: \n', inputs.T)
print('dvalues: \n', dvalues)
dweights = np.dot(inputs.T, dvalues)  # n_sample . n_sample
print('dweights: \n', dweights)

# gradient of a neuron with respect to biases
# derivative of the neuron with respect to its bias is always 1
# the chain rule tells us that d(neural_network)/db = neuron_gradient * 1

# 3 biases for three neurons
biases = [[2, 3, 0.5]]

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print('dbiases: \n', dbiases)


# example layer output
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

drelu = np.zeros_like(z)
drelu[z > 0] = 1

drelu *= dvalues

# OR

drelu = dvalues.copy()
drelu[z <= 0] = 0

print('drelu: \n', drelu)
