"""
    # Basic neuron implementation
    * A neuron in a neural network always has inputs and weights associated with those inputs.
    * The weights are multiplied to the corresponding inputs and their products are summed together.
    * The bias is then added to the sum and that is the final output
"""
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print(output)
