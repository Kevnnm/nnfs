"""
Categorical cross entropy
    l_i = -Sum(target_ij * log(predicted_ij))
"""
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]  # one-hot encoded

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

# numpy version
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_target = [0, 1, 1]

# A zero could break this whole thing, address it by clipping values
print(-np.log(softmax_output[
          range(len(softmax_output)),
          class_target
]))

