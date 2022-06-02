# Backpropagation for one neuron

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2)

z = xw0 + xw1 + xw2 + b
print(z)

y = max(z, 0)
print("y: ", y)

# d_value is the derivatives of the neurons later in the network
d_value = 1.0
relu_dz = d_value * (1. if z > 0 else 0)
print(relu_dz)

# values passed into relu
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

drelu_dxw0 = relu_dz * dsum_dxw0
drelu_dxw1 = relu_dz * dsum_dxw1
drelu_dxw2 = relu_dz * dsum_dxw2
drelu_db = relu_dz * dsum_db

print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0,drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

drelu_dx0 = d_value * (1. if z > 0 else 0) * w[0]

dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradient on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradient on weights
db = drelu_db

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b)

# second forward propagation
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b

y = max(z, 0)
print("y2: ", y)

# Note all these calculations were done to decrease the neuron's output
# In reality there would be one more calculation for the loss function
# and minimization of the output of the loss function would happen instead

