import math
import numpy as np
import matplotlib.pyplot as plt
import random

x = np.arange(-1, 1, 0.001)
random.shuffle(x)

x_train = x[:1600]
y_train = x_train**3

x_test = x[1600:]
y_test = x_test**3

num_of_layers = 4
layers = [1, 4, 4, 1]

T = 2
cooling_rate = 0.96
num_of_iterations = 300

def function_activation(x) :
    matrix = np.copy(x)
    n = matrix.shape[0]

    for i in range(n) :
        matrix[i, 0] = 2./ (1. + math.exp(-2 * matrix[i, 0])) - 1
        # matrix[i, 0] = 1. / (1. + math.exp(-matrix[i, 0]))
    return matrix

def E(weights, layers, y, x) :
    data_size = len(y)
    dim1 = len(layers) - 1
    loss = 0.0
    for index in range (data_size) :
        current_output = np.array([[x[index]]])
        for i in range (dim1) :
            dim2 = layers[i + 1]
            dim3 = layers[i]
            sub_weights = weights[i][:dim2, :dim3]
            weighted_output = np.matmul(sub_weights, current_output)
            current_output = function_activation(weighted_output)
        loss += (y[index] - current_output[0, 0])**2
    return loss / data_size

def neighbour(weights, layers, temperature) :
    t = temperature
    if t > 0.1:
        sigma = 0.2
    else:
        if t > 0.05:
            sigma = 0.1
        else:
            if t > 0.02:
                sigma = 0.05
            else:
                if t > 0.01:
                    sigma = 0.02
                else:
                    sigma = 0.01

    dim1 = weights.shape[0]
    m = weights.shape[1]
    l = weights.shape[2]
    weights_new = np.zeros((dim1, m, l))
    for i in range (dim1) :
        dim2 = layers[i + 1]
        dim3 = layers[i]
        for j in range(dim2) :
            for k in range(dim3) :
                mu = weights[i, j, k]
                weights_new[i, j, k] = np.random.normal(mu, sigma, 1)

    return weights_new

def simulated_annealing(temperature, num_of_iterations, y, x, current_state, layers) :
    t = temperature
    cnt = 0
    E_current = E(current_state, layers, y, x)
    E_best = E_current
    best_state = current_state

    for i in range (num_of_iterations) :
        t = t * cooling_rate
        next_state = neighbour(current_state, layers, t)
        E_next = E(next_state, layers, y, x)

        if E_next < E_current :
            current_state = next_state
            E_current = E_next
        else :
            if  math.exp(-(E_next - E_current)/t) < random.random() :
                current_state = next_state
                E_current = E_next
                cnt += 1
                # print('-----------------------------------------')
                # print(current_state)
                # print(delta, E(current_state,layers, y, x))
        if E_best > E_next :
            best_state = next_state
            E_best = E_next
    print(cnt)
    print(best_state, E_best)
    return best_state

def check_accurecy(weights, layers, y, x) :
    data_size = len(y)
    dim1 = len(layers) - 1
    loss = 0.0
    y_ = []
    for index in range (data_size) :
        current_output = np.array([[x[index]]])
        for i in range (dim1) :
            dim2 = layers[i + 1]
            dim3 = layers[i]
            sub_weights = weights[i][:dim2, :dim3]
            weighted_output = np.matmul(sub_weights, current_output)
            current_output = function_activation(weighted_output)
        y_.append(current_output[0, 0])
        loss += (y[index] - current_output[0, 0])**2
    print('Greska ', loss/data_size)
    plt.scatter(x, y, c='b')
    plt.scatter(x, y_, c='r')
    plt.show()


# dim1 = num_of_layers - 1
# weights = np.zeros((dim1,max(layers),max(layers)))
# cnt = 1
# for i in range(dim1):
#     dim2 = layers[i + 1]
#     dim3 = layers[i]
#     for j in range(dim2):
#         for k in range(dim3):
#             weights[i,j,k] = cnt/10.
#             cnt += 1

dim1 = num_of_layers - 1
weights = np.zeros((dim1,max(layers),max(layers)))
temp = 0.0
for index in range(10) :
    for i in range(dim1):
        dim2 = layers[i + 1]
        dim3 = layers[i]
        for j in range(dim2):
            for k in range(dim3):
                weights[i,j,k] = random.random()
    if index == 0 :
        E_best = E(weights, layers, y_train, x_train)
        weights_best = weights
    else :
        E_current = E(weights, layers, y_train, x_train)
        if E_current < E_best:
            E_best = E_current
            weights_best = weights
    temp += E(weights, layers, y_train, x_train)
T = temp / 10
print(T)
weights = simulated_annealing(T, num_of_iterations, y_train, x_train, weights_best, layers)
check_accurecy(weights, layers, y_train, x_train)

