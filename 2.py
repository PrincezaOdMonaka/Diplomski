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

# plt.scatter(x_test, y_test)
# plt.show()

T = 2
num_of_iterations = 300
cooling_rate = 0.96

def E(theta, y, x, temperature):
    t = temperature
    data_size = len(y)
    n = len(theta)
    y_ = []
    e = 0.0

    for i in range(data_size) :
        y_current = 0.0
        for j in range (n) :
            y_current += theta[j]*(x[i]**(n-j-1))
        y_.append(y_current)
        e += (y[i] - y_current)**2/2
    return e

def neighbour(theta, temperature) :
    t = temperature

    if t > 1:
        sigma = 0.2
    else:
        if t > 0.5:
            sigma = 0.1
        else:
            if t > 0.2:
                sigma = 0.05
            else:
                if t > 0.1:
                    sigma = 0.02
                else:
                    sigma = 0.01
    n = len(theta)
    theta_new = []
    for i in range (n) :
        mu = theta[i]
        theta_new.append(np.random.normal(mu, sigma, 1))

    return theta_new

def simulated_annealing(temperature, num_of_iterations, y, x, current_state) :
    t = temperature
    cnt = 0
    _x = []
    _y = []
    n = 0
    E_current = E(current_state, y, x, t)
    best_state = current_state
    E_best = E_current
    for i in range (num_of_iterations) :
        t = t * cooling_rate
        next_state = neighbour(current_state, t)
        E_next = E(next_state, y, x, t)
        delta = E_next - E_current

        if delta < 0 or math.exp(-delta/t) > random.random() :
            if delta > 0 :
                cnt += 1
            n += 1
            current_state = next_state
            E_current = E_next
            _x.append(n)
            _y.append(E(current_state, y, x, t))
            print(current_state, E_current)
        if E_next < E_best :
            E_best = E_next
            best_state = next_state
    print (cnt)
    # plt.plot(_x, _y)
    # plt.show()
    return best_state

def check_accurecy(theta, y, x, temperature) :
    t = temperature
    data_size = len(y)
    n = len(theta)
    y_ = []
    e = 0.0

    for i in range(data_size):
        y_current = 0.0
        for j in range(n):
            y_current += theta[j] * (x[i] ** (n - j - 1))
        y_.append(y_current)
        e += (y[i] - y_current) ** 2 / 2
    print('Greska ', e)
    plt.scatter(x, y_, c = 'r')
    plt.scatter(x, y, c = 'b')
    plt.show()

n = 5
current_state = []

for i in range(n):
    current_state.append(random.random())

E_best = E(current_state, y_train, x_train, T)

for j in range(10) :
    state = []
    for i in range (n) :
        state.append(random.random())
    E_current = E(state, y_train, x_train, T)
    if(E_current < E_best) :
        E_best = E_current
        current_state = state

theta = simulated_annealing(T, num_of_iterations, y_train, x_train, current_state)
check_accurecy(theta, y_test, x_test, T)
