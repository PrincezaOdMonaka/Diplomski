import math
import random
import matplotlib.pyplot as plt

def f(x, y) :
    return -20*math.exp(-0.2 * math.sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos((2*math.pi*y)))) + math.e + 20

cooling_rate = 0.96
T = 3

num_iterations = 10000

glob_max = 0.0


def neighbour(x, y, t) :
    sigma = 0.2
    dx = random.random() * 2 * sigma - sigma
    dy = random.random() * 2 * sigma - sigma

    return x + dx, y + dy

def simmulated_annealnig(temperature, x_current, y_current):
    x = []
    y = []
    t = temperature
    cnt = 0
    for i in range (num_iterations):
        t = t * cooling_rate
        x_next, y_next = neighbour(x_current, y_current, t)
        delta = f(x_next, y_next) - f(x_current, y_current)
        if delta < 0 or math.exp(-delta/t) > random.random() :
            x_current = x_next
            y_current = y_next
            cnt += 1
            print(x_current, y_current, f(x_current, y_current))
            x.append(cnt)
            y.append(f(x_current, y_current))
    plt.plot(x, y)
    plt.show()

    print(cnt)
    return x_current, y_current, f(x_current, y_current)


x_min = random.random()
y_min = random.random()
e_min = f(x_min, y_min)
temp = 0.0

for i in range(10) :
    x = random.random()
    y = random.random()
    e = f(x, y)
    temp += e
    if e < e_min :
        e_min = e
        x_min = x
        y_min = y

print(simmulated_annealnig(T, x_min, y_min))

# print(min(T))
