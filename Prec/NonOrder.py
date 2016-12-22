import numpy as np


def softmax(x, tau=1):
    x_max = np.max(x)
    return np.exp((x - x_max) * tau) / np.exp((x - x_max) * tau).sum()


np.random.seed(17)
A = np.random.random(24)
M = np.reshape(A, (4, 6))

z1 = [np.max(softmax(x, tau=1)) for x in M]
z8 = [np.max(softmax(x, tau=8)) for x in M]

print(np.argsort(z1), np.argsort(z8))
