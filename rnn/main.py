import numpy as np


class RNN:
    def __init__(self):
        self.h = np.array([0, 0, 0])
        # RNN parameters
        self.W_xh = np.array([0, 0, 0])
        self.W_hh = np.array([0, 0, 0])
        self.W_hy = np.array([0, 0, 0])

    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y


h = np.array([1, 2, 3])
t = np.array([1, 2, 3])
print(np.dot(h, t))
