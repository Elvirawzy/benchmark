from collections import deque
import numpy as np


def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats0 = deque(maxlen=n)
        self.stats1 = deque(maxlen=n)
        self.stats2 = deque(maxlen=n)

    def append(self, x):
        self.stats0.append(x[0])
        self.stats1.append(x[1])
        self.stats2.append((x[0]+x[1])/2)

    def get(self, idx):
        if idx == 2:
            return np.mean(self.stats2)
        if idx == 0:
            return np.mean(self.stats0)
        return np.mean(self.stats1)
