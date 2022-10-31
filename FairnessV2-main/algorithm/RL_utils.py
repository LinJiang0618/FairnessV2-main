import numpy as np
import pandas as pd
import random
import collections

class ReplayBuffer:
    def __init__(self, capacity,batch_size):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, reward, next_state, policy):
        self.buffer.append((state, reward, next_state, policy))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, reward, next_state, policy = zip(*transitions)
        return np.array(state), reward, np.array(next_state), policy

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
