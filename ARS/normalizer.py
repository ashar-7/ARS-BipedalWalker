"""State normalizer class"""

import numpy as np

class Normalizer():

    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.variance = np.zeros(num_inputs)

    def observe(self, x):
        """Calculate the running mean and variance"""

        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x - last_mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.variance = (self.mean_diff / self.n).clip(min=0.01)

    def normalize(self, state):
        """Normalize the input"""

        obs_mean = self.mean
        obs_std = np.sqrt(self.variance)
        return (state - obs_mean) / obs_std