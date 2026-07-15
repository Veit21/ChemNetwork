###############################################################
#
#   Contains classes and function that generate the toy data
#
###############################################################

# Imports
import torch

from sklearn.datasets import make_moons
from scipy.stats import multivariate_normal


class PointCloudGenerator():
    """Generating class that provides functions for drawing data points from a start and end distribution.
    """

    # TODO: Docstrings and unit tests!
    def __init__(self, num_samples: int=50, random_state: int=42):
        self.num_samples    = num_samples
        self.random_state   = num_samples
    
    def draw_x0(self):
        X = multivariate_normal.rvs(mean=[0., 0.], cov=[1., 1.], size=self.num_samples, random_state=self.random_state)
        return torch.from_numpy(X)

    def draw_x1(self):
        X, _ = make_moons(n_samples=self.num_samples, noise=0.2, random_state=self.random_state)
        return torch.from_numpy(X)