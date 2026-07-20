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

    def __init__(self, num_samples: int=50):
        """Instatiates a generator that allows drawing data from initial and target data distributions p_0 and p_1.

        Args:
            num_samples (int, optional): Number of samples to draw, i.e. the batch size. Defaults to 50.
        """
        self.num_samples    = num_samples
    
    def draw_x0(self) -> torch.tensor:
        """Draws samples from the initial data distribution p_0.
        The data distribution is a 2D normal distribution with zero mean and unit covariance.

        Returns:
            torch.tensor: A set of data points drawn from the initial distribution p_0.
        """
        X = multivariate_normal.rvs(mean=[0., 0.], cov=[1., 1.], size=self.num_samples)
        if len(X.shape) == 1:
            X = X[None]             # Probably sketchy workaround for getting the number of dimensions right if size=1
        return torch.from_numpy(X).to(torch.float32)

    def draw_x1(self, noise: float=0.2) -> torch.tensor:
        """Draws samples from the final/traget data distribution p_1.
        The data distribution is a 2D complex distribution "two moons" from sklearn.

        Returns:
            torch.tensor: A set of data points drawn from the target distribution p_1.
        """
        X, _ = make_moons(n_samples=self.num_samples, noise=noise)
        return torch.from_numpy(X).to(torch.float32)