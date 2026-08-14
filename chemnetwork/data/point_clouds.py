###############################################################
#
#   Contains classes and function that generate the toy data
#
###############################################################

# Imports
import torch

from enum import Enum
from sklearn.datasets import make_moons
from scipy.stats import multivariate_normal


class TargetDistribution(str, Enum):
    MOONS = "moons"
    CHECKERBOARD = "checkerboard"


class PointCloudGenerator():
    """Generating class that provides functions for drawing data points from a start and end distribution.
    """

    def __init__(self, num_samples: int=50, target_name: str="moons", target_noise: float=0.2):
        """Instatiates a generator that allows drawing data from initial and target data distributions p_0 and p_1.

        Args:
            num_samples (int, optional): Number of samples to draw, i.e. the batch size. Defaults to 50.
            target_name (str, optional): Name of the target distribution to draw from. Defaults to "moons".
            target_noise (float, optional): Noise parameter for the "moons" distribution. Defaults to 0.2.
        """
        self.num_samples    = num_samples
        self.target_name    = target_name
        self.target_noise   = target_noise

    def _draw_moons(self) -> torch.Tensor:
        """Draws samples from the "two moons" distribution from the sklearn library.

        Returns:
            torch.Tensor: A set of data points drawn from the distribution.  Shape (num_samples, 2).
        """
        X, _ = make_moons(n_samples=self.num_samples, noise=self.target_noise)
        return torch.from_numpy(X).to(torch.float32)
    
    def _draw_checkerboard(self) -> torch.Tensor:
        """Draws samples from a 4x4 "checkerboard" distribution.
        Source: Stochastic Interpolants, M Albergo et al.

        Returns:
            torch.Tensor: A set of data points drawn from the distribution. Shape (num_samples, 2).
        """
        col = torch.randint(4, (self.num_samples,))
        band = torch.randint(2, (self.num_samples,))
        row = 2 * band + (col % 2)
        x1 = col + torch.rand(self.num_samples) - 2
        x2 = row + torch.rand(self.num_samples) - 2
        return torch.stack([x1, x2], 1) * 2
    
    def draw_source(self) -> torch.Tensor:
        """Draws samples from the source data distribution p_0.
        The data distribution is a 2D normal distribution with zero mean and unit covariance.

        Returns:
            torch.Tensor: A set of data points drawn from the initial gaussian distribution p_0.  Shape (num_samples, 2).
        """
        X = multivariate_normal.rvs(mean=[0., 0.], cov=[1., 1.], size=self.num_samples)
        if len(X.shape) == 1:
            X = X[None]             # Probably sketchy workaround for getting the number of dimensions right if size=1
        return torch.from_numpy(X).to(torch.float32)
    
    def draw_target(self) -> torch.Tensor:
        dispatch = {
            TargetDistribution.MOONS: self._draw_moons,
            TargetDistribution.CHECKERBOARD: self._draw_checkerboard,
        }
        target = TargetDistribution(self.target_name.lower())   # Validate the target here directly, otherwise: ValueError
        return dispatch[target]()