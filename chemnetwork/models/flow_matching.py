###############################################################
#
#   Contains all definitions for the flow-matching network
#   incl. the loss function.
#
###############################################################

# Imports
import torch

import numpy as np

from torch import nn


# Define the model
class FlowModel(nn.Module):
    """Flow Matching model."""
    
    def __init__(self, in_features: int=3, hidden_features: int=128, out_features: int=2):
        """Defines the Flow Matching model.
        Takes a torch.tensor([t, x, y]) as input and outputs a torch.tensor([x, y]).

        Args:
            in_features (int, optional): Number of input features. Defaults to 3: One time dimension and two space dimensions.
            hidden_features (int, optional): Number of hidden features. Defaults to 128.
            out_features (int, optional): Number of output features. Defaults to 2: Two space dimensions.
        """
        self.in_features        = in_features
        self.hidden_features    = hidden_features
        self.out_features       = out_features

        # TODO: What about the batch size?
        self.model              = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, x_in: torch.tensor, t: torch.tensor):

        in_tensor   = torch.cat((t, x_in), dim=1)   # (1, 1) + (1, 2) -> (1, 3)
        # TODO: To implement.
        raise NotImplementedError("The forward function is not implemented yet!")


# Define ODE solver
class NumericalODESolver():
    """Custom numerical ODE solver.
    Implements a simple Euler solver and Runge-Kutta type solvers, such as midpoint (Heun) and RK4 solver.
    """

    def __init__(self, model: nn.Module, solver: str="euler", return_trajectory: bool=True):
        self.model              = model
        self.solver             = solver
        self.return_trajectory  = return_trajectory

    def euler_solver(self, x_in: torch.tensor, integration_steps: int=50) -> torch.tensor:
        """Euler solver for integrating a point along a vector field defined by a neural network.

        Args:
            x_in (torch.tensor): 2D input tensor.
            integration_steps (int, optional): Number of integration steps. Defaults to 50.

        Returns:
            torch.tensor: Resulting final state of the input point after integration. Optionally the complete trajectory over time.
        """

        x_current       = x_in
        x_trajectory    = list()    # Save all intermediate integration steps to recreate the trajectory
        t_list          = torch.linspace(start=0., end=1., steps=integration_steps)[1:]  # Do not start at t=0
        d_t             = t_list[1] - t_list[0] # Delta t, necessary to compute each integration step

        # Iterate over all time points
        for t in t_list:
            v_current   = self.model(x_current, t)
            x_next      = x_current + v_current * d_t
            x_current   = x_next
            x_trajectory.append(x_current)
        x_trajectory = torch.stack(x_trajectory)

        if self.return_trajectory:
            return x_trajectory
        else:
            return x_trajectory[-1]

    def midpoint_solver(self):
        raise NotImplementedError("Midpoint solver not implemented yet.")
    
    def rk4_solver(self):
        raise NotImplementedError("Runge-Kutta 4 solver not implemented yet.")

    def forward(self, in_tensor: torch.tensor, integration_steps: int=50):
        if self.solver == "euler":
            return self.euler_solver(x_in=in_tensor, integration_steps=integration_steps)
        else:
            raise NotImplementedError("Other solvers not implemented yet.")


# Define Loss function
class MSELoss():
    """Custom MSE (regression) loss function as proposed in the original Flow Matching paper from Lipman et al.
    """

    def __init__(self, regression_target: str="v"):
        self.regression_target  = regression_target
    
    def __call__(self, u_t: torch.tensor, v_hat: torch.tensor):
        if self.regression_target == "v":
            return torch.mean((x_gt - x_hat) ** 2)
        else:
            raise NotImplementedError("Other regression targets not implemented yet.")