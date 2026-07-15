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
    """Flow Matching model.
    """
    
    def __init__(self, in_features: int=3, hidden_features: int=128, out_features: int=2):
        """Defines the Flow Matching model.
        Takes a torch.tensor([t, x, y]) as input and outputs a torch.tensor([x, y]).

        Args:
            in_features (int, optional): Number of input features. Defaults to 3: One time dimension and two space dimensions.
            hidden_features (int, optional): Number of hidden features. Defaults to 128.
            out_features (int, optional): Number of output features. Defaults to 2: Two space dimensions.
        """
        super().__init__()

        self.in_features        = in_features
        self.hidden_features    = hidden_features
        self.out_features       = out_features

        # TODO: Batch size is implicit, right?
        self.model  = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, x_in: torch.tensor, t: torch.tensor):
        """Model forward pass.

        Args:
            x_in (torch.tensor): N x M dimensional input tensor, i.e. N batch dimensions + M spatial dimensions.
            t (torch.tensor): N x 1 dimensional time tensor.
        """
        in_tensor   = torch.cat((t, x_in), dim=1)   # (1, 1) + (1, 2) -> (1, 3)
        out_tensor  = self.model(in_tensor)
        return(out_tensor)


# Define ODE solver
# TODO: ODE solver needs to be overworked!
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
        """Defines a Mean Squared Error loss function for trining a flow matching network.

        Args:
            regression_target (str, optional): Wether to train the network with the velocity field "v" as a target, the noise vector "eps" or the clean sample "x". Defaults to "v".
        """
        self.regression_target  = regression_target
    
    def __call__(self, x_0: torch.tensor, x_1: torch.tensor, v_hat: torch.tensor) -> torch.tensor:
        """Computes the Mean Squared Error between the network prediction and the target, given the start and end tensor.

        Args:
            x_0 (torch.tensor): Data point at t=0.
            x_1 (torch.tensor): Data point at t=1.
            v_hat (torch.tensor): Neural network prediction of the velocity field.

        Raises:
            NotImplementedError: When a regression target other than the velocity "v" is chosen.

        Returns:
            torch.tensor: A floating point value that is the loss.
        """
        if self.regression_target == "v":
            u_t = self.__compute_target(x_0=x_0, x_1=x_1)
            return torch.mean((u_t - v_hat) ** 2)
        else:
            raise NotImplementedError("Other regression targets not implemented yet.")
    
    def __compute_target(self, x_0: torch.tensor, x_1: torch.tensor) -> torch.tensor:
        """Computes the target vector that the network output regresses against.

        Args:
            x_0 (torch.tensor): Start tensor: Data point at t=0.
            x_1 (torch.tensor): End tensor: Data point at t=1.

        Returns:
            torch.tensor: The target tensor as regression target.
        """
        return x_1 - x_0
