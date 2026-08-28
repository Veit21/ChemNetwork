###############################################################
#
#   Contains all definitions for the flow-matching network
#   incl. the loss function.
#
###############################################################

# Imports
import torch

from torch import nn


# Define the model
class FlowModel(nn.Module):
    """Flow Matching model.
    """
    
    def __init__(self, in_features: int=3, hidden_features: int=128, out_features: int=2):
        """Defines the Flow Matching model.
        Takes a torch.Tensor([t, x, y]) as input and outputs a torch.Tensor([x, y]).

        Args:
            in_features (int, optional): Number of input features. Defaults to 3: One time dimension and two space dimensions.
            hidden_features (int, optional): Number of hidden features. Defaults to 128.
            out_features (int, optional): Number of output features. Defaults to 2: Two space dimensions.
        """
        super().__init__()

        self.in_features        = in_features
        self.hidden_features    = hidden_features
        self.out_features       = out_features

        self.model  = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, x_in: torch.Tensor, t: torch.Tensor):
        """Model forward pass.

        Args:
            x_in (torch.Tensor): N x M dimensional input tensor, i.e. N batch dimensions + M spatial dimensions.
            t (torch.Tensor): N x 1 dimensional time tensor.
        """
        in_tensor   = torch.cat((t, x_in), dim=1)   # (1, 1) + (1, 2) -> (1, 3)
        out_tensor  = self.model(in_tensor)
        return(out_tensor)


# Define ODE solver
class NumericalODESolver():
    """Custom numerical ODE solver.
    Implements a simple Euler solver and Runge-Kutta type solvers, such as midpoint (Heun) and RK4 solver.
    """

    def __init__(self, model: nn.Module, solver: str="euler", integration_steps: int=100, return_trajectory: bool=False):
        """Instantiates a custom numerical ODE solver.

        Args:
            model (nn.Module): A neural network that defines the vector field to integrate along.
            solver (str, optional): Which exact solver to use. Defaults to "euler".
            integration_steps (int, optional): Number of solver steps. Defaults to 100.
            return_trajectory (bool, optional): Whether to return the full trajectory of the data or not. Defaults to False.
        """
        self.model              = model
        self.solver             = solver
        self.integration_steps  = integration_steps
        self.return_trajectory  = return_trajectory

    def _euler_solver(self, x_in: torch.Tensor) -> torch.Tensor:
        """Euler solver for integrating a point along a vector field defined by a neural network.

        Args:
            x_in (torch.Tensor): 2D input tensor.

        Returns:
            torch.Tensor: Resulting final state of the input point after integration. Optionally the complete trajectory over time.
            Shape is always (T, bs, 2), where T is the number of integration steps and bs is the batch size of the input tensor.
            Note that for return_trajectory=False, T is defined as 1 to conserve the shape of the output tensor, i.e. (1, bs, 2).
        """

        _bs, *_         = x_in.shape
        x_current       = x_in
        x_trajectory    = [x_in]    # Save all intermediate integration steps to recreate the trajectory
        t_list          = torch.linspace(start=0., end=1., steps=self.integration_steps)
        d_t             = t_list[1] - t_list[0] # Delta t, necessary to compute each integration step

        # Iterate over all time points
        for i, _ in enumerate(t_list[:-1]):
            t_vec       = t_list[i].expand(_bs, 1) # Expand the time point to match the batch size of the input data
            v_current   = self.model(x_current, t_vec)
            x_next      = x_current + v_current * d_t
            x_current   = x_next
            x_trajectory.append(x_current)
        x_trajectory = torch.stack(x_trajectory)

        if self.return_trajectory:
            return x_trajectory             # Return the full trajectory of the integration, i.e. all intermediate steps
        else:
            return x_trajectory[-1][None]   # Return only the final state of the integration, i.e. the last step. None is necessary to keep the batch dimension.

    def _midpoint_solver(self):
        raise NotImplementedError("Midpoint solver not implemented yet.")
    
    def _rk4_solver(self):
        raise NotImplementedError("Runge-Kutta 4 solver not implemented yet.")

    def __call__(self, in_tensor: torch.Tensor) -> torch.Tensor:
        if self.solver == "euler":
            return self._euler_solver(x_in=in_tensor)
        else:
            raise NotImplementedError("Other solvers not implemented yet.")


# Define Loss function
class MSELoss():
    """Custom MSE (regression) loss function as proposed in the original Flow Matching paper from Lipman et al.
    """

    def __init__(self, regression_target: str="v") -> None:
        """Defines a Mean Squared Error loss function for trining a flow matching network.

        Args:
            regression_target (str, optional): Wether to train the network with the velocity field "v" as a target, the noise vector "eps" or the clean sample "x". Defaults to "v".
        """
        self.regression_target  = regression_target
    
    def __call__(self, u_t: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
        """Computes the Mean Squared Error between the network prediction and the target, given the start and end tensor.

        Args:
            u_t (torch.Tensor): Ground truth regression target.
            v_hat (torch.Tensor): Neural network prediction of the velocity field.

        Raises:
            NotImplementedError: When a regression target other than the velocity "v" is chosen.

        Returns:
            torch.Tensor: A floating point value that is the loss.
        """
        if self.regression_target == "v":
            return torch.mean((u_t - v_hat) ** 2)
        else:
            raise NotImplementedError("Other regression targets not implemented yet.")


# Define the concrete flow matching schema, i.e. how to compute x_t and u_t
class FlowMatcher():
    """Custom minimal model for computing the interpolant and the regression target
    """

    def __init__(self) -> None:
        """Instatiates a FlowMatcher objective that computes the linear interpolation of x_t and the regression target u_t.
        """

    def _sample_t(self, x: torch.Tensor) -> torch.Tensor:
        """Generates a tensor of random time points, uniformly drawn from the interval (0., 1.).
        Keeps the batch size dictated by the data tensors, here x.

        Args:
            x (torch.Tensor): A data tensor (x_0 or x_1) to get the batch size to correctly draw time points t.

        Returns:
            torch.Tensor: A time tensor t with shape (bs_x, 1).
        """
        bs, *_      = x.shape                   # Unpacks batch size into var "bs" and remaining dims into "_"
        t_batched   = torch.rand(size=(bs, 1))
        return t_batched

    def _sample_xt(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the sample x_t via linear interpolation, i.e. x_t = t * x_1 + (1 - t) * x_0.

        Args:
            x_0 (torch.Tensor): Start tensor with shape (bs, data_dim).
            x_1 (torch.Tensor): Final tensor with shape (bs, data_dim).
            t (torch.Tensor): Time tensor with shape (bs, 1).

        Returns:
            torch.Tensor: Intermediate state computed by linear interpolation.
        """
        return t * x_1 + (1 - t) * x_0

    def _sample_ut(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Samples the conditional flow target, i.e. u_t = x_1 - x_0.

        Args:
            x_0 (torch.Tensor): Tensor of initial data points.
            x_1 (torch.Tensor): Tensor of final data points.

        Returns:
            torch.Tensor: Regression target tensor u_t.
        """
        return x_1 - x_0

    def sample_interpolant_and_target(self, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple:
        """Samples random time points t, the corresponding interpolant x_t and a regression target u_t.

        Args:
            x_0 (torch.Tensor): Initial data tensor.
            x_1 (torch.Tensor): Final data tensor.

        Returns:
            tuple: Tuple of (t, x_t, u_t), i.e. the randomly sampled time tensor of shape (bs, 1), the interpolant and the regression target, both of shape (bs, data_dim).
        """

        assert x_0.shape == x_1.shape, f"Shapes of x_0 and x_1 do not match! Got x_0: {x_0.shape}, x_1: {x_1.shape}."
        t   = self._sample_t(x=x_0)
        x_t = self._sample_xt(x_0=x_0, x_1=x_1, t=t)
        u_t = self._sample_ut(x_0=x_0, x_1=x_1)
        
        return t, x_t, u_t
