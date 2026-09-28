###############################################################
#
#   Contains all definitions for the drifting network
#   incl. the loss function.
#   The implementation is based on the paper: Generative Modeling via Drifting by Kaiming He et al., https://arxiv.org/pdf/2602.04770
#
###############################################################

# Imports
from typing import Tuple

import torch

from torch import nn


# Model definition
class DriftModel(nn.Module):
    """Drifting model class
    """

    def __init__(self, in_features: int=2, hidden_features: int = 128, out_features: int=2) -> None:
        """Defines the Drifting model.
        Takes a torch.Tensor([x, y]) as input and outputs a torch.Tensor([x, y]) of the same dimensions.
        Defines a mapping f: R^2 -> R^2.

        Args:
            in_features (int, optional): Number of input features, i.e. two spatial dimensions x and y. Defaults to 2.
            hidden_features (int, optional): Number of hidden features. Defaults to 128.
            out_features (int, optional): Number of output features. Here, the same spatial dimensions as the input. Defaults to 2.
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

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Args:
            x_in (torch.Tensor): Input tensor to the model of N x M dimensions, i.e. N batch dimensions + M spatial dimensions.

        Returns:
            torch.Tensor: Output of the model. Same N x M batch + spatial dimensions.
        """
        out_tensor = self.model(x_in)
        return out_tensor


# Define the loss function
class MSELoss():
    """Custom (MSE) loss function for training a "Drifting" model.
    """

    def __init__(self) -> None:
        """Initializes a standard MSE Loss object.
        """
        pass

    def __call__(self, x: torch.Tensor, x_drifted: torch.Tensor) -> torch.Tensor:
        """Computes the Mean Squared Error between network prediction x and its drifted version x_drifted.
        eps ~ p_eps := N(0, 1).
        x = f(eps) ~ q_theta.
        x_drifted = x + V, with V == drifting field.

        Args:
            x (torch.Tensor): Network prediction @ iteration i.
            x_drifted (torch.Tensor): Drifted version of network prediction @ iteration i.

        Returns:
            torch.Tensor: Loss value between target and prediction.
        """
        return torch.mean((x - x_drifted) ** 2)


# Define the drifting schema, i.e. how to compute the drifting field V and how it acts on data samples
class DrifterObject():
    """Custom minimal model to compute the drifting algorithm.
    """

    def __init__(self, model: DriftModel) -> None:
        """Instantiates a DriftObject that computes the network prediction and the drifted samples
        via a drifting field V.

        Args:
            model (DriftModel): Drifting model to compute the predictions necessary for the drifting field V.
        """
        self.model = model

    def _cdist(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Computes the pairwise distance between two sets of points x and y.
        Implementation from https://github.com/lambertae/drifting/blob/main/drift_loss.py
        See identity:||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩.

        Args:
            x (torch.Tensor): First set of points, shape (N, D).
            y (torch.Tensor): Second set of points, shape (M, D).
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.
        
        Returns:
            torch.Tensor: Pairwise distance matrix of shape (N, M).
        """
        xydot   = torch.einsum("nd,md->nm", x, y)
        xnorm   = torch.einsum("nd,nd->n", x, x)
        ynorm   = torch.einsum("md,md->m", y, y)
        sqdist  = xnorm[:, None] + ynorm[None, :] - 2 * xydot
        return torch.sqrt(torch.clip(sqdist, min=eps))

    def _compute_xhat_and_V(self, x_0: torch.Tensor, x_1: torch.Tensor, tau: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the network prediction x_hat and the drifting field V.
        Implementation as described in paper https://arxiv.org/pdf/2602.04770, Algorithm 2.

        Args:
            x_0 (torch.Tensor): Data samples from the source distribution, e.g. a unit Gaussian.
            x_1 (torch.Tensor): Data samples from the target distribution.
            tau (float, optional): Temperature parameter. Defaults to 0.05.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Network prediction and drifting field V.
        """

        # Compute network prediction and define "negative samples" NOTE: x_1 are considered "positive samples"
        x_hat = self.model(x_0)
        x_neg = x_hat           # NOTE: Reuse network prediction as negative samples for computation of V.

        # Compute pairwise distances
        dist_pos = self._cdist(x_hat, x_1)   # Distance between network prediction and target samples, shape: (N, M)
        dist_neg = self._cdist(x_hat, x_neg) # Distance between network prediction and negative samples (here, the network prediction itself), shape: (N, N)

        # Ignore the diagonal elements of dist_neg to avoid self-comparison
        dist_neg += torch.eye(dist_neg.shape[0], device=dist_neg.device) * 1e6

        # Compute logits
        logit_pos = -dist_pos / tau
        logit_neg = -dist_neg / tau

        # Concat for normalization
        logits = torch.cat([logit_pos, logit_neg], dim=1)   # Shape: (N, M + N)

        # Normalize along both dimensions
        A_row   = torch.softmax(logits, dim=1)  # Row-wise normalization
        A_col   = torch.softmax(logits, dim=0)  # Column-wise normalization
        A       = torch.sqrt(A_row * A_col)     # Element-wise multiplication and square root

        # Back to original shapes
        A_pos, A_neg = torch.split(A, [dist_pos.shape[1], dist_neg.shape[1],], dim=1)

        # Compute the weights
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        # Compute the drifting field V
        drift_pos = W_pos @ x_1
        drift_neg = W_neg @ x_neg
        
        V = drift_pos - drift_neg
        return x_hat, V

    def compute_network_prediction_and_drifted_samples(self, x_0: torch.Tensor, x_1: torch.Tensor, tau: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the drifted samples given the network prediction and the drifting field V.

        Args:
            x_0 (torch.Tensor): Data samples from the source distribution, e.g. a unit Gaussian.
            x_1 (torch.Tensor): Data samples from the target distribution.
            tau (float, optional): Temperature parameter. Defaults to 0.05.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Network prediction and drifted samples.
        """
        x_hat, V = self._compute_xhat_and_V(x_0=x_0, x_1=x_1, tau=tau)
        x_drifted = (x_hat + V).detach()   # NOTE: Detach the drifted samples from the computation graph, as they must not be used for backpropagation.
        return x_hat, x_drifted

# TODO: Write the inference part. Should be easy since its one-step generation x=f(eps) ~ p_target