###############################################################
#
#   Script for sampling from a trained flow matching model.
#
###############################################################

# Imports
import torch

from pathlib import Path

from chemnetwork.models.flow_matching import FlowModel, NumericalODESolver
from chemnetwork.data.point_clouds import PointCloudGenerator


def load_model(
    checkpoint_path: Path,
    in_features: int=3,
    hidden_features: int=128,
    out_features: int=2,
) -> FlowModel:
    """Loads a trained FlowModel from a checkpoint file.

    The architecture arguments must match the configuration that the
    checkpoint was trained with (see preferences/model/mlp.yaml).

    Args:
        checkpoint_path (Path): Path to the .pt checkpoint saved during training.
        in_features (int, optional): Number of input features. Defaults to 3.
        hidden_features (int, optional): Number of hidden features. Defaults to 128.
        out_features (int, optional): Number of output features. Defaults to 2.

    Returns:
        FlowModel: The model with trained weights loaded, set to evaluation mode.
    """
    model = FlowModel(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def generate_samples(
    model: FlowModel,
    num_samples: int=500,
    integration_steps: int=100
) -> torch.tensor:
    """Generates samples from the learned target distribution p_1.

    Draws source points from p_0 and integrates them along the model's
    learned vector field using an ODE solver.

    Args:
        model (FlowModel): A trained flow matching model in evaluation mode.
        num_samples (int, optional): Number of samples to generate. Defaults to 500.
        integration_steps (int, optional): Number of solver integration steps. Defaults to 100.

    Returns:
        torch.tensor: Generated points of shape (num_samples, 2).
    """
    data_generator = PointCloudGenerator(num_samples=num_samples)
    solver = NumericalODESolver(model=model, solver="euler", integration_steps=integration_steps, return_trajectory=False)

    x0 = data_generator.draw_x0()

    with torch.no_grad():
        x1_hat = solver(x0)

    return x1_hat


if __name__ == "__main__":

    # Small test: load a checkpoint and generate a batch of samples.
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / Path("checkpoints/MultiLayerPerceptron_weigths_step_50000.pt")

    model = load_model(checkpoint_path=checkpoint_path)
    samples = generate_samples(model=model, num_samples=500)

    print(f"Generated data samples of shape {tuple(samples.shape)}.")