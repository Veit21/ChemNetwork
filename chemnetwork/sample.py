###############################################################
#
#   Script for sampling from a trained flow matching model.
#
###############################################################

# Imports
import torch

from pathlib import Path
from collections import namedtuple

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
    print(f"Model weights loaded from '{checkpoint_path}'.")

    return model


# TODO: Also serve the ground truth target (x1) as output for a visual comparison? Therefore, the parameters for x1 have to be known here.
def generate_samples(
    model: FlowModel,
    num_samples: int=500,
    integration_steps: int=100,
    return_trajectory: bool=False,
) -> namedtuple:
    """Generates samples from the learned target distribution p_1.

    Draws source points from p_0 and integrates them along the model's
    learned vector field using an ODE solver.

    Args:
        model (FlowModel): A trained flow matching model in evaluation mode.
        num_samples (int, optional): Number of samples to generate. Defaults to 500.
        integration_steps (int, optional): Number of solver integration steps. Defaults to 100.
        return_trajectory (bool, optional): Whether to return the full trajectory of the solver. Defaults to False.

    Returns:
        namedtuple: A named tuple containing the source points and the generated points.
        Output.source - Tensor of shape (num_samples, 2) containing the source points drawn from p_0.
        Output.generated - Tensor of shape (T, num_samples, 2) containing the generated points after integration.
        T is the number of integration steps if return_trajectory is True, otherwise T=1.
    """
    Output = namedtuple('Output', ['source', 'generated'])
    data_generator = PointCloudGenerator(num_samples=num_samples)
    solver = NumericalODESolver(model=model, solver="euler", integration_steps=integration_steps, return_trajectory=return_trajectory)
    source_data = data_generator.draw_x0()

    with torch.no_grad():
        predicted_data = solver(source_data)
    
    out = Output(source=source_data, generated=predicted_data)
    return out


if __name__ == "__main__":

    # Small test: load a checkpoint and generate a batch of samples.
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / Path("checkpoints/MultiLayerPerceptron_weights_step_50000.pt")

    model = load_model(checkpoint_path=checkpoint_path)
    samples = generate_samples(model=model, num_samples=500, return_trajectory=False)

    print(f"Generated data samples of shape {tuple(samples.generated.shape)}.")