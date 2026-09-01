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

LoadedTuple = namedtuple('Loaded', ['model', 'config'])
OutputTuple = namedtuple('Output', ['source', 'generated', 'target'])

def load_model(checkpoint_path: Path) -> LoadedTuple:
    """Loads a trained FlowModel from a checkpoint file.

        Args:
            checkpoint_path (Path): Path to the .pt checkpoint saved during training.

        Returns:
            namedtuple: A named tuple containing the loaded model and its configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FlowModel(**checkpoint["config"]["model"]["params"])

    model.load_state_dict(checkpoint["model"])
    model.eval()
    # print(f"Model weights loaded from '{checkpoint_path}'.")

    return LoadedTuple(model=model, config=checkpoint["config"])

def generate_samples(
    model: FlowModel,
    cfg: dict[str, dict],
    num_samples: int=500,
    integration_steps: int=100,
    return_trajectory: bool=False,
) -> OutputTuple:
    """Generates samples from the learned target distribution p_1.

        TODO: Update this docstring!
        Draws source points from p_0 and integrates them along the model's
        learned vector field using an ODE solver.

        Args:
            model (FlowModel): A trained flow matching model in evaluation mode.
            num_samples (int, optional): Number of samples to generate. Defaults to 500.
            integration_steps (int, optional): Number of solver integration steps. Defaults to 100.
            return_trajectory (bool, optional): Whether to return the full trajectory of the solver. Defaults to False.
            cfg (dict, optional): Configuration dictionary for the model. Defaults to None.

        Returns:
            namedtuple: A named tuple containing the source points and the generated points.
            Output.source - Tensor of shape (num_samples, 2) containing the source points drawn from p_0.
            Output.generated - Tensor of shape (T, num_samples, 2) containing the generated points after integration.
            T is the number of integration steps if return_trajectory is True, otherwise T=1.
            Output.target - The ground truth target distribution the model has been trained on.
    """
    data_generator = PointCloudGenerator(
        num_samples=num_samples,
        target_name=cfg["data"]["target_distribution"],
        target_noise=cfg["data"]["target_data_noise"]
    )
    solver = NumericalODESolver(
        model=model, solver="euler",
        integration_steps=integration_steps,
        return_trajectory=return_trajectory
    )
    source_data = data_generator.draw_source()
    target_data = data_generator.draw_target()

    with torch.no_grad():
        predicted_data = solver(source_data)
    
    return OutputTuple(
        source=source_data,
        generated=predicted_data,
        target=target_data
    )


if __name__ == "__main__":
    # Small test: load a checkpoint and generate a batch of samples.
    # project_root = Path(__file__).resolve().parents[1]
    # checkpoint_path = project_root / Path("checkpoints/MultiLayerPerceptron_weights_step_50000.pt")

    # model = load_model(checkpoint_path=checkpoint_path)
    # samples = generate_samples(model=model, num_samples=500, return_trajectory=False, cfg={})

    # print(f"Generated data samples of shape {tuple(samples.generated.shape)}.")
    pass