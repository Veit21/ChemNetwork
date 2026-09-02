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
    device: torch.device=torch.device("cpu") 
) -> OutputTuple:
    """_summary_

        Args:
            model (FlowModel): _description_
            cfg (dict[str, dict]): _description_
            num_samples (int, optional): _description_. Defaults to 500.
            integration_steps (int, optional): _description_. Defaults to 100.
            return_trajectory (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to torch.device("cpu").

        Returns:
            OutputTuple: _description_
    """

    # Move model to requested device
    model = model.to(device)    # TODO: So far, client is not informed about device computations are performed on!

    # Instantiate data generator
    data_generator = PointCloudGenerator(
        num_samples=num_samples,
        target_name=cfg["data"]["target_distribution"],
        target_noise=cfg["data"]["target_data_noise"],
        device=device
    )

    # Instantiate solver
    solver = NumericalODESolver(
        model=model,
        solver="euler",
        integration_steps=integration_steps,
        return_trajectory=return_trajectory
    )

    # Draw source and target data
    source_data = data_generator.draw_source()
    target_data = data_generator.draw_target()

    # Inference
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