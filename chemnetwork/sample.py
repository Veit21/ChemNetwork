###############################################################
#
#   Script for sampling from a trained flow matching model.
#
###############################################################

# Imports
import torch

from pathlib import Path
from collections import namedtuple
from typing import Dict, Any

from chemnetwork.models.flow_matching import FlowModel, NumericalODESolver
from chemnetwork.data.point_clouds import PointCloudGenerator

LoadedTuple = namedtuple('Loaded', ['model', 'config'])
OutputTuple = namedtuple('Output', ['source', 'generated', 'target'])

def load_model(checkpoint_path: Path) -> LoadedTuple:
    """Loads a trained FlowModel from a checkpoint file.

        Args:
            checkpoint_path (Path): Path to the .pt checkpoint saved during training.

        Returns:
            LoadedTuple: A named tuple containing the loaded model and its configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FlowModel(**checkpoint["config"]["model"]["params"])

    model.load_state_dict(checkpoint["model"])
    model.eval()
    # print(f"Model weights loaded from '{checkpoint_path}'.")

    return LoadedTuple(model=model, config=checkpoint["config"])

def generate_samples(
    model: FlowModel,
    cfg: Dict[str, Any],
    num_samples: int        = 500,
    integration_steps: int  = 100,
    return_trajectory: bool = False,
    device: torch.device    = torch.device("cpu") 
) -> OutputTuple:
    """_summary_

        Args:
            model (FlowModel): _description_
            cfg (Dict[str, Any]): _description_
            num_samples (int, optional): _description_. Defaults to 500.
            integration_steps (int, optional): _description_. Defaults to 100.
            return_trajectory (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to torch.device("cpu").

        Returns:
            OutputTuple: _description_
    """

    # Move model to requested device
    model = model.to(device)    # TODO: How costly is it to move the model per API call? More elegant solution?

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
