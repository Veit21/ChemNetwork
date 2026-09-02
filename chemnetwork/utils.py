import random
import logging

import torch

import numpy as np


log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets all seeds to a given int.

        Args:
            seed (int): Seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device="cpu") -> torch.device:
    """_summary_

        Args:
            device (str | torch.device, optional): _description_. Defaults to "cpu".

        Raises:
            ValueError: _description_

        Returns:
            torch.device: _description_
    """
    requested = str(device).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    try:
        resolved = torch.device(requested)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not interpret '{device}' as a torch device. Choose from ['cpu', 'cuda', 'mps', 'auto'].") from error

    if resolved.type == "cuda" and not torch.cuda.is_available():
        log.warning(f"Device '{device}' was requested, but no CUDA device is available. Falling back to the CPU.")
        return torch.device("cpu")

    if resolved.type == "mps" and not torch.backends.mps.is_available():
        log.warning(f"Device '{device}' was requested, but no MPS device is available. Falling back to the CPU.")
        return torch.device("cpu")

    return resolved
