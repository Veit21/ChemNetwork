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
    """Turns a device string from the config into a usable torch.device.

        Accepts "cpu", "cuda", an indexed device such as "cuda:1", "mps" and "auto",
        where "auto" picks the fastest accelerator that is actually present.
        Whenever the requested accelerator is unavailable, the CPU is returned
        instead (with a warning), so a run never dies just because it was launched
        on a machine without a GPU.

        Args:
            device (str | torch.device, optional): Requested device as given in the config. Defaults to "cpu".

        Raises:
            ValueError: If the argument cannot be interpreted as a torch device at all.

        Returns:
            torch.device: The device that should actually be used.
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
        raise ValueError(f"Could not interpret '{device}' as a torch device. Choose from ['cpu', 'cuda', 'cuda:<idx>', 'mps', 'auto'].") from error

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            log.warning(f"Device '{device}' was requested, but no CUDA device is available. Falling back to the CPU.")
            return torch.device("cpu")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            log.warning(f"CUDA device index {resolved.index} was requested, but only {torch.cuda.device_count()} device(s) are visible. Falling back to the default CUDA device.")
            return torch.device("cuda")

    if resolved.type == "mps" and not torch.backends.mps.is_available():
        log.warning(f"Device '{device}' was requested, but no MPS device is available. Falling back to the CPU.")
        return torch.device("cpu")

    return resolved
