import random
import torch

import numpy as np


def set_seed(seed: int) -> None:
    """Sets all seeds to a given int.

    Args:
        seed (int): Seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)