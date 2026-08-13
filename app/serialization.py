import torch


def downsample_trajectory_tensor(trajectory: torch.tensor, max_steps: int) -> torch.tensor:
    """Downsamples a trajectory of datapoints to a given maximum number of frames.

    Args:
        trajectory (torch.tensor): Input trajectory to be downsampled.
        max_steps (int): Maximum number of steps to downsample to.

    Returns:
        torch.tensor: Downsampled tensor.
    """
    num_frames = min(trajectory.shape[0], max_steps)
    idx = torch.linspace(0, trajectory.shape[0] - 1, steps=num_frames).round().long()
    trajectory_downsampled = trajectory[idx]
    return trajectory_downsampled

def typecast_and_round_output(in_tensor: torch.tensor, decimals: int=3) -> list:
    """Rounds the values of a tensor to a given number of decimal points
    and casts it to a simple python list.

    Args:
        in_tensor (torch.tensor): Input tensor to be modifeid.
        decimals (int, optional): Number of decimal points to truncate to. Defaults to 3.

    Returns:
        list: Truncated and typecasted 'tensor'.
    """
    return in_tensor.to(torch.float64).round(decimals=decimals).tolist()