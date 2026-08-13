import pytest
import torch

from app.serialization import downsample_trajectory_tensor

@pytest.mark.parametrize("max_steps", [2, 32, 50, 100])
def test_downsample_trajectory_tensor_dimensions_diff_steps(max_steps: int) -> None:
    """Tests that assert lengths of differently downsampled trajectories match respective max_steps.
    """
    test_trajectory = torch.rand(1000, 10, 2)   # Define a mock tensor with fixed (T, N, D), T=time_steps, N=num_points, D=dimensions
    test_trajectory_downsampled = downsample_trajectory_tensor(
        trajectory=test_trajectory,
        max_steps=max_steps
    )
    assert test_trajectory_downsampled.shape[0] == max_steps

@pytest.mark.parametrize("max_steps", [2, 21, 50])
def test_downsample_trajectory_tensor_final_frame(max_steps: int) -> None:
    """Tests that assert the final frame of the original trajectory is preserved after downsampling differently.
    """
    test_trajectory = torch.rand(1000, 10, 2)   # Define a mock tensor with fixed (T, N, D), T=time_steps, N=num_points, D=dimensions
    final_frame_original = test_trajectory[-1]  # Get the final frame of the original trajectory
    test_trajectory_downsampled = downsample_trajectory_tensor(     # Downsample trajectory
        trajectory=test_trajectory,
        max_steps=max_steps
    )
    final_frame_downsampled = test_trajectory_downsampled[-1]       # Get final frame of downsampled trajectory
    torch.testing.assert_close(final_frame_original, final_frame_downsampled)

@pytest.mark.parametrize("max_steps", [2, 21, 50])
def test_downsample_trajectory_tensor_initial_frame(max_steps: int) -> None:
    """Tests that assert the initial frame of the original trajectory is preserved after downsampling differently.
    """
    test_trajectory = torch.rand(1000, 10, 2)   # Define a mock tensor with fixed (T, N, D), T=time_steps, N=num_points, D=dimensions
    initial_frame_original = test_trajectory[0]  # Get the initial frame of the original trajectory
    test_trajectory_downsampled = downsample_trajectory_tensor(     # Downsample trajectory
        trajectory=test_trajectory,
        max_steps=max_steps
    )
    initial_frame_downsampled = test_trajectory_downsampled[0]       # Get initial frame of downsampled trajectory
    torch.testing.assert_close(initial_frame_original, initial_frame_downsampled)


@pytest.mark.parametrize("max_steps", [101, 199, 965])
def test_downsample_trajectory_tensor_max_steps_larger_T_original(max_steps: int) -> None:
    """Tests that a trajectory stays identical if being downsampled with max_steps larger than the actual number of frames.
    """
    test_trajectory = torch.rand(100, 10, 2)    # Define a mock tensor with fixed (T, N, D), T=time_steps, N=num_points, D=dimensions
    test_trajectory_downsampled = downsample_trajectory_tensor(     # Downsample trajectory
        trajectory=test_trajectory,
        max_steps=max_steps
    )
    torch.testing.assert_close(test_trajectory, test_trajectory_downsampled)