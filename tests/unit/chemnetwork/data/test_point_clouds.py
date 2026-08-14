import pytest
import torch

from chemnetwork.data.point_clouds import PointCloudGenerator


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_draw_x0_and_x1_data(batch_size: int) -> None:
    """The methods "draw_x0()" and "draw_x1()" from the PointCloudGenerator class should
    both return tensors of shape (batch_size, 2).
    """
    data_generator  = PointCloudGenerator(num_samples=batch_size, target_name="moons", target_noise=0.2)
    x0_batch        = data_generator.draw_source()
    x1_batch        = data_generator.draw_target()

    assert x0_batch.shape == x1_batch.shape == (batch_size, 2)

def test_draw_x0_and_x1_dtype_matches_model_dtype():
    """dtype of the data (both initial and final) should match the model's expected dtype, torch.float32.
    """
    data_generator  = PointCloudGenerator(num_samples=4, target_name="moons", target_noise=0.2)
    x0_batch        = data_generator.draw_source()
    x1_batch        = data_generator.draw_target()
    assert x0_batch.dtype == x1_batch.dtype == torch.float32