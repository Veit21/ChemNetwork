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

@pytest.mark.parametrize("target_name", ["moons", "checkerboard"])
def test_draw_places_data_on_requested_device(target_name: str) -> None:
    """Both draw methods must return tensors on the device the generator was configured with.
    """
    device          = torch.device("cpu")
    data_generator  = PointCloudGenerator(num_samples=4, target_name=target_name, target_noise=0.2, device=device)

    assert data_generator.draw_source().device == device
    assert data_generator.draw_target().device == device

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
@pytest.mark.parametrize("target_name", ["moons", "checkerboard"])
def test_draw_places_data_on_cuda(target_name: str) -> None:
    """The same has to hold for a GPU.
    """
    data_generator  = PointCloudGenerator(num_samples=4, target_name=target_name, target_noise=0.2, device="cuda")

    assert data_generator.draw_source().device.type == "cuda"
    assert data_generator.draw_target().device.type == "cuda"
