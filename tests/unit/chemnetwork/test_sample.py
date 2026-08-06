import torch
import pytest

from chemnetwork.sample import generate_samples
from chemnetwork.models.flow_matching import FlowModel
from chemnetwork.data.point_clouds import PointCloudGenerator


@pytest.fixture
def model() -> FlowModel:
    """Instantiates a FlowModel object for each test case.
    """
    return FlowModel(in_features=3, hidden_features=128, out_features=2)

@pytest.mark.parametrize("num_samples", [1, 64, 512])
def test_generate_samples_different_sizes(model: FlowModel, num_samples: int) -> None:
    """Testing the output shape of the generate_smaples() method with different input sizes.
    """
    x1_hat = generate_samples(model=model, num_samples=num_samples, integration_steps=100)

    assert x1_hat.shape == (num_samples, 2)