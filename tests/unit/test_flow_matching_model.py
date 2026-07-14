import pytest
import torch
from torch import nn

from chemnetwork.models.flow_matching import FlowModel


@pytest.fixture
def model() -> FlowModel:
    """Small FlowModel for each test."""
    return FlowModel(in_features=3, hidden_features=16, out_features=2)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_forward_output_shape(model: FlowModel, batch_size: int) -> None:
    """forward() must return (batch_size, out_features), for any batch size.
    """
    x = torch.randn(batch_size, 2)
    t = torch.randn(batch_size, 1)

    output = model(x_in=x, t=t)

    assert output.shape == (batch_size, model.out_features)


def test_forward_concatenates_time_before_space() -> None:
    """forward() should build [t, x, y] before feeding the MLP.
    """
    model       = FlowModel(in_features=3, hidden_features=16, out_features=2)
    model.model = nn.Identity()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t = torch.tensor([[0.1], [0.9]])

    result      = model(x, t)
    expected    = torch.tensor([[0.1, 1.0, 2.0], [0.9, 3.0, 4.0]])

    torch.testing.assert_close(result, expected)
