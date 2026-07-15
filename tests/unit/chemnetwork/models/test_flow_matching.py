import pytest
import torch
from torch import nn

from chemnetwork.models.flow_matching import FlowModel, MSELoss


@pytest.fixture
def model() -> FlowModel:
    """Small FlowModel for each test.

    Returns:
        FlowModel: A fresh instance of a FlowModel object for each test case.
    """
    return FlowModel(in_features=3, hidden_features=16, out_features=2)

@pytest.fixture
def loss() -> MSELoss:
    """An MSE loss object with the velocity field as target for each test case.

    Returns:
        MSELoss: An instance of the MSELoss object.
    """
    return MSELoss(regression_target="v")


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


@pytest.mark.parametrize(
    "x_0, x_1, v_hat, loss_expected",
    [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 3.0]]), torch.tensor([[2.0, 3.0]]), torch.tensor(1.)),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 3.0]]), torch.tensor([[1.0, 2.0]]), torch.tensor(0.)),
        (torch.tensor([[1.0, 1.0], [1.0, 1.0]]), torch.tensor([[2.0, 3.0], [4.0, 5.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor(16.))
    ]
)
def test_calculation_mse_loss(loss: MSELoss, x_0: torch.tensor, x_1: torch.tensor, v_hat: torch.tensor, loss_expected: torch.tensor) -> None:
    """For different input tensors of varying batch size, the MSE loss function should compute a deterministic value. 
    """
    loss_result     = loss(x_0, x_1, v_hat)

    torch.testing.assert_close(loss_result, loss_expected)