import pytest
import torch
from torch import nn

from chemnetwork.models.flow_matching import FlowModel, MSELoss, FlowMatcher, NumericalODESolver


@pytest.fixture
def model() -> FlowModel:
    """Small FlowModel for each test.
    """
    return FlowModel(in_features=3, hidden_features=16, out_features=2)


@pytest.fixture
def loss() -> MSELoss:
    """An MSE loss object with the velocity field as target for each test case.
    """
    return MSELoss(regression_target="v")


@pytest.fixture
def flow_matcher() -> FlowMatcher:
    """Flow matcher instance that computes a linear interpolation for pairs of (x_0, x_1).
    """
    return FlowMatcher()


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
    "u_t, v_hat, loss_expected",
    [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 3.0]]), torch.tensor(1.)),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0]]), torch.tensor(0.)),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor(16.))
    ]
)
def test_calculation_mse_loss(loss: MSELoss, u_t:torch.tensor, v_hat: torch.tensor, loss_expected: torch.tensor) -> None:
    """For different input tensors of varying batch size, the MSE loss function should compute a deterministic value. 
    """
    loss_result = loss(u_t, v_hat)

    torch.testing.assert_close(loss_result, loss_expected)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_sample_interpolant_and_target(flow_matcher: FlowMatcher, batch_size: int) -> None:
    """Sampling time t, the interpolant x_t and the regression target u_t should always return tensors of the same, deterministic shape.
    """
    x_0_mock    = torch.randn(batch_size, 2)
    x_1_mock    = torch.randn(batch_size, 2)
    t, x_t, u_t = flow_matcher.sample_interpolant_and_target(x_0=x_0_mock, x_1=x_1_mock)

    assert x_t.shape == u_t.shape
    assert t.shape[0] == x_t.shape[0] == u_t.shape[0] == batch_size


def test_sample_interpolant_rejects_mismatched_shapes(flow_matcher: FlowMatcher):
    """The smaple_interpolant_and_target() function should make sure that its arguments x0 and x1 have the same shape.
    """
    with pytest.raises(AssertionError):
        flow_matcher.sample_interpolant_and_target(torch.randn(4, 2), torch.randn(5, 2))


def test_interpolant_matches_endpoints_at_t_bounds(monkeypatch, flow_matcher):
    """At t=0 and t=1, the interpolation formula should compute x0 and x1, respectively.
    """
    x_0 = torch.tensor([[0.0, 0.0]])
    x_1 = torch.tensor([[10.0, 10.0]])

    monkeypatch.setattr(flow_matcher, "_sample_t", lambda x: torch.zeros(x.shape[0], 1))
    _, x_t_at_0, _ = flow_matcher.sample_interpolant_and_target(x_0, x_1)
    torch.testing.assert_close(x_t_at_0, x_0)

    monkeypatch.setattr(flow_matcher, "_sample_t", lambda x: torch.ones(x.shape[0], 1))
    _, x_t_at_1, _ = flow_matcher.sample_interpolant_and_target(x_0, x_1)
    torch.testing.assert_close(x_t_at_1, x_1)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_euler_integration_shape(batch_size: int, model: FlowModel):
    """Tests the output shape of the euler integrator.
    """
    solver = NumericalODESolver(model=model, solver="euler", integration_steps=100, return_trajectory=False)
    x0 = torch.randn(batch_size, 2)
    x1_hat = solver(in_tensor=x0)

    assert x0[None].shape == x1_hat.shape


def test_euler_integration_constant_field_known_displacement():
    """A constant vector field v should displace every point by exactly v over [0, 1]."""
    v = torch.tensor([1.0, -2.0])
    const_model = lambda x, t: v.expand_as(x)
    solver = NumericalODESolver(model=const_model, solver="euler", integration_steps=200, return_trajectory=False)
    x0 = torch.zeros(4, 2)
    x1_hat = solver(in_tensor=x0)
    torch.testing.assert_close(x1_hat, x0[None] + v, atol=1e-3, rtol=0)

def test_sample_t_follows_device_and_dtype_of_data(flow_matcher: FlowMatcher) -> None:
    """Time points have to be drawn on the same device/dtype as the data they are paired with.
    """
    x = torch.randn(8, 2)
    t = flow_matcher._sample_t(x=x)

    assert t.device == x.device
    assert t.dtype == x.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_training_step_runs_entirely_on_cuda(flow_matcher: FlowMatcher, loss: MSELoss) -> None:
    """A full interpolation + forward + loss step must work without any device mismatch on the GPU.
    """
    model   = FlowModel(in_features=3, hidden_features=16, out_features=2).cuda()
    x_0     = torch.randn(8, 2, device="cuda")
    x_1     = torch.randn(8, 2, device="cuda")

    t, x_t, u_t = flow_matcher.sample_interpolant_and_target(x_0=x_0, x_1=x_1)
    v_hat       = model(x_in=x_t, t=t)
    loss_val    = loss(u_t=u_t, v_hat=v_hat)

    assert t.device.type == x_t.device.type == "cuda"
    assert loss_val.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available.")
def test_euler_integration_stays_on_cuda(model: FlowModel) -> None:
    """The solver builds its own time grid, which must not fall back to the CPU.
    """
    solver  = NumericalODESolver(model=model.cuda(), solver="euler", integration_steps=10, return_trajectory=False)
    x_1_hat = solver(in_tensor=torch.randn(4, 2, device="cuda"))

    assert x_1_hat.device.type == "cuda"
