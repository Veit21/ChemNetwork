import pytest
import torch

from torch import nn

from chemnetwork.models.drifting import DriftModel, DrifterObject

@pytest.fixture
def model() -> DriftModel:
    """Small FFN for testing purposes.
    """
    return DriftModel(in_features=2, hidden_features=16, out_features=2)

@pytest.fixture
def drifter_object(model: DriftModel) -> DrifterObject:
    """DrifterObject instance for testing purposes.
    """
    return DrifterObject(model=model)

@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_model_output_shape(model: DriftModel, batch_size: int) -> None:
    """Model forward call must return the same output dimensions as the input
    """
    x       = torch.randn(batch_size, 2)
    output  = model(x)
    assert output.shape == x.shape == (batch_size, model.out_features)

def test_computing_pairwise_distances(drifter_object: DrifterObject) -> None:
    """Tests the computation of pairwise distances between two sets of samples.
    """

    # Define two sets of samples
    x_test = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    y_test = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    # Compute pairwise distances
    xydist_result = drifter_object._cdist(x_test, y_test)
    xydist_expected = torch.tensor([
        [1.4142, 2.8284],
        [1.0000, 2.2361],
        [1.0000, 2.2361]
    ])
    torch.testing.assert_close(xydist_result, xydist_expected, rtol=1e-4, atol=1e-4)

def test_V_matches_expected_shape(drifter_object: DrifterObject) -> None:
    """Tests whether the drifting field V matches the correct dimensions of the input data.
    """

    # Define two sets of samples
    x_0 = torch.randn(5, 2)
    x_1 = torch.randn(5, 2)

    # Compute network prediction and drifting field
    x_hat, V = drifter_object._compute_xhat_and_V(x_0, x_1)

    assert x_0.shape == x_hat.shape == V.shape

def test_xhat_requires_grad(drifter_object: DrifterObject) -> None:
    """Tests whether the network prediction is attached to the computational graph for backpropagation.
    """

    # Define two sets of samples
    x_0 = torch.randn(5, 2)
    x_1 = torch.randn(5, 2)

    # Compute network prediction
    x_hat, _ = drifter_object.compute_network_prediction_and_drifted_samples(x_0, x_1)

    assert x_hat.requires_grad

def test_drifted_sample_not_requires_grad(drifter_object: DrifterObject) -> None:
    """Tests whether the the drifted sample x_drifted is NOT attached to the computational graph,
    i.e. backpropagation is not performed through it.
    """

    # Define two sets of samples
    x_0 = torch.randn(5, 2)
    x_1 = torch.randn(5, 2)

    # Compute network prediction
    _, x_drifted = drifter_object.compute_network_prediction_and_drifted_samples(x_0, x_1)

    assert not x_drifted.requires_grad