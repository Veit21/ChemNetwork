import pytest

from fastapi import status
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from app.main import app
from app.schemas import GenerateRequest
from chemnetwork.data.point_clouds import TargetDistribution

def test_available() -> None:
    """Test that the serving checkpoints are loaded and appear as "available" in the model registry.
    """

    # Start the test client with lifespan
    with TestClient(app) as client:

        # Inside "with TestClient" block, lifespan of "app" starts
        response = client.get(url="/samples/available")
        response_expected = {
            'targets': [
                {'id': 'checkerboard','label': 'Checkerboard'},
                {'id': 'moons','label': 'Two moons'}],
            'default': 'moons'
            }
        assert response.status_code == 200
        assert response.json() == response_expected

@pytest.mark.parametrize("target", ["moons", "checkerboard"])
def test_generate_loads_available_models(target: str) -> None:

    # Define test request
    test_request = GenerateRequest(
        num_samples=2,
        integration_steps=10,
        return_trajectory=True,
        target=TargetDistribution(target),
        device="cpu"
    )

    # Again, first start the test client with lifespan
    with TestClient(app) as client:

        # Generate response
        response = client.post(
            url="/samples/generate",
            json=jsonable_encoder(test_request) # NOTE: Pydantic model needs to be converted to JSON!
        )
        assert response.status_code == 200

@pytest.mark.parametrize("target", ["swirl", "circle"])
def test_generate_unserved_targets(target: str) -> None:

    # Define test request
    test_request = GenerateRequest(
        num_samples=2,
        integration_steps=10,
        return_trajectory=True,
        target=TargetDistribution(target),
        device="cpu"
    )

    # Again, first start the test client with lifespan
    with TestClient(app) as client:

        # Generate response
        response = client.post(
            url="/samples/generate",
            json=jsonable_encoder(test_request) # NOTE: Pydantic model needs to be converted to JSON!
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.parametrize("target", ["spam", "eggs"])
def test_generate_unknown_targets(target: str) -> None:

    with pytest.raises(ValueError) as e:

        # Define test request
        test_request = GenerateRequest(
            num_samples=2,
            integration_steps=10,
            return_trajectory=True,
            target=TargetDistribution(target),
            device="cpu"
        )

        # Again, first start the test client with lifespan
        with TestClient(app) as client:

            # Generate response that triggers an error
            response = client.post(
                url="/samples/generate",
                json=jsonable_encoder(test_request) # NOTE: Pydantic model needs to be converted to JSON!
            )

    assert str(e.value) == f"'{target}' is not a valid TargetDistribution"