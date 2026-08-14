import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client() -> TestClient:
    """Test client (does not run the application lifespan).
    """
    return TestClient(app)


@pytest.mark.skip(reason="Testing another time.")
def test_health_reports_ok_and_configured_checkpoint(client: TestClient) -> None:
    """The health endpoint answers with status "ok" and the configured checkpoint path.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_paths": str(settings.checkpoint_paths),
    }
