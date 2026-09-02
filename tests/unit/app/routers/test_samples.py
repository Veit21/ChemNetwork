import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_model_registry
from app.model_registry import ModelRegistry

# TODO: Apparently the "app" instance has no "registry" attribute -> need lifespan to run in test? Probably yes.
# Define test client
client = TestClient(app)

@pytest.mark.skip("How to cope with dependencies?")
def test_available():
    response = client.get("/samples/available")
    print(response)