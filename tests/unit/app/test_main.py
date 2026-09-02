from fastapi.testclient import TestClient

from app.main import app

# Define test client
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200