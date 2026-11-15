from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction():
    payload = {
        "Type": "M",
        "Air temperature": 300,
        "Process temperature": 310,
        "Rotational speed": 1500,
        "Torque": 45,
        "Tool wear": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["success"] is True
    assert result["machine_failure"] in [0, 1]
    assert 0 <= result["failure_probability"] <= 1
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_invalid_machine_type():
    payload = {
        "Type": "X",
        "Air temperature": 300,
        "Process temperature": 310,
        "Rotational speed": 1500,
        "Torque": 45,
        "Tool wear": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422