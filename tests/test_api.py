import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.predict import MODEL_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_predict_ok(client):
    r = client.post("/predict", json={"texts": ["bahut mast video hai bhai"]})
    assert r.status_code == 200
    label = r.json()["predictions"][0]["label"]
    assert label in {"Positive", "Negative", "Neutral"}


def test_predict_rejects_empty_list(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422


def test_predict_rejects_too_long_text(client):
    assert client.post("/predict", json={"texts": ["a" * 2001]}).status_code == 422
