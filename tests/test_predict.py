import pytest

from src.predict import MODEL_PATH, predict

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")


def test_predict_returns_valid_structure():
    out = predict(["bahut mast video hai bhai", "ye video bakwaas tha"])
    assert len(out) == 2
    for r in out:
        assert r["label"] in {"Positive", "Negative", "Neutral"}
        assert abs(sum(r["probabilities"].values()) - 1.0) < 0.01


def test_predict_handles_empty_and_emoji_only():
    assert len(predict(["", "\U0001F602"])) == 2
