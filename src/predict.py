"""Inference for the Hinglish sentiment model (TF-IDF + logistic regression)."""
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from src.cleaning import clean_text

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tfidf_lr.joblib"


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)):
    """Load the saved model once and reuse it."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first with: python -m src.tfidf_model"
        )
    return joblib.load(path)


def predict(texts: list[str]) -> list[dict]:
    """Return the predicted label and class probabilities for each comment."""
    model = load_model()
    cleaned = [clean_text(t) for t in texts]
    proba = model.predict_proba(cleaned)
    classes = list(model.classes_)
    results = []
    for text, p in zip(texts, proba):
        best = max(range(len(classes)), key=lambda i: p[i])
        results.append({
            "text": text,
            "label": classes[best],
            "probabilities": {c: round(float(x), 3) for c, x in zip(classes, p)},
        })
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    comments = sys.argv[1:] or ["yaar ye ad toh bakwaas tha \U0001F602", "bahut mast video hai bhai"]
    for r in predict(comments):
        print(r["label"], r["probabilities"], "|", r["text"])
