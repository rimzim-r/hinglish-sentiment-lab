import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

TEST = "data/processed/test.csv"
OUT = Path("reports/baseline_results.json")


def vader_predict(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    preds = []
    for t in texts:
        c = sia.polarity_scores(str(t))["compound"]
        preds.append("Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral")
    return preds


def hf_predict(texts, model_name, batch_size=32):
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_name,
                   truncation=True, max_length=128)
    preds = []
    texts = [str(t) for t in texts]
    for i in range(0, len(texts), batch_size):
        out = clf(texts[i:i + batch_size])
        preds.extend(o["label"].capitalize() for o in out)
    return preds


def evaluate(name, y_true, y_pred, results):
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {name} === macro F1: {f1:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    results[name] = round(float(f1), 4)


if __name__ == "__main__":
    test = pd.read_csv(TEST)
    y = test["sentiment"].tolist()
    raw = test["comment"].tolist()
    results = {}

    evaluate("vader", y, vader_predict(raw), results)

    # Multilingual model trained on tweets; labels are negative/neutral/positive
    evaluate("xlm-roberta-twitter",
             y, hf_predict(raw, "cardiffnlp/twitter-xlm-roberta-base-sentiment"), results)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing.update(results)
    OUT.write_text(json.dumps(existing, indent=2))
    print("\nSaved:", OUT)