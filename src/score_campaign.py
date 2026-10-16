from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.baselines import hf_predict, vader_predict
from src.cleaning import clean_text
from src.compare import boot_ci

LABELED = "data/processed/campaign_label_sample.csv"
VALID = {"Positive", "Negative", "Neutral"}

if __name__ == "__main__":
    df = pd.read_csv(LABELED)
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    bad = df[~df["label"].isin(VALID)]
    print(f"{len(df) - len(bad)} labeled, {len(bad)} unlabeled or invalid (dropped)")
    df = df[df["label"].isin(VALID)].reset_index(drop=True)
    print(df["label"].value_counts(), "\n")

    raw = df["comment"].astype(str).tolist()
    preds = {"vader": vader_predict(raw)}

    tfidf = joblib.load("models/tfidf_lr.joblib")
    preds["tfidf"] = tfidf.predict([clean_text(t) for t in raw]).tolist()

    preds["xlmr_twitter"] = hf_predict(raw, "cardiffnlp/twitter-xlm-roberta-base-sentiment")

    if Path("models/distilbert_ft").exists():
        preds["distilbert_ft"] = hf_predict(raw, "models/distilbert_ft")

    y = df["label"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for name, p in preds.items():
        df[f"pred_{name}"] = p
        p = np.array(p)
        lo, hi = boot_ci(y, p, rng)
        f1 = f1_score(y, p, average="macro")
        rows.append((name, f1, lo, hi))
        print(f"=== {name} === macro F1 {f1:.3f}")
        print(classification_report(y, p, zero_division=0))

    table = pd.DataFrame(rows, columns=["model", "f1", "lo", "hi"]).sort_values("f1", ascending=False)
    print(table.round(3).to_string(index=False))
    lines = ["| Model | Macro F1 (campaign) | 95% bootstrap CI |", "|---|---|---|"]
    lines += [f"| {r.model} | {r.f1:.3f} | {r.lo:.3f} to {r.hi:.3f} |" for r in table.itertuples()]
    Path("reports/campaign_results.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv("data/processed/campaign_label_predictions.csv", index=False, encoding="utf-8-sig")
    print("\nSaved reports/campaign_results.md")