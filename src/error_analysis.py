import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

PRED = "data/processed/test_predictions.csv"
FIG = Path("reports/figures")
LABELS = ["Negative", "Neutral", "Positive"]
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED)
    df["clean"] = df["clean"].fillna("")
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    print("Prediction columns found:", pred_cols)

    # 1) confusion matrices
    for col in pred_cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            df["sentiment"], df[col], labels=LABELS, ax=ax,
            cmap="Blues", colorbar=False)
        ax.set_title(col)
        fig.tight_layout()
        fig.savefig(FIG / f"07_confusion_{col}.png", dpi=150)
        plt.close(fig)

    # 2) most confident errors from the TF-IDF model
    model = joblib.load("models/tfidf_lr.joblib")
    proba = model.predict_proba(df["clean"])
    classes = list(model.classes_)
    df["tfidf_conf"] = proba.max(axis=1).round(3)
    wrong = df[df["pred_tfidf"] != df["sentiment"]].copy()
    wrong = wrong.sort_values("tfidf_conf", ascending=False)
    print(f"\nTF-IDF errors: {len(wrong)} of {len(df)} ({len(wrong)/len(df):.1%})")
    print("\n--- 25 most confident TF-IDF errors ---")
    for _, r in wrong.head(25).iterrows():
        print(f"[true={r['sentiment']:<8} pred={r['pred_tfidf']:<8} conf={r['tfidf_conf']}] {r['comment'][:140]}")

    # 3) strongest words per class
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_
    word_idx = [i for i, n in enumerate(names) if n.startswith("word__")]
    print("\n--- Top words per class (TF-IDF) ---")
    for k, cls in enumerate(model.named_steps["clf"].classes_):
        top = sorted(word_idx, key=lambda i: coef[k][i], reverse=True)[:15]
        print(f"{cls}: " + ", ".join(names[i].replace("word__", "") for i in top))

    # 4) macro F1 per test video, and per slice
    print("\n--- Macro F1 by video ---")
    for vid, g in df.groupby("video_id"):
        print(f"{vid}: n={len(g):<4} F1={f1_score(g['sentiment'], g['pred_tfidf'], average='macro'):.3f}")

    df["has_emoji"] = df["comment"].astype(str).apply(lambda t: bool(EMOJI_RE.search(t)))
    df["short"] = df["clean"].str.split().str.len() <= 3
    df["correct"] = df["sentiment"] == df["pred_tfidf"]
    print("\n--- Accuracy by slice ---")
    print("with emoji   :", df[df.has_emoji]["correct"].mean().round(3), "(n=%d)" % df.has_emoji.sum())
    print("without emoji:", df[~df.has_emoji]["correct"].mean().round(3))
    print("<=3 words    :", df[df.short]["correct"].mean().round(3), "(n=%d)" % df.short.sum())
    print(">3 words     :", df[~df.short]["correct"].mean().round(3))

    # 5) file to read and tag by hand
    review = wrong[["video_id", "sentiment", "pred_tfidf", "tfidf_conf", "comment"]].copy()
    review["error_type"] = ""
    review.to_csv("reports/errors_to_review.csv", index=False)
    print("\nSaved reports/errors_to_review.csv")