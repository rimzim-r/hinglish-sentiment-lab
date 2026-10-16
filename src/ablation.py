import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.cleaning import clean_text
from src.tfidf_model import build_pipeline

BEST_C = 2  # same value you used in oof_errors.py

CONFIGS = {
    "v1: emoji tokens, no ?, variants": dict(emoji_mode="token", keep_qmark=False, use_variants=True),
    "emoji dropped": dict(emoji_mode="drop", keep_qmark=False, use_variants=True),
    "keep ? as a token": dict(emoji_mode="token", keep_qmark=True, use_variants=True),
    "no spelling variants": dict(emoji_mode="token", keep_qmark=False, use_variants=False),
}

if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    y, groups = train["sentiment"], train["video_id"]
    for name, cfg in CONFIGS.items():
        text = train["comment"].apply(lambda t: clean_text(t, **cfg))
        folds = []
        for tr, va in GroupKFold(n_splits=5).split(text, y, groups):
            m = build_pipeline(BEST_C).fit(text.iloc[tr], y.iloc[tr])
            folds.append(f1_score(y.iloc[va], m.predict(text.iloc[va]), average="macro"))
        print(f"{name:<36} mean macro F1 {np.mean(folds):.3f}  (std across folds {np.std(folds):.3f})")