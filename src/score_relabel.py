import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from src.compare import boot_ci

if __name__ == "__main__":
    df = pd.read_csv("data/processed/campaign_relabel.csv", encoding="utf-8-sig")
    print("second-pass label counts:\n", df["label2"].value_counts(), "\n")
    df = df[df["label2"].isin(["Positive", "Negative", "Neutral"])].reset_index(drop=True)
    print(f"{len(df)} comments kept (not excluded)")
    print(f"agreement with first pass: {(df.label == df.label2).mean():.1%}, "
          f"Cohen's kappa {cohen_kappa_score(df.label, df.label2):.2f}\n")

    y = df["label2"].to_numpy()
    rng = np.random.default_rng(0)
    for col in [c for c in df.columns if c.startswith("pred_")]:
        p = df[col].to_numpy()
        lo, hi = boot_ci(y, p, rng)
        print(f"{col:<20} macro F1 {f1_score(y, p, average='macro'):.3f} "
              f"({lo:.3f} to {hi:.3f})  accuracy {(y == p).mean():.3f}")