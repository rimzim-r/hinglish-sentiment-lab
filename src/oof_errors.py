import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.tfidf_model import build_pipeline

BEST_C = 2  # set this to the best C that src.tfidf_model printed for you

if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    train["clean"] = train["clean"].fillna("")

    proba = cross_val_predict(
        build_pipeline(BEST_C), train["clean"], train["sentiment"],
        groups=train["video_id"], cv=GroupKFold(n_splits=5),
        method="predict_proba")
    classes = np.array(sorted(train["sentiment"].unique()))
    train["oof_pred"] = classes[proba.argmax(axis=1)]
    train["oof_conf"] = proba.max(axis=1).round(3)

    wrong = train[train["oof_pred"] != train["sentiment"]]
    wrong = wrong.sort_values("oof_conf", ascending=False)
    print(len(wrong), "out-of-fold errors out of", len(train))

    out = wrong[["video_id", "sentiment", "oof_pred", "oof_conf", "comment"]].copy()
    out["error_type"] = ""
    out.to_csv("reports/oof_errors_to_review.csv", index=False)
    print(out.head(40).to_string())