import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

TRAIN, TEST = "data/processed/train.csv", "data/processed/test.csv"
RESULTS = Path("reports/baseline_results.json")


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    return Pipeline([("features", features), ("clf", clf)])


if __name__ == "__main__":
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    for d in (train, test):
        d["clean"] = d["clean"].fillna("")

    # 1) pick C by video-grouped cross-validation on train only
    gkf = GroupKFold(n_splits=5)
    best_c, best_f1 = None, -1.0
    for C in [0.5, 1, 2, 4, 8]:
        pred = cross_val_predict(build_pipeline(C), train["clean"], train["sentiment"],
                                 groups=train["video_id"], cv=gkf)
        f1 = f1_score(train["sentiment"], pred, average="macro")
        print(f"C={C:<4} grouped-CV macro F1: {f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1 = C, f1
    print(f"\nBest C: {best_c} (CV macro F1 {best_f1:.3f})")

    # 2) fit on all of train, evaluate once on the test set
    model = build_pipeline(best_c).fit(train["clean"], train["sentiment"])
    test["pred_tfidf"] = model.predict(test["clean"])
    test_f1 = f1_score(test["sentiment"], test["pred_tfidf"], average="macro")
    print(f"\nTEST macro F1: {test_f1:.3f}")
    print(classification_report(test["sentiment"], test["pred_tfidf"], zero_division=0))

    # 3) save model, predictions (for error analysis) and the score
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/tfidf_lr.joblib")
    test.to_csv("data/processed/test_predictions.csv", index=False)

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results["tfidf_lr"] = round(float(test_f1), 4)
    results["tfidf_lr_cv"] = round(float(best_f1), 4)
    RESULTS.write_text(json.dumps(results, indent=2))
    print("Saved model, predictions and results.")