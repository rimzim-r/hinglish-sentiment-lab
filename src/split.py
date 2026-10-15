import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROCESSED = "data/processed/youtube_hinglish_clean.csv"


def make_split(test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(PROCESSED)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["sentiment"], groups=df["video_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    return train, test


if __name__ == "__main__":
    train, test = make_split()
    print("train:", train.shape, "videos:", train["video_id"].nunique())
    print("test: ", test.shape, "videos:", test["video_id"].nunique())
    print("\ntrain labels:\n", train["sentiment"].value_counts(normalize=True).round(3))
    print("\ntest labels:\n", test["sentiment"].value_counts(normalize=True).round(3))