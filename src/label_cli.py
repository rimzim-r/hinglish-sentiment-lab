import sys

import pandas as pd

PATH = "data/processed/campaign_label_sample.csv"
KEYS = {"1": "Positive", "2": "Negative", "3": "Neutral"}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(PATH, encoding="utf-8-sig")
    df["label"] = df["label"].fillna("").astype(str)
    todo = df.index[df["label"].str.strip() == ""].tolist()
    print(f"{len(df) - len(todo)} already labeled, {len(todo)} to go.")
    print("Keys: 1 = Positive, 2 = Negative, 3 = Neutral, s = skip, q = save and quit\n")

    for k, i in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {df.at[i, 'comment']}")
        while True:
            key = input("label (1/2/3/s/q): ").strip().lower()
            if key in KEYS or key in ("s", "q"):
                break
        if key == "q":
            break
        if key in KEYS:
            df.at[i, "label"] = KEYS[key]
        df.to_csv(PATH, index=False, encoding="utf-8-sig")  # saved after every comment
        print()

    df.to_csv(PATH, index=False, encoding="utf-8-sig")
    print(df["label"].replace("", "(empty)").value_counts())