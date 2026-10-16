import sys

import pandas as pd

SRC = "data/processed/campaign_label_predictions.csv"
OUT = "data/processed/campaign_relabel.csv"
KEYS = {"1": "Positive", "2": "Negative", "3": "Neutral", "x": "Exclude"}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        df = pd.read_csv(OUT, encoding="utf-8-sig")
    except FileNotFoundError:
        df = pd.read_csv(SRC, encoding="utf-8-sig").sample(frac=1, random_state=1).reset_index(drop=True)
        df["label2"] = ""
    df["label2"] = df["label2"].fillna("").astype(str)
    todo = df.index[df["label2"] == ""].tolist()
    print(f"{len(df) - len(todo)} done, {len(todo)} to go.")
    print("Keys: 1 = Positive, 2 = Negative, 3 = Neutral, x = exclude, q = save and quit\n")

    for k, i in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {df.at[i, 'comment']}")
        while True:
            key = input("label (1/2/3/x/q): ").strip().lower()
            if key in KEYS or key == "q":
                break
        if key == "q":
            break
        df.at[i, "label2"] = KEYS[key]
        df.to_csv(OUT, index=False, encoding="utf-8-sig")
        print()

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df["label2"].replace("", "(empty)").value_counts())