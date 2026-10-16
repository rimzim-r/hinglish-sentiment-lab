import sys

import pandas as pd

N = 150

if __name__ == "__main__":
    paths = sys.argv[1:]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["comment"] = df["comment"].astype(str).str.strip()
    df = df[df["comment"].str.len() > 0].drop_duplicates(subset=["video_id", "comment"])
    print(f"{len(df)} usable comments from {len(paths)} file(s)")
    df.to_csv("data/processed/campaign_all.csv", index=False, encoding="utf-8-sig")

    sample = df.sample(min(N, len(df)), random_state=0)[["comment_id", "video_id", "comment"]].copy()
    sample["label"] = ""
    sample.to_csv("data/processed/campaign_label_sample.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(sample)} comments to data/processed/campaign_label_sample.csv")