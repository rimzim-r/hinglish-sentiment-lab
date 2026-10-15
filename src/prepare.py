from pathlib import Path
import pandas as pd
from src.cleaning import clean_text

RAW = "data/raw/youtube_hinglish.csv"
OUT = "data/processed/youtube_hinglish_clean.csv"


def prepare() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["clean"] = df["comment"].apply(clean_text)
    before = len(df)
    df = df[df["clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["video_id", "clean"])
    print(f"rows: {before} -> {len(df)}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = prepare()
    print(df[["comment", "clean"]].sample(10, random_state=2).to_string())
