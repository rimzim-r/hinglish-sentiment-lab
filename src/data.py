from pathlib import Path
import pandas as pd
from datasets import load_dataset

RAW_DIR = Path("data/raw")
HF_NAME = "shae2977/hinglish-youtube-sentiments-dataset"


def load_youtube_hinglish(save: bool = True) -> pd.DataFrame:
    """Download the Hinglish YouTube comments dataset and return one DataFrame."""
    ds = load_dataset(HF_NAME)
    frames = [split.to_pandas().assign(split=name) for name, split in ds.items()]
    df = pd.concat(frames, ignore_index=True)
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / "youtube_hinglish.csv", index=False)
    return df


if __name__ == "__main__":
    df = load_youtube_hinglish()
    print(df.shape)
    print(df.head())