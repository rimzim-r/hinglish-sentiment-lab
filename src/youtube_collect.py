import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def video_id_from(url_or_id: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    return m.group(1) if m else url_or_id


def fetch_comments(video_id: str, max_comments: int = 5000) -> pd.DataFrame:
    yt = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])
    rows, token = [], None
    while len(rows) < max_comments:
        try:
            resp = yt.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=token, textFormat="plainText", order="time",
            ).execute()
        except HttpError as e:
            print("API error:", e)
            break
        for item in resp["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "comment": s["textDisplay"],
                "likes": s["likeCount"],
                "published_at": s["publishedAt"],
            })
        token = resp.get("nextPageToken")
        print(f"collected {len(rows)} comments", end="\r")
        if not token:
            break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    vid = video_id_from(sys.argv[1])
    df = fetch_comments(vid)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out = f"data/raw/campaign_{vid}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} comments to {out}")
    print(df.head())