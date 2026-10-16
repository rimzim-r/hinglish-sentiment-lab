import re
import sys

import pandas as pd

MARKERS = {"hai", "hain", "nahi", "nhi", "bhai", "yaar", "kya", "bahut", "bohot",
           "accha", "acha", "mast", "yeh", "toh", "aur", "mein", "bakwaas", "sahi",
           "kaise", "mujhe", "aap", "hum", "lekin", "bilkul", "wala", "wali",
           "hota", "karo", "kare", "nai"}
DEVA = re.compile(r"[\u0900-\u097F]")
WORD = re.compile(r"[a-z]+")


def other_script(t):
    return any(ch.isalpha() and ord(ch) > 0x24F and not (0x900 <= ord(ch) <= 0x97F)
               for ch in t)


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    text = df["comment"].astype(str)
    deva = text.apply(lambda t: bool(DEVA.search(t)))
    hing = text.str.lower().apply(lambda t: bool(MARKERS & set(WORD.findall(t))))
    other = text.apply(other_script)
    print(f"comments            : {len(df)}")
    print(f"Devanagari script   : {deva.mean():.1%}")
    print(f"Hinglish word hints : {hing.mean():.1%}")
    print(f"other scripts       : {other.mean():.1%}")
    print(f"very short (<=2 words): {(text.str.split().str.len() <= 2).mean():.1%}")
    print("\nSample comments with Hinglish hints:")
    print(text[hing].head(10).to_string())