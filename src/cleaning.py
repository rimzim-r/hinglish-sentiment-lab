import re
import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # sooooo -> soo
SPACE_RE = re.compile(r"\s+")

# Spelling variants of common Hinglish words -> one canonical form.
# Start small; we extend it after error analysis.
VARIANTS = {
    "bakwas": "bakwaas", "bkwas": "bakwaas", "bakwass": "bakwaas",
    "nahin": "nahi", "nhi": "nahi", "nai": "nahi", "nahii": "nahi",
    "acha": "accha", "achha": "accha", "acchha": "accha", "achchha": "accha",
    "kia": "kya", "kyaa": "kya",
    "hain": "hai", "he": "hai",
    "bohot": "bahut", "bohut": "bahut", "bhut": "bahut", "boht": "bahut",
    "mazza": "maza", "mazaa": "maza", "majaa": "maza", "maja": "maza",
    "bekar": "bekaar",
    "pyar": "pyaar",
    "bhaiya": "bhai", "bro": "bhai",
    "ye": "yeh", "yah": "yeh",
    "kyu": "kyun", "kyon": "kyun",
    "thik": "theek", "tik": "theek",
    "sai": "sahi",
}


def handle_emoji(text: str) -> str:
    """Turn each emoji into a word token, e.g. 😂 -> ' emo_face_with_tears_of_joy '."""
    return emoji.replace_emoji(
        text, replace=lambda ch, data: f" emo_{emoji.demojize(ch).strip(':')} "
    )


def normalise_tokens(text: str) -> str:
    return " ".join(VARIANTS.get(tok, tok) for tok in text.split())


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = handle_emoji(text)
    text = REPEAT_RE.sub(r"\1\1", text)           # collapse 3+ repeats to 2
    text = re.sub(r"[^\w\s]", " ", text)          # drop punctuation (emoji tokens are already words)
    text = SPACE_RE.sub(" ", text).strip()
    return normalise_tokens(text)


if __name__ == "__main__":
    for t in ["yaar ye ad toh bakwaas tha 😂", "Soooo gooood!!! nhi pasand https://x.co @brand"]:
        print(t, "->", clean_text(t))