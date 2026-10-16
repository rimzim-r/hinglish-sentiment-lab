import re
import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # sooooo -> soo
SPACE_RE = re.compile(r"\s+")

# Spelling variants of common Hinglish words -> one canonical form.
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


def clean_text(text, emoji_mode="token", keep_qmark=False, use_variants=True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if emoji_mode == "token":
        text = handle_emoji(text)
    elif emoji_mode == "drop":
        text = emoji.replace_emoji(text, replace=" ")
    if keep_qmark:
        text = text.replace("?", " qmark ")
    text = REPEAT_RE.sub(r"\1\1", text)           # collapse 3+ repeats to 2
    text = re.sub(r"[^\w\s]", " ", text)          # drop punctuation
    text = SPACE_RE.sub(" ", text).strip()
    return normalise_tokens(text) if use_variants else text


if __name__ == "__main__":
    for t in ["yaar ye ad toh bakwaas tha 😂", "Soooo gooood!!! nhi pasand https://x.co @brand",
              "Pyaz koun dalega?"]:
        print(t, "->", clean_text(t))
        print("   with ? kept ->", clean_text(t, keep_qmark=True))