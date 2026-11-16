from src.cleaning import clean_text


def test_emoji_becomes_token():
    assert "emo_face_with_tears_of_joy" in clean_text("ye ad mast tha 😂")


def test_spelling_variant_is_normalised():
    assert clean_text("nhi pasand") == "nahi pasand"


def test_urls_and_mentions_removed():
    out = clean_text("check https://x.co @brand nice")
    assert "http" not in out and "brand" not in out


def test_repeated_letters_collapsed():
    assert clean_text("soooo") == "soo"


def test_non_string_input_returns_empty():
    assert clean_text(None) == ""