"""Streamlit demo for the Hinglish Sentiment Lab (TF-IDF + logistic regression)."""
import pandas as pd
import streamlit as st

from src.cleaning import clean_text
from src.predict import MODEL_PATH, predict

st.set_page_config(page_title="Hinglish Sentiment Lab", page_icon="💬", layout="centered")

st.title("Hinglish Sentiment Lab")
st.caption(
    "TF-IDF + logistic regression trained on 2,861 Hindi-English code-mixed YouTube comments "
    "(0.699 macro F1 on 5 held-out videos vs 0.410 for off-the-shelf VADER and 0.535 for XLM-R)."
)

st.info(
    "**Scope & limitations:** Trained on YouTube creator and review comments. "
    "Not validated on brand-campaign ad comments, and the probabilities below are raw "
    "logistic-regression scores rather than calibrated confidence.",
)

DEFAULT_COMMENTS = "\n".join([
    "bahut mast video hai bhai 🔥",
    "yaar ye video toh bakwaas tha 😂",
    "part 2 kab aayega bro?",
    "bohot badiya explanation sir, concept clear ho gaya",
])

raw_input = st.text_area(
    "Enter Hinglish comments (one per line, up to 100):",
    value=DEFAULT_COMMENTS,
    height=140,
)

if st.button("Classify comments", type="primary"):
    lines = [line.strip() for line in raw_input.splitlines() if line.strip()]
    if not lines:
        st.warning("Please enter at least one comment.")
    elif len(lines) > 100:
        st.error("Please enter at most 100 comments at a time.")
    elif any(len(t) > 2000 for t in lines):
        st.error("Each comment must be at most 2,000 characters.")
    elif not MODEL_PATH.exists():
        st.error("Model file not found (`models/tfidf_lr.joblib`). Run `python -m src.tfidf_model` first.")
    else:
        results = predict(lines)
        rows = []
        for r in results:
            probs = r["probabilities"]
            rows.append({
                "Comment": r["text"],
                "Cleaned text": clean_text(r["text"]),
                "Predicted": r["label"],
                "P(Positive)": probs.get("Positive", 0.0),
                "P(Neutral)": probs.get("Neutral", 0.0),
                "P(Negative)": probs.get("Negative", 0.0),
            })

        df = pd.DataFrame(rows)

        counts = df["Predicted"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", int(counts.get("Positive", 0)))
        c2.metric("Neutral", int(counts.get("Neutral", 0)))
        c3.metric("Negative", int(counts.get("Negative", 0)))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(Positive)": st.column_config.NumberColumn(format="%.3f"),
                "P(Neutral)": st.column_config.NumberColumn(format="%.3f"),
                "P(Negative)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
