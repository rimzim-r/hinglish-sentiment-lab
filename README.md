# Hinglish Sentiment Lab

Sentiment classification for Hindi-English code-mixed ("Hinglish") YouTube comments,
benchmarked against off-the-shelf tools, with a PySpark scoring pipeline.

## Key result
On 5 held-out videos (319 comments), models trained on Hinglish comments beat off-the-shelf
sentiment tools by a wide margin:

| Model | Macro F1 | 95% CI |
|---|---|---|
| VADER (off the shelf) | 0.410 | 0.362 to 0.454 |
| XLM-R Twitter (off the shelf) | 0.535 | 0.479 to 0.588 |
| DistilBERT multilingual (fine-tuned) | 0.650 | 0.598 to 0.706 |
| **TF-IDF + logistic regression** | **0.699** | 0.650 to 0.749 |

TF-IDF and fine-tuned DistilBERT are statistically indistinguishable. The test set is small, and
grouped cross-validation gives a lower TF-IDF score (0.584), so 0.699 is partly a favourable split.

## What I built
- **Cleaning:** emoji turned into word tokens, spelling-variant normalisation, repeated-letter
  collapsing. Dropping emoji cost 0.06 macro F1 in cross-validation; the spelling dictionary
  made no measurable difference.
- **Video-grouped train/test split**, so near-duplicate comments don't leak between sets.
- **Four-model comparison** with bootstrap confidence intervals.
- **YouTube Data API collector** and a **PySpark pipeline**: about 18,700 comments/s on 500K
  rows vs about 5,500/s for single-process pandas on my laptop (the rows are replicated
  comments, so this measures throughput only).

## What did not work
I tried to check the model on a real brand ad (a Diwali campaign video, 256 comments). My
hand labels were not reliable: relabeling blind, I agreed with myself on only 23.5% of comments
(Cohen's kappa -0.15), so I do not report campaign accuracy. Many ad comments have no judgeable
opinion (greetings, emoji, other languages). Whether the model carries over to brand campaigns
remains untested.

## Limitations
Small, noisy dataset from one source; one annotator; SemEval-2020 data, emotion labels and full
transliteration normalisation from the original plan were not implemented.

More detail (ablation, error analysis, benchmarks, campaign check): [docs/DETAILS.md](docs/DETAILS.md)

## Run it
```
pip install -r requirements.txt
python -m src.data && python -m src.prepare && python -m src.split
python -m src.tfidf_model && python -m src.baselines && python -m src.compare
```
Needs Python 3; Java 17 for the PySpark step. The YouTube collector needs a `YOUTUBE_API_KEY`
in a local `.env` file.