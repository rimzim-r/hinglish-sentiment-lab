# Hinglish Sentiment Lab

Sentiment classification for Hindi-English code-mixed ("Hinglish") YouTube comments,
benchmarked against off-the-shelf tools, with a PySpark pipeline for scoring at scale.

## Question
Off-the-shelf sentiment tools are built for English. How much better does a classifier
trained on Hinglish comments do, and does it carry over to a brand-campaign video?

**Short answer:** on held-out videos from the same dataset, models trained on Hinglish comments
beat the off-the-shelf tools by a wide margin (macro F1 0.70 vs 0.41 to 0.54). Whether that
carries over to a brand campaign I could not establish, because my hand-labeled campaign sample
turned out to be unreliable (see "Campaign evaluation").

## Data
- **Training/test data:** the public Hugging Face dataset `shae2977/hinglish-youtube-sentiments-dataset`
  (3,190 YouTube comments from 21 videos; labels Positive / Negative / Neutral; 3,180 comments after
  removing duplicates and empty ones). Check the dataset card for its terms.
- **Deviation from the original plan:** the SemEval-2020 Task 9 (SentiMix) dataset was not used,
  because I could not confirm a direct public download. It could be added as a second dataset.
- **Campaign data:** top-level comments of one Indian Diwali ad video, collected with the YouTube
  Data API (256 comments; about 11% contain Devanagari). A Sony spec commercial was tried first and
  dropped because its comments contained no Hinglish. Raw comments are not committed to this repo.

## Method
1. **Cleaning** (`src/cleaning.py`): lowercasing, URL and mention removal, emoji turned into word
   tokens, repeated-letter collapsing, and a small dictionary mapping common spelling variants to one
   form (e.g. `nhi` to `nahi`). This is light normalisation, not a full transliteration model.
2. **Split by video** (`src/split.py`): train = 2,861 comments, test = 319 comments from 5 unseen
   videos. Splitting by video avoids near-duplicate comments leaking between train and test.
3. **Models:** VADER and `cardiffnlp/twitter-xlm-roberta-base-sentiment` used as-is on raw text;
   TF-IDF (word 1-2-grams + character 2-5-grams) with class-weighted logistic regression, with the
   regularisation strength chosen by video-grouped cross-validation on the training set only;
   `distilbert-base-multilingual-cased` fine-tuned for 3 epochs with class weights.
4. **Metric:** macro F1 (classes are imbalanced: about 45% Negative, 31% Positive, 24% Neutral),
   with 95% bootstrap confidence intervals.

## Results on the held-out test videos (319 comments)

| Model | Macro F1 | 95% CI |
|---|---|---|
| VADER (off the shelf) | 0.410 | 0.362 to 0.454 |
| XLM-R Twitter (off the shelf) | 0.535 | 0.479 to 0.588 |
| DistilBERT multilingual (fine-tuned) | 0.650 | 0.598 to 0.706 |
| **TF-IDF + logistic regression** | **0.699** | 0.650 to 0.749 |

- The models trained on this data clearly beat the off-the-shelf tools. Part of that gap is that they
  learned this dataset's labelling style, which the off-the-shelf tools never saw.
- TF-IDF and fine-tuned DistilBERT are statistically indistinguishable on this test set.
- Grouped 5-fold cross-validation on the training videos gives a lower and more variable TF-IDF score
  (mean macro F1 0.584, std 0.031 across folds), so 0.699 is partly a favourable split.
- The test set is small (319 comments from 5 videos, one of which is 65% of it), so differences of a
  few points should not be over-read.

### Cleaning ablation (grouped CV on train only; the test set was not used)

| Variant | Mean macro F1 |
|---|---|
| Emoji as tokens, spelling variants (final) | 0.584 |
| Emoji dropped | 0.520 |
| Keep `?` as a token | 0.585 |
| No spelling variants | 0.579 |

Emoji handling is the one clear win (+0.06). The `?` token and spelling dictionary are within noise.

### Error analysis
From reading the most confident errors (informal review, not systematic counts):
- `nahi` is a strong Negative cue, but it also appears in neutral questions and observations
  ("mirchi nahi dali?").
- The same emoji carries different sentiment in different comments (a crying emoji appears in
  both Positive and Negative comments).
- Sarcasm, mixed praise and complaint, and some inconsistent labels (similar comments labelled
  differently).
- Some top-weighted words are topic cues (`yogi`, `ai`, `bhai`), not sentiment, so the model
  partly learns its training videos.

## Campaign evaluation (inconclusive)

I hand-labeled 119 randomly sampled comments from the Diwali ad video, then relabeled them blind
(without seeing my first labels or any model output). The second pass covered 67 comments, of
which 16 were excluded as unjudgeable (Malayalam text, bare emoji, gibberish, bare links).

My two passes agreed on only 23.5% of the 51 remaining comments (Cohen's kappa -0.15), which is
no better than chance, and the disagreements show no systematic pattern such as swapped labels.
The labels are therefore not reliable enough to measure model accuracy on this campaign, so I do
not report campaign scores as a result. For transparency: macro F1 of VADER, XLM-R and TF-IDF
against the first pass was 0.31 to 0.38, and against the second pass 0.23 to 0.35, with confidence
intervals too wide to rank anything. (The fine-tuned DistilBERT was not scored on the campaign.)

What this does show: much of an ad's comment section has no clear sentiment to label
(greetings, devotional remarks, emoji, other languages). A usable campaign evaluation would need a
written labelling guide, an "unjudgeable" class, more than one annotator, and a larger sample.

## PySpark pipeline (`src/spark_pipeline.py`)
Cleaning and scoring run as a pandas UDF in Spark (local mode, all cores).

| Rows | Spark | Plain pandas (1 process) |
|---|---|---|
| 22,260 | 1,208 comments/s | 5,256 comments/s |
| 502,440 | 18,702 comments/s | 5,501 comments/s |

The rows are the 3,180 real comments **replicated**, so this measures throughput, not new data.
Spark is slower on small inputs because of startup cost, and about 3.4 times faster at 500K rows on
this laptop (it uses all cores; the pandas baseline uses one). Each size was timed once.
The pipeline also produces a predicted sentiment mix for the 256 campaign comments (43% Positive,
42% Neutral, 15% Negative); this is unvalidated model output and should not be read as audience opinion.

## Limitations
- Small test set (319 comments, 5 videos), so differences of a few points are noise.
- Labels in the public dataset are noisy and come from one source.
- The campaign evaluation failed to produce a reliable result (see above).
- Emotion classification and full transliteration normalisation (from the original idea) are not
  implemented, and the SemEval-2020 data was not used.
- Test errors were read during error analysis; no model or cleaning change was made afterwards.

## Reproduce
```
pip install -r requirements.txt
python -m src.data && python -m src.prepare && python -m src.split
python -m src.tfidf_model && python -m src.transformer_model
python -m src.baselines && python -m src.compare && python -m src.ablation
# campaign (needs YOUTUBE_API_KEY in a local .env file)
python -m src.youtube_collect "<video url>"
python -m src.make_label_sample data/raw/campaign_<id>.csv
python -m src.label_cli && python -m src.score_campaign
python -m src.relabel_cli && python -m src.score_relabel
python -m src.spark_pipeline 500000 data/raw/campaign_<id>.csv
```
Requires Python 3 and Java 17 (for PySpark).