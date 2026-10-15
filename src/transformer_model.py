import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-multilingual-cased"
TAG = MODEL.split("/")[-1]
LABELS = ["Negative", "Neutral", "Positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
TRAIN = "data/processed/train.csv"
TEST_PRED = Path("data/processed/test_predictions.csv")
TEST = TEST_PRED if TEST_PRED.exists() else Path("data/processed/test.csv")
RESULTS = Path("reports/baseline_results.json")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(texts, truncation=True, max_length=96, padding=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights, to handle the label imbalance."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id=L2I)

    # raw comments: the tokenizer already handles emojis and Devanagari
    ytr = train["sentiment"].map(L2I).tolist()
    yte = test["sentiment"].map(L2I).tolist()
    train_ds = TextDataset(train["comment"].astype(str).tolist(), ytr, tok)
    test_ds = TextDataset(test["comment"].astype(str).tolist(), yte, tok)

    counts = np.bincount(ytr, minlength=3)
    weights = torch.tensor(len(ytr) / (3 * counts), dtype=torch.float)

    args = TrainingArguments(
        output_dir="models/tmp_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=50,
        seed=42,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                              class_weights=weights)
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    pred = [LABELS[i] for i in logits.argmax(axis=1)]
    f1 = f1_score(test["sentiment"], pred, average="macro")
    print(f"\nTEST macro F1 ({TAG}): {f1:.3f}")
    print(classification_report(test["sentiment"], pred, zero_division=0))

    # save predictions next to the TF-IDF ones, for error analysis
    test[f"pred_{TAG}"] = pred
    test.to_csv("data/processed/test_predictions.csv", index=False)

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[f"finetuned_{TAG}"] = round(float(f1), 4)
    RESULTS.write_text(json.dumps(results, indent=2))
    print("Saved predictions and results.")