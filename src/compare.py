import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PRED = "data/processed/test_predictions.csv"
N_BOOT = 1000


def boot_ci(y, p, rng):
    n = len(y)
    scores = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y[idx], p[idx], average="macro"))
    return np.percentile(scores, [2.5, 97.5])


if __name__ == "__main__":
    df = pd.read_csv(PRED)
    y = df["sentiment"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for col in [c for c in df.columns if c.startswith("pred_")]:
        p = df[col].to_numpy()
        lo, hi = boot_ci(y, p, rng)
        rows.append((col.replace("pred_", ""), f1_score(y, p, average="macro"), lo, hi))
    table = pd.DataFrame(rows, columns=["model", "f1", "lo", "hi"]).sort_values("f1", ascending=False)
    print(table.round(3).to_string(index=False))

    lines = ["| Model | Macro F1 | 95% bootstrap CI |", "|---|---|---|"]
    lines += [f"| {r.model} | {r.f1:.3f} | {r.lo:.3f} to {r.hi:.3f} |" for r in table.itertuples()]
    open("reports/results_table.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\nSaved reports/results_table.md")