"""Paired comparison of saved test predictions (exact McNemar test).

All approaches predict the same test images, so their errors can be compared
pairwise on identical samples. The exact McNemar test asks whether the number of
images classified correctly by exactly one of the two models is asymmetric.

Usage: python src/compare_models.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from utils import PRED_DIR, TABLES_DIR

TAGS = [
    ("full", "Full fine-tuning"),
    ("head", "Classifier head only"),
    ("sklearn_logreg", "Logistic regression (frozen)"),
    ("sklearn_linear_svm", "Linear SVM (frozen)"),
    ("sklearn_random_forest", "Random forest (frozen)"),
]


def load(tag: str) -> pd.DataFrame:
    path = Path(PRED_DIR) / f"{tag}_test_predictions.csv"
    df = pd.read_csv(path).sort_values("filename").reset_index(drop=True)
    return df


def main() -> None:
    frames = {tag: load(tag) for tag, _ in TAGS}
    reference = frames[TAGS[0][0]]
    for tag, df in frames.items():
        if not (df.filename.values == reference.filename.values).all():
            raise SystemExit(f"{tag}: prediction files are not aligned on filename")
        if not (df.label.values == reference.label.values).all():
            raise SystemExit(f"{tag}: label columns disagree")

    labels = dict(TAGS)
    rows = []
    for (a, _), (b, _) in itertools.combinations(TAGS, 2):
        ca = frames[a].label.values == frames[a].prediction.values
        cb = frames[b].label.values == frames[b].prediction.values
        only_a = int((ca & ~cb).sum())
        only_b = int((~ca & cb).sum())
        p = binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.0
        rows.append(
            {
                "model_a": labels[a],
                "model_b": labels[b],
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "mcnemar_p": p,
            }
        )

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "mcnemar.csv", index=False)
    with pd.option_context("display.width", 160):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
