"""Shared helpers: seeding, metrics, JSON I/O, result directories."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CKPT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"
FEATURES_DIR = RESULTS_DIR / "features"


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    Deterministic algorithms are enabled where available.  ``warn_only=True`` keeps
    operations without a deterministic CPU kernel from aborting the run; the
    remaining sources of variation are documented in the README.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def compute_metrics(y_true, y_pred, names: list[str] | None = None) -> dict:
    labels = list(range(len(names))) if names is not None else None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "n": int(len(y_true)),
    }
    if names is not None:
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
        )
        metrics["per_class"] = {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in names
        }
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
