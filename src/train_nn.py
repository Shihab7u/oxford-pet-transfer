"""Train MobileNetV3-Small on Oxford-IIIT Pet with full fine-tuning or head-only training.

Examples
--------
python src/train_nn.py --mode full --epochs 12 --lr 3e-4
python src/train_nn.py --mode head --epochs 15 --lr 1e-3

Model selection uses the validation split only (best macro-F1 checkpoint).  The
official test split is evaluated separately by ``evaluate_nn.py``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model, trainable_parameter_names
from pet_data import PetDataset, build_transforms, class_names, make_datasets, resolve_data_root
from utils import (
    CKPT_DIR,
    METRICS_DIR,
    compute_metrics,
    count_total,
    count_trainable,
    save_json,
    seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["full", "head"], required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["off", "bf16"], default="off")
    p.add_argument("--tag", default=None, help="output name; defaults to mode")
    return p.parse_args()


def evaluate(model, loader, criterion, device, names, amp_dtype=None):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.inference_mode():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cpu", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(loss.item() * labels.size(0))
            preds.append(logits.argmax(1).cpu())
            targets.append(labels.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    metrics = compute_metrics(y_true, y_pred, names)
    metrics["loss"] = float(sum(losses) / len(y_true))
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    tag = args.tag or args.mode
    epochs = args.epochs or (12 if args.mode == "full" else 15)
    lr = args.lr or (3e-4 if args.mode == "full" else 1e-3)

    train_ds, val_ds, test_ds, manifest = make_datasets(args.data_root, args.image_size)
    names = class_names(args.data_root)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    # Training-set loader without augmentation, used for the train/test gap.
    image_dir = resolve_data_root(args.data_root) / "images"
    train_eval_ds = PetDataset(
        manifest[manifest.split == "train"], image_dir, build_transforms(False, args.image_size)
    )
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.mode).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay
    )
    warmup_epochs = 1
    steps_per_epoch = len(train_loader)

    def lr_lambda(step: int) -> float:
        total = epochs * steps_per_epoch
        warmup = warmup_epochs * steps_per_epoch
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    info = {
        "tag": tag,
        "mode": args.mode,
        "args": vars(args),
        "trainable_parameters": count_trainable(model),
        "total_parameters": count_total(model),
        "trainable_layer_names": trainable_parameter_names(model),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_names": names,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "optimizer": "AdamW",
            "scheduler": "1-epoch linear warmup + cosine decay",
            "loss": "cross-entropy (label smoothing 0.1)",
            "model_selection": "best validation macro-F1",
            "amp": args.amp,
        },
    }
    print(json.dumps({k: info[k] for k in ("tag", "mode", "trainable_parameters", "total_parameters")}))

    history = []
    best_f1 = -1.0
    best_epoch = -1
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss, seen, correct = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

        val_metrics = evaluate(model, val_loader, criterion, device, names, amp_dtype)
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy_augmented": correct / seen,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(entry)
        print(
            f"epoch {epoch:02d}/{epochs} | train loss {entry['train_loss']:.4f} acc {entry['train_accuracy_augmented']:.4f} "
            f"| val loss {entry['val_loss']:.4f} acc {entry['val_accuracy']:.4f} macroF1 {entry['val_macro_f1']:.4f} "
            f"| {entry['seconds']:.1f}s",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "class_names": names,
                    "epoch": epoch,
                    "val_metrics": {k: v for k, v in val_metrics.items() if k != "per_class"},
                    "args": vars(args),
                },
                CKPT_DIR / f"{tag}.pt",
            )

    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(CKPT_DIR / f"{tag}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    train_metrics = evaluate(model, train_eval_loader, criterion, device, names, amp_dtype)
    val_metrics = evaluate(model, val_loader, criterion, device, names, amp_dtype)

    info.update(
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "training_seconds": training_seconds,
            "train_eval_accuracy": train_metrics["accuracy"],
            "train_eval_macro_f1": train_metrics["macro_f1"],
            "final_val_accuracy": val_metrics["accuracy"],
            "final_val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": [h["seconds"] for h in history],
        }
    )
    save_json(info, METRICS_DIR / f"{tag}_training.json")
    print(
        f"done: best epoch {best_epoch} val macroF1 {best_f1:.4f} | "
        f"train-eval acc {train_metrics['accuracy']:.4f} | {training_seconds/60:.1f} min total",
        flush=True,
    )


if __name__ == "__main__":
    main()
