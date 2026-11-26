"""Evaluate the two neural checkpoints once on the official test split."""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import build_model
from pet_data import class_names, make_datasets
from utils import CKPT_DIR, METRICS_DIR, PRED_DIR, compute_metrics, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["full", "head"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def predict(model, loader, device):
    probs, preds, targets, files = [], [], [], []
    for images, labels, names in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        targets.append(labels)
        files.extend(names)
    return (
        torch.cat(probs).numpy(),
        torch.cat(preds).numpy(),
        torch.cat(targets).numpy(),
        files,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds, _ = make_datasets(args.data_root)
    names = class_names(args.data_root)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for tag in args.tags:
        ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location=device, weights_only=False)
        model = build_model(ckpt["mode"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        start = time.perf_counter()
        probs, preds, targets, files = predict(model, loader, device)
        elapsed = time.perf_counter() - start
        metrics = compute_metrics(targets, preds, names)
        metrics["inference_seconds"] = elapsed
        metrics["checkpoint_epoch"] = ckpt["epoch"]
        metrics["val_macro_f1_at_checkpoint"] = ckpt["val_metrics"]["macro_f1"]
        save_json(metrics, METRICS_DIR / f"{tag}_test.json")

        pd.DataFrame(
            {
                "filename": files,
                "label": targets,
                "prediction": preds,
                "confidence": probs.max(1),
                "true_name": [names[i] for i in targets],
                "pred_name": [names[i] for i in preds],
            }
        ).to_csv(PRED_DIR / f"{tag}_test_predictions.csv", index=False)
        print(f"{tag}: test acc {metrics['accuracy']:.4f} macroF1 {metrics['macro_f1']:.4f} "
              f"({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
