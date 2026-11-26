"""Extract frozen MobileNetV3-Small penultimate features (576-d) for train/val/test."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import FeatureExtractor, build_model
from pet_data import make_datasets
from utils import FEATURES_DIR, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def extract(extractor, loader, device):
    feats, labels, files = [], [], []
    for images, targets, names in loader:
        feats.append(extractor(images.to(device)).cpu().numpy().astype(np.float32))
        labels.append(targets.numpy())
        files.extend(names)
    return np.concatenate(feats), np.concatenate(labels), np.array(files)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, _ = make_datasets(args.data_root, args.image_size)
    model = build_model("frozen").to(device).eval()
    extractor = FeatureExtractor(model).to(device).eval()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    timing = {}
    arrays = {}
    for split, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        start = time.perf_counter()
        feats, labels, files = extract(extractor, loader, device)
        timing[split] = time.perf_counter() - start
        arrays[f"X_{split}"], arrays[f"y_{split}"], arrays[f"files_{split}"] = feats, labels, files
        print(f"{split}: {feats.shape} in {timing[split]:.1f}s", flush=True)

    np.savez_compressed(FEATURES_DIR / "mobilenetv3small_features.npz", **arrays)
    save_json(
        {
            "feature_dim": int(arrays["X_train"].shape[1]),
            "feature_layer": "output of model.features + global average pooling (penultimate, 576-d)",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT (ImageNet-1K), frozen",
            "extraction_seconds": timing,
            "total_seconds": sum(timing.values()),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "torch": torch.__version__,
        },
        FEATURES_DIR / "extraction_meta.json",
    )


if __name__ == "__main__":
    main()
