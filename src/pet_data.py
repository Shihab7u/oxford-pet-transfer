"""Dataset indexing, stratified splits and image transforms for Oxford-IIIT Pet.

The official trainval/test partitions are used.  The official trainval partition is
split once into train/validation (80/20, stratified by breed, seed 42) and the
resulting manifests are written to ``data/splits`` so every experiment uses exactly
the same images.

Expected on-disk layout (the layout produced by extracting the official tarballs)::

    <data_root>/annotations/trainval.txt
    <data_root>/annotations/test.txt
    <data_root>/annotations/list.txt
    <data_root>/images/<image_name>.jpg

Set the ``PET_DATA_ROOT`` environment variable or pass ``--data-root``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_ROOT = Path(
    os.environ.get("PET_DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "oxford-iiit-pet")
)

SPLIT_SEED = 42
VAL_FRACTION = 0.2


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not (root / "annotations" / "trainval.txt").exists():
        raise FileNotFoundError(
            f"Oxford-IIIT Pet annotations not found under {root}. "
            "Extract annotations.tar.gz and images.tar.gz there or set PET_DATA_ROOT."
        )
    return root


def class_names(data_root: str | Path | None = None) -> list[str]:
    """Return the 37 breed names indexed by the official class id (1..37)."""
    root = resolve_data_root(data_root)
    mapping: dict[int, str] = {}
    with open(root / "annotations" / "list.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, class_id, _species, _breed_id = line.split()
            class_id = int(class_id)
            if class_id not in mapping:
                mapping[class_id] = re.sub(r"_\d+$", "", name)
    return [mapping[i] for i in sorted(mapping)]


def display_name(raw: str) -> str:
    return raw.replace("_", " ")


def _read_partition(path: Path, image_dir: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, class_id = parts[0], int(parts[1])
            if (image_dir / f"{name}.jpg").exists():
                rows.append({"filename": name, "label": class_id - 1})  # 0-based
    return pd.DataFrame(rows)


def build_splits(data_root: str | Path | None = None, out_dir: str | Path | None = None) -> pd.DataFrame:
    """Create (or load) the train/val/test manifests.  Test stays untouched."""
    root = resolve_data_root(data_root)
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    image_dir = root / "images"
    trainval = _read_partition(root / "annotations" / "trainval.txt", image_dir)
    test = _read_partition(root / "annotations" / "test.txt", image_dir)

    train, val = train_test_split(
        trainval,
        test_size=VAL_FRACTION,
        stratify=trainval["label"],
        random_state=SPLIT_SEED,
    )
    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")

    manifest = pd.concat([train, val, test], ignore_index=True)
    manifest = manifest.sort_values(["split", "filename"]).reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)
    for split in ("train", "val", "test"):
        manifest[manifest.split == split][["filename", "label"]].to_csv(
            out_dir / f"{split}.csv", index=False
        )
    return manifest


def split_counts(manifest: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    counts = (
        manifest.pivot_table(index="label", columns="split", values="filename", aggfunc="count")
        .reindex(range(len(names)))
        .fillna(0)
        .astype(int)
        .rename(index=lambda i: display_name(names[i]))
        .rename_axis("breed")
        .reset_index()
    )
    for split in ("train", "val", "test"):
        if split not in counts:
            counts[split] = 0
    counts["total"] = counts[["train", "val", "test"]].sum(axis=1)
    return counts[["breed", "train", "val", "test", "total"]]


def build_transforms(train: bool, image_size: int = 224):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # Standard ImageNet evaluation protocol: shorter side 256, centre crop 224.
    resize_to = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_to, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PetDataset(Dataset):
    """Image dataset driven by a manifest dataframe (filename, label, split)."""

    def __init__(self, manifest: pd.DataFrame, image_dir: str | Path, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img = Image.open(self.image_dir / f"{row.filename}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row.label), row.filename


def make_datasets(data_root=None, image_size: int = 224):
    root = resolve_data_root(data_root)
    manifest = build_splits(root)
    image_dir = root / "images"
    return (
        PetDataset(manifest[manifest.split == "train"], image_dir, build_transforms(True, image_size)),
        PetDataset(manifest[manifest.split == "val"], image_dir, build_transforms(False, image_size)),
        PetDataset(manifest[manifest.split == "test"], image_dir, build_transforms(False, image_size)),
        manifest,
    )
