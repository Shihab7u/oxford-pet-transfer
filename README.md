# Oxford-IIIT Pet: transfer-learning comparison with MobileNetV3-Small

Code and results: <https://github.com/Shihab7u/oxford-pet-transfer>

Reproducible PyTorch pipeline comparing three ways of adapting an ImageNet-pretrained
MobileNetV3-Small to the 37 breed classes of the Oxford-IIIT Pet dataset:

1. **Full fine-tuning** — all weights updated.
2. **Classifier-head training** — convolutional features frozen; only the two fully
   connected layers (`classifier[0]`, `classifier[3]`) are trained.
3. **Frozen features + scikit-learn** — 576-d penultimate features from the frozen
   network, classified with logistic regression, a linear SVM, and a random forest.

The official trainval partition is split 80/20 into train/validation (stratified by
breed, seed 42). The official test partition is used only for the final evaluation.

## Environment

The experiments were run on a CPU-only Windows 11 machine (Intel i5-1135G7, 4 cores).

| Package | Version |
|---|---|
| Python | 3.13.9 |
| PyTorch | 2.15.0+cpu |
| TorchVision | 0.29.2+cpu |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| Matplotlib | 3.11.2 |
| Pillow | 13.0.0 |

Create the environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a CUDA machine, install the matching `torch`/`torchvision` wheels instead of the
CPU index; the code selects the GPU automatically.

## Dataset

**Oxford-IIIT Pet** (Parkhi, Vedaldi, Zisserman & Jawahar, CVPR 2012), 37 cat and dog
breeds, ~200 images per breed.

- Official page: <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Files: `images.tar.gz` (7,390 JPEG images) and `annotations.tar.gz`
  (`trainval.txt`, `test.txt`, `list.txt`, trimaps, XML head boxes).
- License / terms: the dataset is released **for research purposes only**. The images
  come from original websites and remain subject to the terms of use of those sites;
  see the dataset README and the CVPR 2012 paper. The data is therefore **not
  redistributed** in this repository.

Download and extract into a directory outside the repository, e.g.:

```powershell
mkdir C:\data\oxford-iiit-pet
tar -xzf images.tar.gz -C C:\data\oxford-iiit-pet
tar -xzf annotations.tar.gz -C C:\data\oxford-iiit-pet
$env:PET_DATA_ROOT = "C:\data\oxford-iiit-pet"
```

Expected layout:

```
oxford-iiit-pet/
  annotations/{trainval.txt,test.txt,list.txt,trimaps/,xmls/}
  images/<name>.jpg
```

`src/pet_data.py` builds the manifests once and writes
`data/splits/{manifest,train,val,test}.csv` (filename + 0-based label), so all
approaches use identical partitions. The split is deterministic (seed 42) and the
files are committed under `data/splits/`.

| Split | Images |
|---|---|
| Train | 2,944 |
| Validation | 736 |
| Test (official) | 3,669 |
| Total used | 7,349 |

(7,390 JPEGs are present. All 7,349 images referenced by the official lists are used;
the remaining 41 files are not in the lists and are ignored.)

## Preprocessing and augmentation

- Training: `RandomResizedCrop(224, scale=(0.65, 1.0))`, random horizontal flip,
  mild colour jitter, `ToTensor`, ImageNet normalization
  (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- Validation/test: resize shorter side to 256, centre crop 224, same normalization.
- Augmentation is applied **only** to the training set.
- No class balancing: the dataset is nearly balanced (≈79–80 training images per
  breed), so plain cross-entropy is used and accuracy/macro-F1 are appropriate.

## Training protocol

Both neural approaches use AdamW, batch size 32, cross-entropy with label smoothing
0.1, one epoch of linear warm-up followed by cosine decay, and checkpoint selection by
**validation macro-F1** (test data is never used for model selection).

| | Full fine-tuning | Classifier head |
|---|---|---|
| Epochs | 12 | 15 |
| Learning rate | 3e-4 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| Trainable params | 1,555,781 | 628,773 |
| Precision | CPU bfloat16 autocast | CPU bfloat16 autocast |

The classifier head consists of `Linear(576, 1024)`, Hardswish, Dropout(0.2),
`Linear(1024, 37)`. In head-only mode both linear layers are trainable and everything
else is frozen; the frozen backbone is kept in evaluation mode so its BatchNorm running
statistics are not updated.

For frozen features, the 576-d output of `features + avgpool` is extracted for every
split. `StandardScaler` is fit on **training features only**, then logistic regression
(C grid `1e-3 … 1`), linear SVM (same grid) and a 500-tree random forest are trained.
The configuration with the best validation macro-F1 is evaluated once on test.

## Reproduce

```powershell
$env:PET_DATA_ROOT = "C:\data\oxford-iiit-pet"

# 1) full fine-tuning (~37 min on the 4-core CPU used here)
python src/train_nn.py --mode full --epochs 12 --lr 3e-4 --amp bf16

# 2) classifier-head training (~14 min)
python src/train_nn.py --mode head --epochs 15 --lr 1e-3 --amp bf16

# 3) frozen features + scikit-learn classifiers
python src/extract_features.py
python src/train_sklearn.py

# 4) final test evaluation (once per neural checkpoint)
python src/evaluate_nn.py --tags full head

# 5) paired McNemar comparisons of the saved test predictions
python src/compare_models.py

# 6) tables, confusion matrices, prediction examples, report assets
python src/make_report_assets.py

# 7) compile the report (requires a LaTeX installation)
cd report
pdflatex report.tex
pdflatex report.tex
```

Or run everything with `scripts/run_all.ps1`.

## Reproducibility notes

- `seed_everything(42)` seeds Python, NumPy and PyTorch, sets `PYTHONHASHSEED`, and
  enables deterministic cuDNN/`torch.use_deterministic_algorithms` (warn-only).
- The train/validation split is persisted in `data/splits/manifest.csv`; re-running
  uses the saved file, so the partition cannot drift.
- Remaining sources of variation: CPU/GPU kernel non-determinism in some convolutions
  and bfloat16 autocast, thread scheduling, and library version differences. Small
  differences in test scores (≲1 pp) should not be treated as meaningful.
- Each selected checkpoint is saved to `results/checkpoints/{full,head}.pt` with its
  epoch and validation metrics. Neural metrics are in `results/metrics/*.json`, test
  predictions in `results/predictions/*.csv`, and scikit-learn configurations in
  `results/metrics/sklearn_*.json`.

## Repository layout

```
src/
  pet_data.py            dataset manifests, stratified split, transforms
  models.py              MobileNetV3-Small construction, freezing, feature extractor
  train_nn.py            full fine-tuning / head-only training
  extract_features.py    frozen 576-d feature extraction
  train_sklearn.py       scaling, hyper-parameter search, sklearn classifiers
  evaluate_nn.py         one-shot test evaluation of neural checkpoints
  compare_models.py      paired McNemar tests on the saved test predictions
  make_report_assets.py  tables, confusion matrices, curves, example predictions
  utils.py               seeding, metrics, JSON helpers
data/splits/             committed train/val/test manifests (seed 42)
results/
  checkpoints/           best neural checkpoints (by validation macro-F1)
  metrics/               training histories and test metrics
  predictions/           per-image test predictions with confidences
  tables/                generated CSV/LaTeX tables
  figures/               generated PDF/PNG figures
report/                  LaTeX report and compiled PDF
scripts/run_all.ps1      end-to-end reproduction script
```

## Results

| Approach | Trainable params | Time (min) | Test acc. | Macro-F1 |
|---|---|---|---|---|
| Full fine-tuning | 1,555,781 | 37.1 | 84.8 | 84.6 |
| Classifier head only | 628,773 | 13.8 | **87.0** | **86.9** |
| Logistic regression (frozen) | 21,349 | 2.1 | 85.4 | 85.3 |
| Linear SVM (frozen) | 21,349 | 2.3 | 83.9 | 83.6 |
| Random forest (frozen) | 21,349 | 2.2 | 83.0 | 82.5 |

Training only the classifier head on frozen features is the most accurate strategy
(exact McNemar tests: p < 0.001 against both full fine-tuning and logistic regression)
and is cheaper than full fine-tuning. Logistic regression on the same features is
1.6 points behind but costs about seven times less. See
`results/tables/results.csv`, `results/tables/mcnemar.csv`, and the report
(`report/report.pdf`).
