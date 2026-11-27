# End-to-end reproduction (PowerShell).
# Usage: .\scripts\run_all.ps1 -DataRoot C:\data\oxford-iiit-pet
param(
    [string]$DataRoot = $env:PET_DATA_ROOT,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
if (-not $DataRoot) { throw "Set PET_DATA_ROOT or pass -DataRoot" }
$env:PET_DATA_ROOT = $DataRoot
$env:PYTHONPATH = "src"

& $Python src/train_nn.py --mode full --epochs 12 --lr 3e-4 --amp bf16
& $Python src/train_nn.py --mode head --epochs 15 --lr 1e-3 --amp bf16
& $Python src/extract_features.py
& $Python src/train_sklearn.py
& $Python src/evaluate_nn.py --tags full head
& $Python src/compare_models.py
& $Python src/make_report_assets.py

Push-Location report
pdflatex report.tex
pdflatex report.tex
Pop-Location
