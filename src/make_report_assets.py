"""Generate every table and figure used in the report from saved results.

Usage: python src/make_report_assets.py --data-root C:/Users/krony/hw2_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image as mpimg

from pet_data import build_splits, class_names, display_name, resolve_data_root, split_counts
from utils import FEATURES_DIR, FIGURES_DIR, METRICS_DIR, PRED_DIR, RESULTS_DIR, TABLES_DIR, load_json

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def load_all():
    nn = {}
    for tag in ("full", "head"):
        nn[tag] = {
            "train": load_json(METRICS_DIR / f"{tag}_training.json"),
            "test": load_json(METRICS_DIR / f"{tag}_test.json"),
        }
    sklearn = {
        fam: load_json(METRICS_DIR / f"sklearn_{fam}.json")
        for fam in ("logreg", "linear_svm", "random_forest")
    }
    return nn, sklearn


def results_table(nn, sklearn, extraction_seconds, n_classes):
    rows = []
    for tag, label in (("full", "Full fine-tuning"), ("head", "Classifier head only")):
        t, te = nn[tag]["train"], nn[tag]["test"]
        rows.append(
            {
                "approach": label,
                "params": t["trainable_parameters"],
                "time_min": t["training_seconds"] / 60,
                "test_acc": te["accuracy"],
                "test_f1": te["macro_f1"],
                "train_acc": t["train_eval_accuracy"],
                "val_acc": t["final_val_accuracy"],
                "gap": t["train_eval_accuracy"] - te["accuracy"],
            }
        )
    names = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "random_forest": "Random forest"}
    for fam, res in sklearn.items():
        rows.append(
            {
                "approach": f"{names[fam]} (frozen)",
                "params": res["n_features"] * n_classes + n_classes,
                "time_min": (extraction_seconds + res["fit_seconds"]) / 60,
                "test_acc": res["test"]["accuracy"],
                "test_f1": res["test"]["macro_f1"],
                "train_acc": res["train_accuracy"],
                "val_acc": res["val_accuracy"],
                "gap": res["train_accuracy"] - res["test"]["accuracy"],
                "config": res["selected_config"],
            }
        )
    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "results.csv", index=False)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Approach & Trainable & Time & Test & Macro & Train & Gap \\",
        r" & params & (min) & acc.\ (\%) & F1 (\%) & acc.\ (\%) & (pp) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.approach} & {r.params:,} & {r.time_min:.1f} & {pct(r.test_acc)} & "
            f"{pct(r.test_f1)} & {pct(r.train_acc)} & {100 * r.gap:+.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "results_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return df


def dataset_table(manifest, names):
    counts = split_counts(manifest, names)
    counts.to_csv(TABLES_DIR / "dataset_counts.csv", index=False)
    half = int(np.ceil(len(counts) / 2))
    left = counts.iloc[:half].reset_index(drop=True)
    right = counts.iloc[half:].reset_index(drop=True)
    lines = [
        r"\begin{tabular}{lrrrr@{\hskip 12pt}lrrrr}",
        r"\toprule",
        r"Breed & Train & Val & Test & Total & Breed & Train & Val & Test & Total \\",
        r"\midrule",
    ]
    for i in range(half):
        a = left.iloc[i]
        row = f"{tex_escape(a.breed)} & {a.train} & {a.val} & {a.test} & {a.total}"
        if i < len(right):
            b = right.iloc[i]
            row += f" & {tex_escape(b.breed)} & {b.train} & {b.val} & {b.test} & {b.total}"
        else:
            row += " & & & & & "
        lines.append(row + r" \\")
    lines += [
        r"\midrule",
        f"Total & {counts.train.sum()} & {counts.val.sum()} & {counts.test.sum()} & "
        f"{counts.total.sum()} & & & & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES_DIR / "dataset_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return counts


def confusion_figure(nn, sklearn, names):
    panels = [
        ("Full fine-tuning", nn["full"]["test"]),
        ("Classifier head only", nn["head"]["test"]),
        ("Logistic regression (frozen)", sklearn["logreg"]["test"]),
    ]
    short = [display_name(n) for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, (title, metrics) in zip(axes, panels):
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\nacc {pct(metrics['accuracy'])}%, macro-F1 {pct(metrics['macro_f1'])}%")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, rotation=90, fontsize=4.6)
        ax.set_yticklabels(short, fontsize=4.6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("Fraction of true class")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "confusion_matrices.pdf")
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close(fig)

    for key, title, metrics in [
        ("full", "Full fine-tuning", nn["full"]["test"]),
        ("head", "Classifier head only", nn["head"]["test"]),
        ("logreg", "Logistic regression", sklearn["logreg"]["test"]),
    ]:
        fig, ax = plt.subplots(figsize=(11, 9.5))
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Confusion matrix (row-normalised) -- {title}")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, rotation=90, fontsize=6.5)
        ax.set_yticklabels(short, fontsize=6.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction of true class")
        fig.savefig(FIGURES_DIR / f"confusion_{key}.pdf")
        fig.savefig(FIGURES_DIR / f"confusion_{key}.png", dpi=250)
        plt.close(fig)

    top_confusions(nn["head"]["test"], names)
    confusion_pair(nn, names)


def confusion_pair(nn, names):
    """Compact two-panel matrix at final print size, shared y labels."""
    short = [display_name(n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.4), gridspec_kw={"wspace": 0.06})
    for ax, (tag, title) in zip(axes, [("full", "Full fine-tuning"), ("head", "Classifier head")]):
        metrics = nn[tag]["test"]
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title} (acc {pct(metrics['accuracy'])}%)", fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, rotation=90, fontsize=3.8)
        if tag == "full":
            ax.set_yticklabels(short, fontsize=5.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=1.5, pad=1)
        ax.set_xlabel("Predicted", fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("True breed", fontsize=6)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Fraction of true class", fontsize=6)
    fig.savefig(FIGURES_DIR / "confusion_pair.pdf")
    fig.savefig(FIGURES_DIR / "confusion_pair.png", dpi=300)
    plt.close(fig)


def per_class_table(nn, names, tag="head"):
    rows = []
    for name in names:
        m = nn[tag]["test"]["per_class"][name]
        rows.append(
            {
                "breed": display_name(name),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "support": int(m["support"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("f1")
    df.to_csv(TABLES_DIR / f"per_class_f1_{tag}.csv", index=False)
    return df


def format_p(p: float) -> str:
    """Return a LaTeX relation, so the report can write ``$p\\pFullHead$``."""
    return "<0.001" if p < 0.001 else f"={p:.2f}"


def mcnemar_p(a: str, b: str) -> float:
    """Look up the paired McNemar p-value written by compare_models.py."""
    path = TABLES_DIR / "mcnemar.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    row = df[(df.model_a == a) & (df.model_b == b)]
    if row.empty:
        row = df[(df.model_a == b) & (df.model_b == a)]
    return float(row.mcnemar_p.iloc[0]) if not row.empty else float("nan")


def write_macros(nn, sklearn, extraction_seconds, names):
    df = per_class_table(nn, names)
    worst = df.iloc[0]
    best = df.iloc[-1]
    top = pd.read_csv(TABLES_DIR / "top_confusions.csv").iloc[0]
    full_acc = nn["full"]["test"]["accuracy"]
    head_acc = nn["head"]["test"]["accuracy"]
    lr_acc = sklearn["logreg"]["test"]["accuracy"]
    full_min = nn["full"]["train"]["training_seconds"] / 60
    head_min = nn["head"]["train"]["training_seconds"] / 60
    lr_min = (extraction_seconds + sklearn["logreg"]["fit_seconds"]) / 60
    full_pred = pd.read_csv(PRED_DIR / "full_test_predictions.csv").sort_values("filename")
    lr_pred = pd.read_csv(PRED_DIR / "sklearn_logreg_test_predictions.csv").sort_values("filename")
    full_wrong = (full_pred.label != full_pred.prediction).values
    lr_wrong = (lr_pred.label != lr_pred.prediction).values
    vals = {
        "fullAcc": pct(nn["full"]["test"]["accuracy"]),
        "fullFone": pct(nn["full"]["test"]["macro_f1"]),
        "fullTrainAcc": pct(nn["full"]["train"]["train_eval_accuracy"]),
        "fullGap": f"{100 * (nn['full']['train']['train_eval_accuracy'] - nn['full']['test']['accuracy']):.1f}",
        "fullTime": f"{nn['full']['train']['training_seconds'] / 60:.1f}",
        "fullParams": f"{nn['full']['train']['trainable_parameters']:,}",
        "fullEpoch": str(nn["full"]["train"]["best_epoch"]),
        "headAcc": pct(nn["head"]["test"]["accuracy"]),
        "headFone": pct(nn["head"]["test"]["macro_f1"]),
        "headTrainAcc": pct(nn["head"]["train"]["train_eval_accuracy"]),
        "headGap": f"{100 * (nn['head']['train']['train_eval_accuracy'] - nn['head']['test']['accuracy']):.1f}",
        "headTime": f"{nn['head']['train']['training_seconds'] / 60:.1f}",
        "headParams": f"{nn['head']['train']['trainable_parameters']:,}",
        "headEpoch": str(nn["head"]["train"]["best_epoch"]),
        "lrAcc": pct(sklearn["logreg"]["test"]["accuracy"]),
        "lrFone": pct(sklearn["logreg"]["test"]["macro_f1"]),
        "lrTrainAcc": pct(sklearn["logreg"]["train_accuracy"]),
        "lrGap": f"{100 * (sklearn['logreg']['train_accuracy'] - sklearn['logreg']['test']['accuracy']):.1f}",
        "lrTime": f"{(extraction_seconds + sklearn['logreg']['fit_seconds']) / 60:.1f}",
        "lrConfig": sklearn["logreg"]["selected_config"],
        "svmAcc": pct(sklearn["linear_svm"]["test"]["accuracy"]),
        "svmFone": pct(sklearn["linear_svm"]["test"]["macro_f1"]),
        "svmTime": f"{(extraction_seconds + sklearn['linear_svm']['fit_seconds']) / 60:.1f}",
        "rfAcc": pct(sklearn["random_forest"]["test"]["accuracy"]),
        "rfFone": pct(sklearn["random_forest"]["test"]["macro_f1"]),
        "rfTime": f"{(extraction_seconds + sklearn['random_forest']['fit_seconds']) / 60:.1f}",
        "extractTime": f"{extraction_seconds / 60:.1f}",
        "worstClass": tex_escape(worst.breed),
        "worstFone": f"{100 * worst.f1:.1f}",
        "bestClass": tex_escape(best.breed),
        "bestFone": f"{100 * best.f1:.1f}",
        "topPairA": tex_escape(top.true),
        "topPairB": tex_escape(top.predicted),
        "topPairCount": str(int(top["count"])),
        "topPairFrac": f"{100 * top.fraction_of_true:.1f}",
        "fullVal": pct(nn["full"]["train"]["final_val_accuracy"]),
        "headVal": pct(nn["head"]["train"]["final_val_accuracy"]),
        "headLrDiff": f"{100 * (head_acc - lr_acc):.1f}",
        "timeRatioFH": f"{full_min / head_min:.1f}",
        "fullHeadDiff": f"{100 * abs(full_acc - head_acc):.1f}",
        "lrFullDiff": f"{100 * (lr_acc - full_acc):+.1f}",
        "costRatio": f"{full_min / lr_min:.0f}",
        "pFullLr": format_p(mcnemar_p("Full fine-tuning", "Logistic regression (frozen)")),
        "pFullHead": format_p(mcnemar_p("Full fine-tuning", "Classifier head only")),
        "pHeadLr": format_p(mcnemar_p("Classifier head only", "Logistic regression (frozen)")),
        "sharedErrors": str(int((full_wrong & lr_wrong).sum())),
        "lrErrors": str(int(lr_wrong.sum())),
    }
    lines = ["% Auto-generated by src/make_report_assets.py"]
    for key, value in vals.items():
        lines.append(rf"\newcommand{{\{key}}}{{{value}}}")
    (TABLES_DIR / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_confusions(metrics, names):
    cm = np.asarray(metrics["confusion_matrix"])
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], cm[i, j] / cm[i].sum(), names[i], names[j]))
    pairs.sort(reverse=True)
    rows = [
        {
            "true": display_name(t),
            "predicted": display_name(p),
            "count": int(c),
            "fraction_of_true": float(f),
        }
        for c, f, t, p in pairs[:10]
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "top_confusions.csv", index=False)
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"True breed & Predicted as & Count & \% of true class \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{tex_escape(r['true'])} & {tex_escape(r['predicted'])} & {r['count']} & "
            f"{100 * r['fraction_of_true']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "top_confusions_table.tex").write_text("\n".join(lines), encoding="utf-8")


def training_curves(nn):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for tag, label, colour in (("full", "Full fine-tuning", "#1f77b4"), ("head", "Classifier head", "#d62728")):
        h = pd.DataFrame(nn[tag]["train"]["history"])
        axes[0].plot(h.epoch, h.train_loss, marker="o", ms=3, color=colour, label=label)
        axes[0].plot(h.epoch, h.val_loss, marker="s", ms=3, ls="--", color=colour, alpha=0.7)
        axes[1].plot(h.epoch, h.val_accuracy * 100, marker="o", ms=3, color=colour, label=label)
        axes[2].plot(h.epoch, h.val_macro_f1 * 100, marker="o", ms=3, color=colour, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss (solid: train, dashed: val)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Validation accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation macro-F1 (%)")
    axes[2].set_title("Validation macro-F1")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.pdf")
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=250)
    plt.close(fig)


def example_grid(pred_csv: Path, data_root: Path, out_stem: str, title_prefix: str, n_correct=4, n_wrong=4):
    df = pd.read_csv(pred_csv)
    # One example per true breed keeps the grid representative instead of showing
    # four near-duplicates of whichever breed the model is most confident about.
    correct = (
        df[df.label == df.prediction]
        .sort_values("confidence", ascending=False)
        .drop_duplicates("true_name")
        .head(n_correct)
    )
    wrong = (
        df[df.label != df.prediction]
        .sort_values("confidence", ascending=False)
        .drop_duplicates("true_name")
        .head(n_wrong)
    )
    picks = pd.concat([correct, wrong])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, (_, r) in zip(axes.ravel(), picks.iterrows()):
        img = mpimg.imread(data_root / "images" / f"{r.filename}.jpg")
        ax.imshow(img)
        ok = r.label == r.prediction
        ax.set_title(
            f"{'OK' if ok else 'WRONG'}: true {display_name(r.true_name)}\n"
            f"pred {display_name(r.pred_name)} ({100 * r.confidence:.0f}%)",
            fontsize=7.5,
            color="#1a7a1a" if ok else "#b30000",
        )
        ax.axis("off")
    fig.suptitle(title_prefix, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{out_stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{out_stem}.png", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    names = class_names(data_root)
    manifest = build_splits(data_root)
    nn, sklearn = load_all()
    extraction_meta = load_json(FEATURES_DIR / "extraction_meta.json")
    extraction_seconds = extraction_meta["total_seconds"]

    df = results_table(nn, sklearn, extraction_seconds, len(names))
    dataset_table(manifest, names)
    confusion_figure(nn, sklearn, names)
    training_curves(nn)
    example_grid(
        PRED_DIR / "full_test_predictions.csv",
        data_root,
        "examples_full",
        "Full fine-tuning: most confident correct (top) and incorrect (bottom) test predictions",
    )
    example_grid(
        PRED_DIR / "head_test_predictions.csv",
        data_root,
        "examples_head",
        "Classifier-head training: most confident correct (top) and incorrect (bottom)",
    )
    write_macros(nn, sklearn, extraction_seconds, names)

    summary = {
        "full_train_seconds": nn["full"]["train"]["training_seconds"],
        "head_train_seconds": nn["head"]["train"]["training_seconds"],
        "extraction_seconds": extraction_seconds,
        "full_trainable": nn["full"]["train"]["trainable_parameters"],
        "head_trainable": nn["head"]["train"]["trainable_parameters"],
        "best_epoch_full": nn["full"]["train"]["best_epoch"],
        "best_epoch_head": nn["head"]["train"]["best_epoch"],
        "test": {k: {"accuracy": v["test"]["accuracy"], "macro_f1": v["test"]["macro_f1"]} for k, v in nn.items()},
        "sklearn": {k: {"config": v["selected_config"], "accuracy": v["test"]["accuracy"],
                        "macro_f1": v["test"]["macro_f1"], "fit_seconds": v["fit_seconds"]}
                    for k, v in sklearn.items()},
    }
    (TABLES_DIR / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
