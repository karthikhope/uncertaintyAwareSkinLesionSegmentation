"""
Multi-model comparison: overlaid severity plots, bar charts, heatmaps,
and a consolidated markdown report.

Usage:
    python scripts/compare_models.py

    python scripts/compare_models.py \
        --csvs reports/bayesian/eval_results.csv \
              reports/attention_unet/eval_results.csv \
              reports/resunet/eval_results.csv \
        --labels "U-Net" "Attention U-Net" "ResUNet" \
        --output_dir reports/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULTS = {
    "csvs": [
        "reports/bayesian/eval_results.csv",
        "reports/attention_unet/eval_results.csv",
        "reports/resunet/eval_results.csv",
    ],
    "labels": ["U-Net (ResNet34)", "Attention U-Net", "ResUNet (ResNet50)"],
}

MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MODEL_MARKERS = ["o", "s", "^"]

CORRUPTION_ORDER = [
    "gaussian_blur", "motion_blur", "gaussian_noise", "speckle_noise",
    "jpeg_compression", "brightness_shift", "contrast_shift", "downscale",
]

CORRUPTION_LABELS = {
    "gaussian_blur": "Gaussian Blur",
    "motion_blur": "Motion Blur",
    "gaussian_noise": "Gaussian Noise",
    "speckle_noise": "Speckle Noise",
    "jpeg_compression": "JPEG Compression",
    "brightness_shift": "Brightness Shift",
    "contrast_shift": "Contrast Shift",
    "downscale": "Downscale",
}


def load_data(csv_paths, labels):
    frames = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        df["model_label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_metric_overlay(df, metric, ylabel, title, save_path, figsize=(12, 7)):
    """One subplot per corruption type, all models overlaid."""
    corruptions = [c for c in CORRUPTION_ORDER if c in df["corruption"].unique()]
    n = len(corruptions)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    labels = df["model_label"].unique()

    for i, corruption in enumerate(corruptions):
        ax = axes[i]
        for j, label in enumerate(labels):
            sub = df[(df["corruption"] == corruption) & (df["model_label"] == label)]
            sub = sub.sort_values("severity")
            ax.plot(sub["severity"], sub[metric],
                    marker=MODEL_MARKERS[j], color=MODEL_COLORS[j],
                    label=label, linewidth=1.5, markersize=5)

        ax.set_title(CORRUPTION_LABELS.get(corruption, corruption), fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Severity", fontsize=9)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_clean_bars(df, save_path):
    """Grouped bar chart of clean-data metrics for all models."""
    clean = df[df["corruption"] == "clean"].copy()
    labels = clean["model_label"].values
    metrics = ["dice", "iou", "ece", "pece"]
    metric_labels = ["Dice", "IoU", "ECE", "pECE"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = clean[metric].values
        bars = ax.bar(range(len(labels)), vals, color=MODEL_COLORS[:len(labels)],
                      edgecolor="black", linewidth=0.5)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in labels],
                           fontsize=8, ha="center")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        if metric in ("dice", "iou"):
            ax.set_ylim(min(vals) - 0.01, max(vals) + 0.01)
        else:
            ax.set_ylim(0, max(vals) * 1.3)

    fig.suptitle("Clean Test Set Performance — Model Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_severity5_heatmap(df, save_path):
    """Heatmap of Dice at severity 5 (models x corruptions)."""
    s5 = df[(df["severity"] == 5) & (df["corruption"] != "clean")]
    labels = df["model_label"].unique()
    corruptions = [c for c in CORRUPTION_ORDER if c in s5["corruption"].unique()]

    matrix = np.zeros((len(labels), len(corruptions)))
    for i, label in enumerate(labels):
        for j, corruption in enumerate(corruptions):
            row = s5[(s5["model_label"] == label) & (s5["corruption"] == corruption)]
            if not row.empty:
                matrix[i, j] = row["dice"].values[0]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([CORRUPTION_LABELS.get(c, c) for c in corruptions],
                       fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(corruptions)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    best_per_col = matrix.argmax(axis=0)
    for j, best_i in enumerate(best_per_col):
        ax.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                                   fill=False, edgecolor="gold",
                                   linewidth=2.5))

    plt.colorbar(im, ax=ax, label="Dice Score", shrink=0.8)
    ax.set_title("Dice at Severity 5 — Model Comparison (gold = best)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_mean_degradation(df, save_path):
    """Bar chart: mean Dice drop from clean to severity 5, per model."""
    labels = df["model_label"].unique()
    s5 = df[(df["severity"] == 5) & (df["corruption"] != "clean")]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.6

    drops = []
    for label in labels:
        clean_dice = df[(df["model_label"] == label) & (df["corruption"] == "clean")]["dice"].values[0]
        mean_s5_dice = s5[s5["model_label"] == label]["dice"].mean()
        drops.append(clean_dice - mean_s5_dice)

    bars = ax.bar(x, drops, width, color=MODEL_COLORS[:len(labels)],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Dice Drop (clean → severity 5)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Mean Robustness: Average Dice Drop at Severity 5",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def write_comparison_report(df, out_dir):
    """Consolidated markdown report with tables and analysis."""
    labels = df["model_label"].unique()
    clean = df[df["corruption"] == "clean"]
    s5 = df[(df["severity"] == 5) & (df["corruption"] != "clean")]

    lines = [
        "# Multi-Model Comparison Report\n",
        "## Clean Test Set Performance\n",
        "| Model | Dice | IoU | ECE | pECE | MI |",
        "|-------|------|-----|-----|------|----|",
    ]
    for _, row in clean.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['dice']:.4f} | {row['iou']:.4f} "
            f"| {row['ece']:.4f} | {row['pece']:.4f} | {row['mean_mi']:.4f} |"
        )

    lines.append("\n## Dice at Severity 5 (All Corruptions)\n")
    header = "| Corruption |"
    sep = "|------------|"
    for label in labels:
        header += f" {label} |"
        sep += "------|"
    header += " Best |"
    sep += "------|"
    lines.extend([header, sep])

    corruptions = [c for c in CORRUPTION_ORDER if c in s5["corruption"].unique()]
    for corruption in corruptions:
        row_str = f"| {CORRUPTION_LABELS.get(corruption, corruption)} |"
        vals = []
        for label in labels:
            r = s5[(s5["model_label"] == label) & (s5["corruption"] == corruption)]
            v = r["dice"].values[0] if not r.empty else 0
            vals.append((label, v))
            row_str += f" {v:.4f} |"
        best_label = max(vals, key=lambda x: x[1])[0]
        row_str += f" {best_label} |"
        lines.append(row_str)

    lines.append("\n## Mean Dice Drop (Clean → Severity 5)\n")
    lines.append("| Model | Clean Dice | Mean S5 Dice | Mean Drop | Relative Drop |")
    lines.append("|-------|-----------|-------------|-----------|---------------|")
    for label in labels:
        clean_dice = clean[clean["model_label"] == label]["dice"].values[0]
        mean_s5 = s5[s5["model_label"] == label]["dice"].mean()
        drop = clean_dice - mean_s5
        rel_drop = drop / clean_dice * 100
        lines.append(
            f"| {label} | {clean_dice:.4f} | {mean_s5:.4f} "
            f"| {drop:.4f} | {rel_drop:.1f}% |"
        )

    lines.append("\n## ECE at Severity 5\n")
    header = "| Corruption |"
    sep = "|------------|"
    for label in labels:
        header += f" {label} |"
        sep += "------|"
    header += " Best |"
    sep += "------|"
    lines.extend([header, sep])

    for corruption in corruptions:
        row_str = f"| {CORRUPTION_LABELS.get(corruption, corruption)} |"
        vals = []
        for label in labels:
            r = s5[(s5["model_label"] == label) & (s5["corruption"] == corruption)]
            v = r["ece"].values[0] if not r.empty else 1
            vals.append((label, v))
            row_str += f" {v:.4f} |"
        best_label = min(vals, key=lambda x: x[1])[0]
        row_str += f" {best_label} |"
        lines.append(row_str)

    lines.append("\n## Key Findings\n")
    best_clean = clean.loc[clean["dice"].idxmax()]
    best_ece = clean.loc[clean["ece"].idxmin()]

    mean_s5_per_model = {
        label: s5[s5["model_label"] == label]["dice"].mean() for label in labels
    }
    most_robust = max(mean_s5_per_model, key=mean_s5_per_model.get)

    lines.extend([
        f"1. **Best clean accuracy:** {best_clean['model_label']} "
        f"(Dice {best_clean['dice']:.4f})",
        f"2. **Best calibration:** {best_ece['model_label']} "
        f"(ECE {best_ece['ece']:.4f})",
        f"3. **Most robust under corruption:** {most_robust} "
        f"(mean severity-5 Dice {mean_s5_per_model[most_robust]:.4f})",
        "4. **All models fail on Gaussian noise s5** — "
        "this is a shared vulnerability from training on clean data only",
        "",
    ])

    report_path = out_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison")
    parser.add_argument("--csvs", nargs="+", default=DEFAULTS["csvs"],
                        help="Paths to eval_results.csv files")
    parser.add_argument("--labels", nargs="+", default=DEFAULTS["labels"],
                        help="Display labels for each model")
    parser.add_argument("--output_dir", type=str, default="reports/comparison",
                        help="Output directory for comparison plots")
    args = parser.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError(f"Number of CSVs ({len(args.csvs)}) must match "
                         f"number of labels ({len(args.labels)})")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csvs, args.labels)

    plot_clean_bars(df, out / "clean_comparison.png")
    plot_severity5_heatmap(df, out / "severity5_heatmap.png")
    plot_mean_degradation(df, out / "mean_degradation.png")

    plot_metric_overlay(df, "dice", "Dice", "Dice vs Severity — All Models",
                        out / "dice_comparison.png")
    plot_metric_overlay(df, "ece", "ECE", "ECE vs Severity — All Models",
                        out / "ece_comparison.png")
    plot_metric_overlay(df, "mean_mi", "Mean MI",
                        "Mutual Information vs Severity — All Models",
                        out / "mi_comparison.png")

    write_comparison_report(df, out)

    print(f"\nAll comparison outputs saved to {out}/")


if __name__ == "__main__":
    main()
