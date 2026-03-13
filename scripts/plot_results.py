"""
Generate Dice/IoU vs severity and ECE vs severity plots, plus an
uncertainty statistics table from the evaluation CSV.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results_csv reports/eval_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric_vs_severity(df, metric, ylabel, title, save_path, clean_val):
    """Line plot of a metric vs severity, one line per corruption type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    corruptions = [c for c in df["corruption"].unique() if c != "clean"]
    colors = plt.cm.tab10(range(len(corruptions)))

    for corruption, color in zip(corruptions, colors):
        sub = df[df["corruption"] == corruption].sort_values("severity")
        ax.plot(
            sub["severity"], sub[metric],
            marker="o", label=corruption.replace("_", " "), color=color,
        )

    ax.axhline(y=clean_val, color="black", linestyle="--", linewidth=1.5,
               label=f"Clean ({clean_val:.3f})")

    ax.set_xlabel("Corruption Severity", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, 6))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def create_uncertainty_table(df, save_path):
    """Markdown table of mean MI across corruptions and severities."""
    clean_mi = df[df["corruption"] == "clean"]["mean_mi"].values[0]

    lines = [
        "# Uncertainty Statistics (Mean Mutual Information)\n",
        f"**Clean test set MI:** {clean_mi:.4f}\n",
        "| Corruption | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |",
        "|------------|-------|-------|-------|-------|-------|",
    ]

    corruptions = [c for c in df["corruption"].unique() if c != "clean"]
    for corruption in corruptions:
        sub = df[df["corruption"] == corruption].sort_values("severity")
        vals = sub["mean_mi"].values
        row = f"| {corruption.replace('_', ' ')} |"
        for v in vals:
            row += f" {v:.4f} |"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--results_csv", type=str, default="reports/eval_results.csv")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    clean = df[df["corruption"] == "clean"].iloc[0]
    corrupted = df[df["corruption"] != "clean"]

    plot_metric_vs_severity(
        corrupted, "dice", "Dice Score",
        "Dice Score vs Corruption Severity",
        reports / "dice_vs_severity.png",
        clean["dice"],
    )

    plot_metric_vs_severity(
        corrupted, "iou", "IoU Score",
        "IoU vs Corruption Severity",
        reports / "iou_vs_severity.png",
        clean["iou"],
    )

    plot_metric_vs_severity(
        corrupted, "ece", "ECE",
        "ECE vs Corruption Severity",
        reports / "ece_vs_severity.png",
        clean["ece"],
    )

    plot_metric_vs_severity(
        corrupted, "pece", "pECE",
        "Per-class ECE vs Corruption Severity",
        reports / "pece_vs_severity.png",
        clean["pece"],
    )

    create_uncertainty_table(df, reports / "uncertainty_table.md")

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
