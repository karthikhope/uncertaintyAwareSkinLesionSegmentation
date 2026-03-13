"""
Plot training curves from training_history.csv.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --csv src/training_history.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(csv_path):
    epochs, losses, dices, ious, bests = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            dices.append(float(row["val_dice"]))
            ious.append(float(row["val_iou"]))
            bests.append(row["is_best"].strip().lower() == "true")
    return (np.array(epochs), np.array(losses), np.array(dices),
            np.array(ious), np.array(bests))


def plot(epochs, losses, dices, ious, bests, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Train Loss ---
    ax = axes[0]
    ax.plot(epochs, losses, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (BCE + Dice)")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # --- Val Dice ---
    ax = axes[1]
    ax.plot(epochs, dices, color="seagreen", linewidth=1.2, label="Val Dice")
    best_mask = bests
    ax.scatter(epochs[best_mask], dices[best_mask], color="red", s=30,
               zorder=5, label="New best")
    best_epoch = epochs[best_mask][-1]
    best_dice = dices[best_mask][-1]
    ax.axhline(best_dice, color="red", linestyle="--", alpha=0.4)
    ax.annotate(f"Best: {best_dice:.4f} (ep {best_epoch})",
                xy=(best_epoch, best_dice),
                xytext=(best_epoch - 25, best_dice - 0.015),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Dice")
    ax.set_title("Validation Dice Score")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Val IoU ---
    ax = axes[2]
    ax.plot(epochs, ious, color="darkorange", linewidth=1.2, label="Val IoU")
    ax.scatter(epochs[best_mask], ious[best_mask], color="red", s=30,
               zorder=5, label="New best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val IoU")
    ax.set_title("Validation IoU (Jaccard)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bayesian U-Net (ResNet34 + MC Dropout) -- ISIC 2018 Training",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--csv", type=str, default="src/training_history.csv")
    parser.add_argument("--output", type=str, default="reports/training_curves_local.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    epochs, losses, dices, ious, bests = load_history(csv_path)
    print(f"Loaded {len(epochs)} epochs from {csv_path}")
    print(f"  Best Dice: {dices.max():.4f} at epoch {epochs[dices.argmax()]}")
    print(f"  Final Loss: {losses[-1]:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(epochs, losses, dices, ious, bests, out_path)


if __name__ == "__main__":
    main()
