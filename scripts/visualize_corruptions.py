"""
Visual sanity check: grid of all corruptions at all severities
for a single sample image.

Produces a PNG with rows = corruption types, columns = severities.

Usage:
    python scripts/visualize_corruptions.py
    python scripts/visualize_corruptions.py --image_id ISIC_0024306
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from augment.corruptions import apply_corruption, CORRUPTION_TYPES


def main():
    parser = argparse.ArgumentParser(description="Visualize corruption grid")
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--image_id", type=str, default=None,
                        help="Specific image ID. If None, uses the first test image.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output", type=str, default="reports/corruption_grid.png")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.image_id is None:
        test_csv = Path(args.splits_root) / "test.csv"
        with open(test_csv, "r") as f:
            reader = csv.DictReader(f)
            args.image_id = next(reader)["image_id"]
        print(f"Using first test image: {args.image_id}")

    img_path = data_root / "images" / f"{args.image_id}.jpg"
    img = Image.open(img_path).convert("RGB")
    img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0

    n_corruptions = len(CORRUPTION_TYPES)
    n_severities = 5

    fig, axes = plt.subplots(
        n_corruptions + 1, n_severities + 1,
        figsize=(3 * (n_severities + 1), 3 * (n_corruptions + 1)),
    )

    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")
    for j in range(1, n_severities + 1):
        axes[0, j].axis("off")
        axes[0, j].set_title(f"Severity {j}", fontsize=9)

    for i, corruption in enumerate(CORRUPTION_TYPES):
        row = i + 1
        axes[row, 0].imshow(img_arr)
        axes[row, 0].set_ylabel(corruption.replace("_", "\n"), fontsize=8, rotation=0,
                                 labelpad=60, ha="right", va="center")
        axes[row, 0].set_title("Clean" if i == 0 else "", fontsize=8)
        axes[row, 0].axis("off")

        for severity in range(1, 6):
            corrupted = apply_corruption(img_arr, corruption, severity, seed=0)
            axes[row, severity].imshow(np.clip(corrupted, 0, 1))
            axes[row, severity].axis("off")

    plt.suptitle(f"Corruption Grid — {args.image_id}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved corruption grid to {out_path}")


if __name__ == "__main__":
    main()
