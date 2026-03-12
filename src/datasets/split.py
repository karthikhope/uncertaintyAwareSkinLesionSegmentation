"""
Create deterministic train/val/test splits for ISIC 2018 Task 1.

Split ratio: 70% train, 15% val, 15% test.
Random seed: 42 (fixed for reproducibility).

Usage:
    python -m datasets.split              # run from src/
    python src/datasets/split.py          # run from project root
"""

import os
import sys
import csv
from pathlib import Path


def _shuffle_split(items, train_ratio=0.70, seed=42):
    """Split a list into train/val/test using a seeded Fisher-Yates shuffle."""
    import random
    rng = random.Random(seed)

    items = sorted(items)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = (n - n_train) // 2
    # n_test gets the remainder so nothing is lost
    n_test = n - n_train - n_val

    train_ids = [items[i] for i in indices[:n_train]]
    val_ids = [items[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [items[i] for i in indices[n_train + n_val:]]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def _write_csv(ids, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id"])
        for image_id in ids:
            writer.writerow([image_id])


def create_splits(data_root, output_dir, train_ratio=0.70, seed=42):
    images_dir = Path(data_root) / "images"
    masks_dir = Path(data_root) / "masks"

    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    image_ids = [f.replace(".jpg", "") for f in image_files]

    mask_ids = set(
        f.replace("_segmentation.png", "")
        for f in os.listdir(masks_dir)
        if f.endswith(".png")
    )
    missing = [iid for iid in image_ids if iid not in mask_ids]
    if missing:
        print(f"WARNING: {len(missing)} images have no matching mask. They will be excluded.")
        image_ids = [iid for iid in image_ids if iid in mask_ids]

    train_ids, val_ids, test_ids = _shuffle_split(image_ids, train_ratio, seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(train_ids, out / "train.csv")
    _write_csv(val_ids, out / "val.csv")
    _write_csv(test_ids, out / "test.csv")

    print(f"Total images: {len(image_ids)}")
    print(f"  Train: {len(train_ids)}  ({100*len(train_ids)/len(image_ids):.1f}%)")
    print(f"  Val:   {len(val_ids)}  ({100*len(val_ids)/len(image_ids):.1f}%)")
    print(f"  Test:  {len(test_ids)}  ({100*len(test_ids)/len(image_ids):.1f}%)")
    print(f"Splits saved to {out.resolve()}")

    overlap_tv = set(train_ids) & set(val_ids)
    overlap_tt = set(train_ids) & set(test_ids)
    overlap_vt = set(val_ids) & set(test_ids)
    assert not overlap_tv, f"Train/Val overlap: {len(overlap_tv)}"
    assert not overlap_tt, f"Train/Test overlap: {len(overlap_tt)}"
    assert not overlap_vt, f"Val/Test overlap: {len(overlap_vt)}"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(image_ids), "Count mismatch"
    print("No overlaps detected. Splits are clean.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_root = project_root / "data" / "ISIC2018"
    output_dir = project_root / "data" / "splits"
    create_splits(str(data_root), str(output_dir))
