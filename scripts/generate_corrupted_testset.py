"""
Generate matched corrupted test sets for domain-shift evaluation.

For every test image, produces corrupted variants at 5 severity levels
for each corruption type, plus a manifest CSV.

Usage:
    python scripts/generate_corrupted_testset.py
    python scripts/generate_corrupted_testset.py --data_root data/ISIC2018 --splits_root data/splits
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from augment.corruptions import apply_corruption, CORRUPTION_TYPES


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted test sets")
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits_root = Path(args.splits_root)

    test_csv = splits_root / "test.csv"
    if not test_csv.exists():
        print(f"ERROR: {test_csv} not found. Run src/datasets/split.py first.")
        sys.exit(1)

    test_ids = []
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row["image_id"])

    print(f"Found {len(test_ids)} test images")
    print(f"Corruptions: {CORRUPTION_TYPES}")
    print(f"Severities: 1-5")
    total = len(test_ids) * len(CORRUPTION_TYPES) * 5
    print(f"Total images to generate: {total}")

    manifest_rows = []
    done = 0

    for corruption in CORRUPTION_TYPES:
        for severity in range(1, 6):
            out_dir = data_root / "corrupted" / corruption / f"severity_{severity}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, image_id in enumerate(test_ids):
                src_path = data_root / "images" / f"{image_id}.jpg"
                if not src_path.exists():
                    continue

                img = Image.open(src_path).convert("RGB")
                img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
                img_arr = np.array(img, dtype=np.float32) / 255.0

                corrupted = apply_corruption(img_arr, corruption, severity, seed=idx)

                out_path = out_dir / f"{image_id}.jpg"
                out_img = Image.fromarray((corrupted * 255).astype(np.uint8))
                out_img.save(str(out_path), quality=95)

                manifest_rows.append({
                    "corruption_type": corruption,
                    "severity": severity,
                    "image_id": image_id,
                    "path": str(out_path),
                })

                done += 1
                if done % 500 == 0:
                    print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")

    manifest_path = data_root / "corrupted" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["corruption_type", "severity", "image_id", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone! Generated {done} corrupted images.")
    print(f"Manifest: {manifest_path} ({len(manifest_rows)} rows)")
    expected = len(test_ids) * len(CORRUPTION_TYPES) * 5
    assert len(manifest_rows) == expected, \
        f"Expected {expected} rows, got {len(manifest_rows)}"


if __name__ == "__main__":
    main()
