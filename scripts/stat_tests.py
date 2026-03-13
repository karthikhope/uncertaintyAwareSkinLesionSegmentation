"""
Statistical significance tests: Wilcoxon signed-rank for Dice drops,
bootstrap comparison for ECE, and summary significance table.

Usage:
    python scripts/stat_tests.py --checkpoint src/best_model.pth
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.unet import get_unet
from metrics.seg import dice_score
from metrics.calibration import pixel_ece
from utils import enable_dropout


CORRUPTION_TYPES = [
    "gaussian_blur", "motion_blur", "gaussian_noise", "speckle_noise",
    "jpeg_compression", "brightness_shift", "contrast_shift", "downscale",
]

TEST_SEVERITIES = [3, 5]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def per_image_dice(model, loader, device, mc_passes=0, desc=""):
    """Compute Dice score for each image individually."""
    model.eval()
    if mc_passes > 0:
        enable_dropout(model)

    dice_values = []
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc=f"  {desc} inference", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        if mc_passes > 0:
            preds = []
            for _ in range(mc_passes):
                with torch.no_grad():
                    logits = model(images)
                    preds.append(torch.sigmoid(logits))
            mean_pred = torch.stack(preds).mean(dim=0)
            logits_equiv = torch.logit(mean_pred.clamp(1e-6, 1 - 1e-6))
        else:
            with torch.no_grad():
                logits_equiv = model(images)
            mean_pred = torch.sigmoid(logits_equiv)

        for i in range(images.shape[0]):
            d = dice_score(
                logits_equiv[i:i+1], masks[i:i+1]
            ).item()
            dice_values.append(d)

        all_probs.append(mean_pred.cpu().numpy().flatten())
        all_labels.append(masks.cpu().numpy().flatten())
        pbar.set_postfix(dice=f"{np.mean(dice_values):.4f}")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return np.array(dice_values), all_probs, all_labels


def bootstrap_ece(probs, labels, n_bootstrap=1000, n_bins=15, seed=42,
                  max_pixels=1_000_000, desc=""):
    """Bootstrap the ECE to obtain a distribution.

    Subsamples to max_pixels for speed (25M+ pixels make full bootstrap
    prohibitively slow). 1M pixels is statistically sufficient for ECE.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)

    if n > max_pixels:
        subsample_idx = rng.choice(n, size=max_pixels, replace=False)
        probs = probs[subsample_idx]
        labels = labels[subsample_idx]
        n = max_pixels

    ece_samples = []
    pbar = tqdm(range(n_bootstrap), desc=f"  {desc} bootstrap ECE",
                leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
    for _ in pbar:
        idx = rng.integers(0, n, size=n)
        ece_val, _, _, _ = pixel_ece(probs[idx], labels[idx], n_bins)
        ece_samples.append(ece_val)

    return np.array(ece_samples)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--checkpoint", type=str, default="src/best_model.pth")
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = get_unet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}")

    from datasets.isic import ISICDataset

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    total_pairs = len(CORRUPTION_TYPES) * len(TEST_SEVERITIES)
    t_start = time.time()

    print("\n[0/{total}] Computing per-image Dice on clean test set...".format(
        total=total_pairs))
    clean_ds = ISICDataset(
        "test", data_root=args.data_root, splits_root=args.splits_root,
        image_size=args.image_size,
    )
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, num_workers=0)
    dice_clean, probs_clean, labels_clean = per_image_dice(
        model, clean_loader, device, args.mc_passes, desc="Clean"
    )
    ece_clean, _, _, _ = pixel_ece(probs_clean, labels_clean)
    print(f"  Clean: mean Dice={dice_clean.mean():.4f}, ECE={ece_clean:.4f}")

    dice_rows = []
    ece_rows = []
    summary_rows = []
    pair_idx = 0

    for corruption in CORRUPTION_TYPES:
        for severity in TEST_SEVERITIES:
            pair_idx += 1
            elapsed = time.time() - t_start
            if pair_idx > 1:
                eta = elapsed / (pair_idx - 1) * (total_pairs - pair_idx + 1)
                eta_str = f" | ETA: {format_time(eta)}"
            else:
                eta_str = ""

            print(f"\n[{pair_idx}/{total_pairs}] {corruption} severity={severity} "
                  f"| elapsed: {format_time(elapsed)}{eta_str}")

            corrupt_ds = ISICDataset(
                "test", data_root=args.data_root, splits_root=args.splits_root,
                image_size=args.image_size,
                corruption_type=corruption, corruption_severity=severity,
            )
            corrupt_loader = DataLoader(
                corrupt_ds, batch_size=args.batch_size, num_workers=0
            )
            dice_corrupt, probs_corrupt, labels_corrupt = per_image_dice(
                model, corrupt_loader, device, args.mc_passes,
                desc=f"{corruption}_s{severity}"
            )
            ece_corrupt, _, _, _ = pixel_ece(probs_corrupt, labels_corrupt)

            n_common = min(len(dice_clean), len(dice_corrupt))
            stat, p_value = stats.wilcoxon(
                dice_clean[:n_common], dice_corrupt[:n_common],
                alternative="two-sided",
            )
            dice_drop = dice_corrupt.mean() - dice_clean.mean()

            dice_rows.append({
                "corruption": corruption,
                "severity": severity,
                "mean_dice_clean": round(dice_clean.mean(), 4),
                "mean_dice_corrupt": round(dice_corrupt.mean(), 4),
                "dice_drop": round(dice_drop, 4),
                "wilcoxon_stat": round(stat, 2),
                "p_value": f"{p_value:.2e}",
            })

            print(f"  Dice: {dice_clean.mean():.4f} -> {dice_corrupt.mean():.4f} "
                  f"(drop={dice_drop:+.4f}, p={p_value:.2e})")

            ece_boot_clean = bootstrap_ece(
                probs_clean, labels_clean, desc="clean")
            ece_boot_corrupt = bootstrap_ece(
                probs_corrupt, labels_corrupt, desc=f"{corruption}_s{severity}")
            ece_diff = ece_boot_corrupt - ece_boot_clean
            ci_lo, ci_hi = np.percentile(ece_diff, [2.5, 97.5])
            ece_increase = ece_corrupt - ece_clean

            ece_rows.append({
                "corruption": corruption,
                "severity": severity,
                "ece_clean": round(ece_clean, 4),
                "ece_corrupt": round(ece_corrupt, 4),
                "ece_increase": round(ece_increase, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
            })

            print(f"  ECE: {ece_clean:.4f} -> {ece_corrupt:.4f} "
                  f"(increase={ece_increase:+.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}])")

            summary_rows.append({
                "corruption": corruption,
                "severity": severity,
                "dice_drop": round(dice_drop, 4),
                "p_value": f"{p_value:.2e}",
                "ece_increase": round(ece_increase, 4),
                "ci_95": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            })

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All {total_pairs} pairs complete in {format_time(total_time)}")
    print(f"{'='*60}")

    dice_csv = reports / "stat_tests_dice.csv"
    with open(dice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dice_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dice_rows)
    print(f"Saved: {dice_csv}")

    ece_csv = reports / "stat_tests_ece.csv"
    with open(ece_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ece_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ece_rows)
    print(f"Saved: {ece_csv}")

    summary_md = reports / "significance_summary.md"
    with open(summary_md, "w") as f:
        f.write("# Statistical Significance Summary\n\n")
        f.write("| Corruption | Severity | Dice Drop | p-value | ECE Increase | 95% CI |\n")
        f.write("|------------|----------|-----------|---------|--------------|--------|\n")
        for row in summary_rows:
            f.write(
                f"| {row['corruption']} | {row['severity']} | "
                f"{row['dice_drop']:+.4f} | {row['p_value']} | "
                f"{row['ece_increase']:+.4f} | {row['ci_95']} |\n"
            )
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
