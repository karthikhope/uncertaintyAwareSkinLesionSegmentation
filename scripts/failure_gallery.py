"""
Task 6.2 — Failure case gallery.

Finds the N worst-performing test images (lowest Dice) under clean and
corrupted conditions, then plots a gallery showing:
  Input | Ground Truth | Prediction | Epistemic (MI) | Dice score

Helps identify where the model struggles and whether uncertainty correlates
with failure (high epistemic uncertainty on failure cases = well-calibrated).

Usage:
    python scripts/failure_gallery.py
    python scripts/failure_gallery.py --checkpoint src/best_model.pth --top_k 8
    python scripts/failure_gallery.py --corruption gaussian_noise --severity 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.unet import get_unet
from datasets.isic import ISICDataset, IMAGENET_MEAN, IMAGENET_STD
from metrics.seg import dice_score
from metrics.uncertainty import predictive_entropy, mutual_information
from utils import enable_dropout


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mc_inference(model, image, T, device):
    """Run T stochastic forward passes; return mean prediction and samples."""
    model.eval()
    enable_dropout(model)

    preds = []
    for _ in range(T):
        with torch.no_grad():
            logits = model(image.to(device))
            preds.append(torch.sigmoid(logits).cpu())

    samples = torch.stack(preds)
    mean_pred = samples.mean(dim=0)
    return mean_pred, samples


def denormalize(img_tensor):
    """Undo ImageNet normalization for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def compute_per_image_dice(model, dataset, device, mc_passes):
    """Return per-image Dice scores, predictions, and uncertainty maps."""
    results = []
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0)

        mean_pred, samples = mc_inference(model, image_batch, mc_passes, device)
        logits_equiv = torch.logit(mean_pred.clamp(1e-6, 1 - 1e-6))
        d = dice_score(logits_equiv, mask.unsqueeze(0)).item()

        epistemic = mutual_information(mean_pred, samples.to(mean_pred.device))
        total_unc = predictive_entropy(mean_pred)

        results.append({
            "idx": idx,
            "dice": d,
            "input": denormalize(image),
            "gt": mask[0].numpy(),
            "pred": (mean_pred[0, 0] > 0.5).float().numpy(),
            "epistemic": epistemic[0, 0].numpy(),
            "total_unc": total_unc[0, 0].numpy(),
            "mean_mi": epistemic[0, 0].mean().item(),
        })

        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"  Processed {idx + 1}/{len(dataset)} images")

    return results


def plot_gallery(items, title, save_path):
    """Plot a gallery of worst-performing images."""
    n = len(items)
    fig, axes = plt.subplots(n, 5, figsize=(14, n * 2.6))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction",
                  "Epistemic (MI)", "Total Uncertainty"]

    for row_idx, item in enumerate(items):
        axes[row_idx, 0].imshow(item["input"])
        axes[row_idx, 1].imshow(item["gt"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(item["pred"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 3].imshow(item["epistemic"], cmap="hot", vmin=0)
        axes[row_idx, 4].imshow(item["total_unc"], cmap="hot", vmin=0)

        axes[row_idx, 0].set_ylabel(
            f"Dice={item['dice']:.3f}\nMI={item['mean_mi']:.4f}",
            fontsize=8, rotation=0, labelpad=55, va="center",
        )

    for col_idx, t in enumerate(col_titles):
        axes[0, col_idx].set_title(t, fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.07, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Failure case gallery")
    parser.add_argument("--checkpoint", type=str, default="src/best_model.pth")
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of worst cases to display")
    parser.add_argument("--corruption", type=str, default=None)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = get_unet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, Dice {ckpt['val_dice']:.4f}")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # --- Clean failures ---
    print("\n--- Computing per-image Dice on clean test set ---")
    clean_ds = ISICDataset(
        "test", data_root=args.data_root, splits_root=args.splits_root,
        image_size=args.image_size,
    )
    clean_results = compute_per_image_dice(model, clean_ds, device, args.mc_passes)
    clean_sorted = sorted(clean_results, key=lambda x: x["dice"])

    worst_clean = clean_sorted[:args.top_k]
    dices = [r["dice"] for r in clean_results]
    print(f"\nClean test set: mean Dice={np.mean(dices):.4f}, "
          f"worst={min(dices):.4f}, best={max(dices):.4f}")

    plot_gallery(
        worst_clean,
        f"Worst {args.top_k} Failures — Clean Test Set",
        reports / "failure_gallery_clean.png",
    )

    # --- Corrupted failures ---
    corruption = args.corruption or "gaussian_noise"
    sev = args.severity
    print(f"\n--- Computing per-image Dice on {corruption} severity={sev} ---")
    corrupt_ds = ISICDataset(
        "test", data_root=args.data_root, splits_root=args.splits_root,
        image_size=args.image_size,
        corruption_type=corruption, corruption_severity=sev,
    )
    corrupt_results = compute_per_image_dice(model, corrupt_ds, device, args.mc_passes)
    corrupt_sorted = sorted(corrupt_results, key=lambda x: x["dice"])

    worst_corrupt = corrupt_sorted[:args.top_k]
    dices_c = [r["dice"] for r in corrupt_results]
    print(f"\n{corruption} s={sev}: mean Dice={np.mean(dices_c):.4f}, "
          f"worst={min(dices_c):.4f}, best={max(dices_c):.4f}")

    plot_gallery(
        worst_corrupt,
        f"Worst {args.top_k} Failures — {corruption} (severity {sev})",
        reports / f"failure_gallery_{corruption}_s{sev}.png",
    )

    # --- Correlation summary ---
    mi_clean = [r["mean_mi"] for r in clean_results]
    corr = np.corrcoef(dices, mi_clean)[0, 1]
    print(f"\nCorrelation(Dice, MI) on clean set: r={corr:.4f}")
    if corr < -0.1:
        print("  -> Negative correlation: high uncertainty correlates with low Dice (good)")
    else:
        print("  -> Weak/positive correlation: uncertainty does not strongly predict failure")

    print("\nDone — failure galleries generated.")


if __name__ == "__main__":
    main()
