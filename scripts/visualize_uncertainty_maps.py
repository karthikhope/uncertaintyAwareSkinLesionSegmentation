"""
Task 6.1 — Uncertainty map grid: clean vs corrupted on real ISIC images.

For a set of sample test images, generates a grid showing:
  Row per image: Input | Ground Truth | Prediction | Epistemic (MI) | Total Uncertainty

Produces two grids side-by-side for clean and one selected corruption,
so the reader can visually see how uncertainty increases under corruption.

Usage:
    python scripts/visualize_uncertainty_maps.py
    python scripts/visualize_uncertainty_maps.py --checkpoint src/best_model.pth
    python scripts/visualize_uncertainty_maps.py --corruption gaussian_noise --severity 5
    python scripts/visualize_uncertainty_maps.py --n_samples 8
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

    samples = torch.stack(preds)       # [T, 1, 1, H, W]
    mean_pred = samples.mean(dim=0)    # [1, 1, H, W]
    return mean_pred, samples


def denormalize(img_tensor):
    """Undo ImageNet normalization and convert to HWC numpy for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def build_grid(model, dataset, indices, device, mc_passes):
    """Compute predictions and uncertainty maps for given image indices."""
    rows = []
    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0)

        mean_pred, samples = mc_inference(model, image_batch, mc_passes, device)

        total_unc = predictive_entropy(mean_pred)
        epistemic = mutual_information(mean_pred, samples.to(mean_pred.device))

        rows.append({
            "input": denormalize(image),
            "gt": mask[0].numpy(),
            "pred": (mean_pred[0, 0] > 0.5).float().numpy(),
            "epistemic": epistemic[0, 0].numpy(),
            "total_unc": total_unc[0, 0].numpy(),
        })
    return rows


def plot_grid(rows_clean, rows_corrupt, corruption_label, save_path):
    """Plot a side-by-side grid: clean (left block) vs corrupted (right block)."""
    n = len(rows_clean)
    cols_per_block = 5  # input, GT, pred, epistemic, total
    fig, axes = plt.subplots(
        n, cols_per_block * 2,
        figsize=(cols_per_block * 2 * 2.4, n * 2.4),
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction",
                  "Epistemic (MI)", "Total Uncertainty"]

    for row_idx in range(n):
        for block, rows in enumerate([rows_clean, rows_corrupt]):
            r = rows[row_idx]
            offset = block * cols_per_block

            axes[row_idx, offset + 0].imshow(r["input"])
            axes[row_idx, offset + 1].imshow(r["gt"], cmap="gray", vmin=0, vmax=1)
            axes[row_idx, offset + 2].imshow(r["pred"], cmap="gray", vmin=0, vmax=1)
            axes[row_idx, offset + 3].imshow(r["epistemic"], cmap="hot", vmin=0)
            axes[row_idx, offset + 4].imshow(r["total_unc"], cmap="hot", vmin=0)

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9)
        axes[0, col_idx + cols_per_block].set_title(title, fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    mid_clean = cols_per_block / 2
    mid_corrupt = cols_per_block + cols_per_block / 2
    fig.text(mid_clean / (cols_per_block * 2), 1.01, "Clean",
             ha="center", fontsize=13, fontweight="bold",
             transform=fig.transFigure)
    fig.text(mid_corrupt / (cols_per_block * 2), 1.01, corruption_label,
             ha="center", fontsize=13, fontweight="bold",
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate uncertainty map grids (clean vs corrupted)")
    parser.add_argument("--checkpoint", type=str, default="src/best_model.pth")
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="data/splits")
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--corruption", type=str, default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of test images to include in the grid")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = get_unet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, Dice {ckpt['val_dice']:.4f}")

    clean_ds = ISICDataset(
        "test", data_root=args.data_root, splits_root=args.splits_root,
        image_size=args.image_size,
    )
    corrupt_ds = ISICDataset(
        "test", data_root=args.data_root, splits_root=args.splits_root,
        image_size=args.image_size,
        corruption_type=args.corruption,
        corruption_severity=args.severity,
    )

    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(len(clean_ds), size=args.n_samples, replace=False))
    print(f"Selected test indices: {indices}")

    print(f"\nRunning MC Dropout (T={args.mc_passes}) on {args.n_samples} clean images...")
    rows_clean = build_grid(model, clean_ds, indices, device, args.mc_passes)

    corr_label = f"{args.corruption} (severity {args.severity})"
    print(f"Running MC Dropout (T={args.mc_passes}) on {args.n_samples} {corr_label} images...")
    rows_corrupt = build_grid(model, corrupt_ds, indices, device, args.mc_passes)

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    save_path = reports / f"uncertainty_grid_{args.corruption}_s{args.severity}.png"

    plot_grid(rows_clean, rows_corrupt, corr_label, save_path)

    # Also generate grids for a few other key corruptions
    extra_corruptions = [
        ("gaussian_blur", 5),
        ("contrast_shift", 5),
        ("motion_blur", 5),
    ]
    for corr_type, sev in extra_corruptions:
        if corr_type == args.corruption and sev == args.severity:
            continue

        extra_ds = ISICDataset(
            "test", data_root=args.data_root, splits_root=args.splits_root,
            image_size=args.image_size,
            corruption_type=corr_type, corruption_severity=sev,
        )
        label = f"{corr_type} (severity {sev})"
        print(f"\nRunning MC Dropout (T={args.mc_passes}) on {args.n_samples} {label} images...")
        rows_extra = build_grid(model, extra_ds, indices, device, args.mc_passes)

        extra_path = reports / f"uncertainty_grid_{corr_type}_s{sev}.png"
        plot_grid(rows_clean, rows_extra, label, extra_path)

    print("\nDone — all uncertainty grids generated.")


if __name__ == "__main__":
    main()
