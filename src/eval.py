"""
Central evaluation script: evaluates a model checkpoint on clean and all
corrupted test sets, computing Dice, IoU, ECE, pECE, and mean MI.

Saves results to a structured CSV and generates reliability diagrams.

Usage:
    cd src
    python eval.py --checkpoint best_model.pth
    python eval.py --checkpoint best_model.pth --mc_passes 20
    python eval.py --checkpoint best_model.pth --mc_passes 20 --use_isic
"""

import argparse
import csv
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models.unet import get_unet, get_attention_unet, get_resunet
from metrics.seg import dice_score, iou_score
from metrics.uncertainty import predictive_entropy, mutual_information
from metrics.calibration import pixel_ece, per_class_ece, plot_reliability_diagram
from utils import enable_dropout


MODEL_REGISTRY = {
    "unet": get_unet,
    "attention_unet": get_attention_unet,
    "resunet": get_resunet,
}

CORRUPTION_TYPES = [
    "gaussian_blur", "motion_blur", "gaussian_noise", "speckle_noise",
    "jpeg_compression", "brightness_shift", "contrast_shift", "downscale",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mc_inference(model, images, T, device):
    """Run T stochastic forward passes and return mean prediction + samples."""
    model.eval()
    enable_dropout(model)

    preds = []
    for _ in range(T):
        with torch.no_grad():
            logits = model(images.to(device))
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu())

    samples = torch.stack(preds)          # [T, B, 1, H, W]
    mean_pred = samples.mean(dim=0)       # [B, 1, H, W]
    return mean_pred, samples


def evaluate_loader(model, loader, device, mc_passes=0):
    """
    Evaluate model on a DataLoader.

    Returns dict with dice, iou, ece, pece, mean_mi.
    """
    all_dice, all_iou = [], []
    all_probs, all_labels = [], []
    all_mi = []

    use_mc = mc_passes > 0

    model.eval()
    if use_mc:
        enable_dropout(model)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        if use_mc:
            mean_pred, samples = mc_inference(model, images, mc_passes, device)
            mean_pred = mean_pred.to(device)
            logits_for_dice = torch.logit(mean_pred.clamp(1e-6, 1 - 1e-6))

            mi = mutual_information(mean_pred, samples.to(device))
            all_mi.append(mi.mean().item())
        else:
            with torch.no_grad():
                logits_for_dice = model(images)
            mean_pred = torch.sigmoid(logits_for_dice)

        all_dice.append(dice_score(logits_for_dice, masks).item())
        all_iou.append(iou_score(logits_for_dice, masks).item())

        probs_np = mean_pred.detach().cpu().numpy().flatten()
        labels_np = masks.detach().cpu().numpy().flatten()
        all_probs.append(probs_np)
        all_labels.append(labels_np)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    ece_val, bin_acc, bin_conf, bin_counts = pixel_ece(all_probs, all_labels)
    pece_val, _, _ = per_class_ece(all_probs, all_labels)

    results = {
        "dice": np.mean(all_dice),
        "iou": np.mean(all_iou),
        "ece": ece_val,
        "pece": pece_val,
        "mean_mi": np.mean(all_mi) if all_mi else 0.0,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_counts": bin_counts,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian U-Net")
    parser.add_argument("--model", type=str, default="unet",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture: unet, attention_unet, resunet")
    parser.add_argument("--checkpoint", type=str, default="best_model_unet.pth")
    parser.add_argument("--data_root", type=str, default="../data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="../data/splits")
    parser.add_argument("--mc_passes", type=int, default=20,
                        help="0 for deterministic, >0 for MC Dropout")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--use_isic", action="store_true")
    parser.add_argument("--model_name", type=str, default="bayesian_unet")
    parser.add_argument("--reports_dir", type=str, default="../reports")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn().to(device)
    print(f"Model architecture: {args.model}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, Dice {ckpt['val_dice']:.4f}")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    results_csv = reports / "eval_results.csv"

    if not args.use_isic:
        print("Evaluation requires --use_isic for meaningful results.")
        sys.exit(1)

    from datasets.isic import ISICDataset

    rows = []

    def eval_setting(corruption=None, severity=None):
        ds = ISICDataset(
            "test", data_root=args.data_root, splits_root=args.splits_root,
            image_size=args.image_size, corruption_type=corruption,
            corruption_severity=severity,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        label = corruption or "clean"
        sev = severity or 0
        print(f"  Evaluating {label} severity={sev} ({len(ds)} images)...", end=" ")

        res = evaluate_loader(model, loader, device, mc_passes=args.mc_passes)

        print(f"Dice={res['dice']:.4f}  IoU={res['iou']:.4f}  "
              f"ECE={res['ece']:.4f}  pECE={res['pece']:.4f}  MI={res['mean_mi']:.4f}")

        row = {
            "model": args.model_name,
            "corruption": label,
            "severity": sev,
            "dice": round(res["dice"], 4),
            "iou": round(res["iou"], 4),
            "ece": round(res["ece"], 4),
            "pece": round(res["pece"], 4),
            "mean_mi": round(res["mean_mi"], 4),
        }
        rows.append(row)

        save_diagram = (
            corruption is None
            or severity in (3, 5)
        )
        if save_diagram:
            tag = f"{label}_s{sev}" if corruption else "clean"
            plot_reliability_diagram(
                res["bin_acc"], res["bin_conf"], res["bin_counts"],
                res["ece"],
                title=f"{args.model_name} — {tag}",
                save_path=str(reports / f"reliability_{tag}.png"),
            )

    print("\n--- Clean test set ---")
    eval_setting()

    print("\n--- Corrupted test sets ---")
    for corruption in CORRUPTION_TYPES:
        print(f"\n[{corruption}]")
        for severity in range(1, 6):
            eval_setting(corruption, severity)

    with open(results_csv, "w", newline="") as f:
        fieldnames = ["model", "corruption", "severity", "dice", "iou", "ece", "pece", "mean_mi"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {results_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
