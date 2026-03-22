"""
Training loop for Bayesian U-Net on ISIC 2018 skin lesion segmentation.

Supports:
    - MPS (Apple Silicon), CUDA, and CPU devices
    - Real ISIC 2018 data or synthetic fallback
    - BCE + Dice combined loss
    - Checkpointing on best validation Dice

Usage:
    cd src
    python train.py                          # synthetic data (sanity check)
    python train.py --use_isic               # train on ISIC 2018
    python train.py --use_isic --epochs 100  # full training run
"""

import argparse
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from models.unet import get_unet, get_attention_unet, get_resunet
from metrics.seg import dice_score, iou_score

MODEL_REGISTRY = {
    "unet": get_unet,
    "attention_unet": get_attention_unet,
    "resunet": get_resunet,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train_one_epoch(model, loader, optimizer, bce_fn, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]",
        bar_format="{l_bar}{bar:20}{r_bar}", leave=False,
    )
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
            logits = model(images)
            loss = bce_fn(logits, masks) + dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

    return total_loss / len(loader)


def validate(model, loader, device, epoch, total_epochs):
    model.eval()
    total_dice, total_iou = 0.0, 0.0

    pbar = tqdm(
        loader, desc=f"Epoch {epoch+1}/{total_epochs} [val]  ",
        bar_format="{l_bar}{bar:20}{r_bar}", leave=False,
    )
    with torch.no_grad(), torch.autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
        for images, masks in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            logits = model(images)
            d = dice_score(logits, masks).item()
            total_dice += d
            total_iou += iou_score(logits, masks).item()
            pbar.set_postfix(dice=f"{d:.4f}")

    n = len(loader)
    return total_dice / n, total_iou / n


def make_synthetic_data():
    """Generate 20 synthetic images with random ellipses for sanity checking."""
    rng = np.random.RandomState(42)
    imgs, msks = [], []
    for _ in range(20):
        img = rng.uniform(0.1, 0.4, (3, 256, 256)).astype(np.float32)
        mask = np.zeros((1, 256, 256), dtype=np.float32)
        for _ in range(rng.randint(1, 4)):
            cx, cy = rng.randint(40, 216, 2)
            rx, ry = rng.randint(20, 60, 2)
            yy, xx = np.ogrid[:256, :256]
            ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            colour = rng.uniform(0.6, 1.0, (3, 1, 1)).astype(np.float32)
            img[:, ellipse] = colour[:, 0, 0][:, None] * np.ones_like(
                img[:, ellipse]
            )
            mask[0, ellipse] = 1.0
        imgs.append(torch.from_numpy(img))
        msks.append(torch.from_numpy(mask))

    all_imgs = torch.stack(imgs)
    all_msks = torch.stack(msks)
    train_ds = TensorDataset(all_imgs[:16], all_msks[:16])
    val_ds = TensorDataset(all_imgs[16:], all_msks[16:])
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian U-Net")
    parser.add_argument("--model", type=str, default="unet",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model variant: unet, attention_unet, resunet")
    parser.add_argument("--use_isic", action="store_true",
                        help="Train on real ISIC 2018 data instead of synthetic")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_root", type=str, default="../data/ISIC2018")
    parser.add_argument("--splits_root", type=str, default="../data/splits")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn().to(device)
    print(f"Model: {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_fn = nn.BCEWithLogitsLoss()

    if args.use_isic:
        from datasets.isic import ISICDataset

        train_ds = ISICDataset(
            "train", data_root=args.data_root, splits_root=args.splits_root,
            image_size=args.image_size, augment=True,
        )
        val_ds = ISICDataset(
            "val", data_root=args.data_root, splits_root=args.splits_root,
            image_size=args.image_size, augment=False,
        )
        print(f"ISIC 2018: {len(train_ds)} train, {len(val_ds)} val images")
    else:
        train_ds, val_ds = make_synthetic_data()
        print("Using synthetic data (20 images) for sanity check")

    num_workers = 2 if args.use_isic else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    best_dice = 0.0
    checkpoint_path = Path(f"best_model_{args.model}.pth")
    history_path = Path(f"training_history_{args.model}.csv")
    total_epochs = args.epochs
    history = []

    print(f"\n{'='*70}")
    print(f"  Training: {total_epochs} epochs | "
          f"Batch size: {args.batch_size} | "
          f"LR: {args.lr}")
    print(f"  Train batches/epoch: {len(train_loader)} | "
          f"Val batches/epoch: {len(val_loader)}")
    print(f"  Workers: {num_workers} | "
          f"Autocast: {device.type in ('cuda', 'mps')}")
    if device.type == "mps":
        print(f"  NOTE: First batch is slow (MPS JIT warmup). Speed improves after.")
    print(f"{'='*70}\n")

    for epoch in range(total_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, bce_fn, device, epoch, total_epochs,
        )
        val_dice, val_iou = validate(
            model, val_loader, device, epoch, total_epochs,
        )

        elapsed = time.time() - t0
        star = ""

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                str(checkpoint_path),
            )
            star = " *best*"

        print(
            f"Epoch {epoch+1:3d}/{total_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"{elapsed:.0f}s{star}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_dice": round(val_dice, 6),
            "val_iou": round(val_iou, 6),
            "time_s": round(elapsed, 1),
            "is_best": star != "",
        })

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\n{'='*70}")
    print(f"  Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"  Checkpoint: {checkpoint_path.resolve()}")
    print(f"  History:    {history_path.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
