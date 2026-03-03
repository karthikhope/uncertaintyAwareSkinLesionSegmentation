"""
STEP 6 — Overfit Test (CRITICAL)
Train UNet on 20 synthetic images for 200 iterations.
Expected: loss drops sharply, dice approaches 0.9+.
"""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.unet import get_unet
from metrics.seg import dice_score


# ── Synthetic dataset with learnable shapes ──────────────────────────────
class SyntheticSegDataset(Dataset):
    """Generate images with random ellipses/rectangles and matching binary masks."""

    def __init__(self, n_samples=20, img_size=256, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.images = []
        self.masks = []

        for _ in range(n_samples):
            # Background: smooth random colour
            img = rng.uniform(0.1, 0.4, (3, img_size, img_size)).astype(np.float32)
            mask = np.zeros((1, img_size, img_size), dtype=np.float32)

            # Draw 1-3 bright shapes
            n_shapes = rng.randint(1, 4)
            for _ in range(n_shapes):
                cx, cy = rng.randint(40, img_size - 40, 2)
                rx, ry = rng.randint(20, 60, 2)
                yy, xx = np.ogrid[:img_size, :img_size]
                ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
                colour = rng.uniform(0.6, 1.0, (3, 1, 1)).astype(np.float32)
                img[:, ellipse] = colour[:, 0, 0][:, None] * np.ones_like(img[:, ellipse])
                mask[0, ellipse] = 1.0

            self.images.append(torch.from_numpy(img))
            self.masks.append(torch.from_numpy(mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ── Setup ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = get_unet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # higher LR for overfitting
bce = nn.BCEWithLogitsLoss()

dataset = SyntheticSegDataset(n_samples=20, img_size=256)
# batch_size=4 → 5 batches/epoch, 200 iters ≈ 40 epochs
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ── Sanity checks ───────────────────────────────────────────────────────
sample_img, sample_mask = dataset[0]
print(f"Image  shape: {sample_img.shape}  min={sample_img.min():.3f}  max={sample_img.max():.3f}")
print(f"Mask   shape: {sample_mask.shape}  unique={sample_mask.unique().tolist()}")
assert sample_mask.unique().tolist() == [0.0, 1.0] or sample_mask.unique().tolist() == [0.0] or sample_mask.unique().tolist() == [1.0], \
    f"Mask values must be 0/1, got {sample_mask.unique().tolist()}"
print("✓ Mask values are 0/1\n")

# ── Training loop ───────────────────────────────────────────────────────
iteration = 0
target_iters = 200
best_dice = 0.0

print(f"{'Iter':>5}  {'Loss':>8}  {'Dice':>8}")
print("-" * 28)

model.train()
while iteration < target_iters:
    for images, masks in loader:
        if iteration >= target_iters:
            break

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute dice every iteration for visibility
        with torch.no_grad():
            dice = dice_score(logits, masks).item()

        iteration += 1
        best_dice = max(best_dice, dice)

        if iteration <= 10 or iteration % 10 == 0 or iteration == target_iters:
            print(f"{iteration:>5}  {loss.item():>8.4f}  {dice:>8.4f}")

# ── Final evaluation on entire dataset ──────────────────────────────────
model.eval()
all_dice = []
all_loss = []
with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        all_loss.append(bce(logits, masks).item())
        all_dice.append(dice_score(logits, masks).item())

final_loss = np.mean(all_loss)
final_dice = np.mean(all_dice)

print(f"\n{'='*40}")
print(f"Final eval  —  Loss: {final_loss:.4f}  |  Dice: {final_dice:.4f}")
print(f"Best dice seen during training: {best_dice:.4f}")
print(f"{'='*40}")

if final_dice >= 0.9:
    print("\n✅ OVERFIT TEST PASSED — Dice ≥ 0.9")
    print("   Safe to proceed to full training.")
else:
    print("\n❌ OVERFIT TEST FAILED — Dice < 0.9")
    print("   Checklist:")
    print("   • Mask scaling: must be 0/1 (not 0/255)")
    print("   • Resizing: masks must use NEAREST interpolation")
    print("   • Normalization: images should be [0,1] or standardized")
    print("   DO NOT proceed until this passes.")
    sys.exit(1)
