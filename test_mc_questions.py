"""
Experiment: Answer MC Dropout diagnostic questions.
Q2: Does increasing dropout p from 0.3 → 0.5 increase MI?
Q3: Does reducing T from 20 → 5 make MI noisier?
Q4: Does MI concentrate around object borders?
"""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.unet import DropoutUnet
from utils import enable_dropout
from metrics.uncertainty import predictive_entropy, expected_entropy, mutual_information

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_test_image(seed=99):
    rng = np.random.RandomState(seed)
    img = rng.uniform(0.1, 0.4, (3, 256, 256)).astype(np.float32)
    mask_gt = np.zeros((256, 256), dtype=np.float32)
    for _ in range(2):
        cx, cy = rng.randint(40, 216, 2)
        rx, ry = rng.randint(20, 60, 2)
        yy, xx = np.ogrid[:256, :256]
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        colour = rng.uniform(0.6, 1.0, (3, 1, 1)).astype(np.float32)
        img[:, ellipse] = colour[:, 0, 0][:, None] * np.ones_like(img[:, ellipse])
        mask_gt[ellipse] = 1.0
    return img, mask_gt


def mc_inference(model, image, T=20):
    model.eval()
    enable_dropout(model)
    preds = []
    for _ in range(T):
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits)
            preds.append(probs)
    preds = torch.stack(preds)
    mean_pred = preds.mean(dim=0)
    return mean_pred, preds


# Load trained checkpoint (p=0.3)
model_03 = DropoutUnet(p=0.3).to(device)
ckpt = torch.load("src/best_model.pth", map_location=device)
model_03.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint (p=0.3) from epoch {ckpt['epoch']}")

# Create model with p=0.5, load same weights (dropout p doesn't affect state_dict)
model_05 = DropoutUnet(p=0.5).to(device)
model_05.load_state_dict(ckpt["model_state_dict"])
print("Created model with p=0.5 (same weights)\n")

img_clean, mask_gt = make_test_image(seed=99)
image = torch.from_numpy(img_clean).unsqueeze(0).to(device)

# ============================================================
# Q2: Dropout 0.3 vs 0.5 — does MI increase?
# ============================================================
print("=" * 55)
print("Q2: Dropout p=0.3 vs p=0.5 — does MI increase?")
print("=" * 55)

mean_03, samples_03 = mc_inference(model_03, image, T=20)
mi_03 = mutual_information(mean_03, samples_03)

mean_05, samples_05 = mc_inference(model_05, image, T=20)
mi_05 = mutual_information(mean_05, samples_05)

print(f"  p=0.3  →  MI mean = {mi_03.mean().item():.6f}   std(preds) = {samples_03.std().item():.6f}")
print(f"  p=0.5  →  MI mean = {mi_05.mean().item():.6f}   std(preds) = {samples_05.std().item():.6f}")
if mi_05.mean() > mi_03.mean():
    print("  ✅ YES — Higher dropout increases MI (more stochastic variation)")
else:
    print("  ❌ NO — MI did not increase")

# ============================================================
# Q3: T=20 vs T=5 — is MI noisier with fewer passes?
# ============================================================
print(f"\n{'=' * 55}")
print("Q3: T=20 vs T=5 — is MI noisier with fewer passes?")
print("=" * 55)

# Run T=5 multiple times to measure variance
mi_t5_runs = []
mi_t20_runs = []
n_trials = 5

for trial in range(n_trials):
    mean_t5, samp_t5 = mc_inference(model_03, image, T=5)
    mi_t5 = mutual_information(mean_t5, samp_t5).mean().item()
    mi_t5_runs.append(mi_t5)

    mean_t20, samp_t20 = mc_inference(model_03, image, T=20)
    mi_t20 = mutual_information(mean_t20, samp_t20).mean().item()
    mi_t20_runs.append(mi_t20)

mi_t5_arr = np.array(mi_t5_runs)
mi_t20_arr = np.array(mi_t20_runs)

print(f"  T=5   →  MI across {n_trials} trials: {mi_t5_arr.round(6)}")
print(f"           mean={mi_t5_arr.mean():.6f}  std={mi_t5_arr.std():.6f}")
print(f"  T=20  →  MI across {n_trials} trials: {mi_t20_arr.round(6)}")
print(f"           mean={mi_t20_arr.mean():.6f}  std={mi_t20_arr.std():.6f}")

if mi_t5_arr.std() > mi_t20_arr.std():
    print("  ✅ YES — T=5 produces noisier MI estimates (higher variance across runs)")
else:
    print("  ⚠️  Variance difference is small — may need more trials to confirm")

# ============================================================
# Q4: Does MI concentrate around object borders?
# ============================================================
print(f"\n{'=' * 55}")
print("Q4: Does MI concentrate around ambiguous borders?")
print("=" * 55)

# Compute border mask using morphological dilation - erosion
from scipy import ndimage

border_width = 10
dilated = ndimage.binary_dilation(mask_gt, iterations=border_width).astype(float)
eroded = ndimage.binary_erosion(mask_gt, iterations=border_width).astype(float)
border_mask = (dilated - eroded).astype(bool)
interior_mask = eroded.astype(bool)
exterior_mask = (~dilated.astype(bool)) & (~border_mask)

mi_map = mi_03[0, 0].cpu().numpy()

mi_border = mi_map[border_mask].mean() if border_mask.any() else 0
mi_interior = mi_map[interior_mask].mean() if interior_mask.any() else 0
mi_exterior = mi_map[exterior_mask].mean() if exterior_mask.any() else 0

print(f"  MI at border region:   {mi_border:.6f}  ({border_mask.sum()} pixels)")
print(f"  MI at interior:        {mi_interior:.6f}  ({interior_mask.sum()} pixels)")
print(f"  MI at exterior:        {mi_exterior:.6f}  ({exterior_mask.sum()} pixels)")

if mi_border > mi_interior and mi_border > mi_exterior:
    print("  ✅ YES — MI is highest at object borders (ambiguous regions)")
else:
    print("  ⚠️  MI distribution may be diffuse — check visualization")

# ── Save visualization for Q4 ────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_clean.transpose(1, 2, 0))
axes[0].set_title("Input Image")

axes[1].imshow(mask_gt, cmap="gray")
axes[1].set_title("Ground Truth")

axes[2].imshow(mi_map, cmap="hot")
axes[2].set_title("Epistemic (MI) Map")

axes[3].imshow(border_mask.astype(float), cmap="gray")
axes[3].contour(mask_gt, colors='cyan', linewidths=1)
axes[3].set_title(f"Border Region (±{border_width}px)")

for ax in axes:
    ax.axis("off")

plt.suptitle("Q4: Does MI concentrate at borders?", fontweight="bold")
plt.tight_layout()
plt.savefig("q4_mi_borders.png", dpi=150)
print(f"\n✓ Border analysis saved to q4_mi_borders.png")
