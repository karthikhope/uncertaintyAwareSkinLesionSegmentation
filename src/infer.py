import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without display)
import matplotlib.pyplot as plt

from models.unet import get_unet
from utils import enable_dropout
from metrics.uncertainty import predictive_entropy, expected_entropy, mutual_information

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_unet().to(device)

checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (Val Dice: {checkpoint['val_dice']:.4f})")


# ── MC Dropout Inference ─────────────────────────────────────────────────
def mc_inference(model, image, T=20):
    """
    Run T stochastic forward passes with dropout enabled.
    Returns mean prediction, variance, and all samples.
    """
    model.eval()
    enable_dropout(model)

    preds = []

    for _ in range(T):
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits)
            preds.append(probs)

    preds = torch.stack(preds)  # [T, B, 1, H, W]

    mean_pred = preds.mean(dim=0)
    var_pred = preds.var(dim=0)

    return mean_pred, var_pred, preds


# ── Standard (single-pass) inference ─────────────────────────────────────
def infer(image_tensor):
    """Run single deterministic inference on image tensor [1, 3, H, W]."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)

    return probs


# ── Synthetic test image generator ───────────────────────────────────────
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


if __name__ == "__main__":
    T = 20
    print(f"\n{'='*50}")
    print(f"MC Dropout Inference  (T={T} passes)")
    print(f"{'='*50}\n")

    # ── Clean image ──────────────────────────────────────────────────
    img_clean, mask_gt = make_test_image(seed=99)
    image_clean = torch.from_numpy(img_clean).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(mask_gt)

    mean_pred, var_pred, samples = mc_inference(model, image_clean, T=T)

    # STEP 7 — Debug check: variance across passes
    print(f"[DEBUG] preds.std() across {T} passes: {samples.std().item():.6f}")
    if samples.std().item() < 1e-6:
        print("⚠️  WARNING: std ≈ 0 → dropout may not be active!")
    else:
        print("✓ Dropout is active (non-zero variance across passes)\n")

    # STEP 4 — Compute uncertainty maps
    total_unc = predictive_entropy(mean_pred)
    exp_ent = expected_entropy(samples)
    epistemic_unc = mutual_information(mean_pred, samples)

    print(f"Mean pred   — min={mean_pred.min():.4f}  max={mean_pred.max():.4f}")
    print(f"Pred Entropy (total)    — mean={total_unc.mean():.4f}")
    print(f"Expected Entropy (alea) — mean={exp_ent.mean():.4f}")
    print(f"Mutual Info (epistemic) — mean={epistemic_unc.mean():.4f}")

    # ── Corrupted image (Gaussian blur + noise) ─────────────────────
    img_corrupt = img_clean.copy()
    # Add heavy Gaussian noise
    img_corrupt += np.random.RandomState(0).normal(0, 0.3, img_corrupt.shape).astype(np.float32)
    img_corrupt = np.clip(img_corrupt, 0, 1)
    image_corrupt = torch.from_numpy(img_corrupt).unsqueeze(0).to(device)

    mean_pred_c, var_pred_c, samples_c = mc_inference(model, image_corrupt, T=T)

    total_unc_c = predictive_entropy(mean_pred_c)
    exp_ent_c = expected_entropy(samples_c)
    epistemic_unc_c = mutual_information(mean_pred_c, samples_c)

    print(f"\n--- Corrupted image ---")
    print(f"Mean pred   — min={mean_pred_c.min():.4f}  max={mean_pred_c.max():.4f}")
    print(f"Pred Entropy (total)    — mean={total_unc_c.mean():.4f}")
    print(f"Expected Entropy (alea) — mean={exp_ent_c.mean():.4f}")
    print(f"Mutual Info (epistemic) — mean={epistemic_unc_c.mean():.4f}")

    # Check: epistemic uncertainty should increase under corruption
    mi_clean = epistemic_unc.mean().item()
    mi_corrupt = epistemic_unc_c.mean().item()
    print(f"\n📊 Epistemic (MI) clean={mi_clean:.6f}  corrupt={mi_corrupt:.6f}")
    if mi_corrupt > mi_clean:
        print("✅ Epistemic uncertainty INCREASED under corruption — expected behavior")
    else:
        print("⚠️  Epistemic uncertainty did NOT increase — check dropout / T value")

    # ── STEP 6 — Visual sanity check ────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 1: Clean image
    axes[0, 0].imshow(img_clean.transpose(1, 2, 0))
    axes[0, 0].set_title("Clean Input")

    axes[0, 1].imshow(mask_gt, cmap="gray")
    axes[0, 1].set_title("Ground Truth")

    axes[0, 2].imshow(mean_pred[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Mean Prediction")

    axes[0, 3].imshow(epistemic_unc[0, 0].cpu().numpy(), cmap="hot")
    axes[0, 3].set_title("Epistemic (MI)")

    axes[0, 4].imshow(total_unc[0, 0].cpu().numpy(), cmap="hot")
    axes[0, 4].set_title("Total Uncertainty")

    # Row 2: Corrupted image
    axes[1, 0].imshow(np.clip(img_corrupt.transpose(1, 2, 0), 0, 1))
    axes[1, 0].set_title("Corrupted Input")

    axes[1, 1].imshow(mask_gt, cmap="gray")
    axes[1, 1].set_title("Ground Truth")

    axes[1, 2].imshow(mean_pred_c[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Mean Prediction")

    axes[1, 3].imshow(epistemic_unc_c[0, 0].cpu().numpy(), cmap="hot")
    axes[1, 3].set_title("Epistemic (MI)")

    axes[1, 4].imshow(total_unc_c[0, 0].cpu().numpy(), cmap="hot")
    axes[1, 4].set_title("Total Uncertainty")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"MC Dropout Uncertainty (T={T})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mc_dropout_uncertainty.png", dpi=150)
    print(f"\n✓ Visual result saved to mc_dropout_uncertainty.png")

