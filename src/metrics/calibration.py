"""
Calibration metrics for binary segmentation: pixel-wise ECE, pECE,
and reliability diagram plotting.

Usage:
    from metrics.calibration import pixel_ece, per_class_ece, plot_reliability_diagram
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pixel_ece(all_probs, all_labels, n_bins=15):
    """
    Expected Calibration Error computed over all pixels.

    Parameters
    ----------
    all_probs : np.ndarray
        Flattened predicted probabilities, shape (N,).
    all_labels : np.ndarray
        Flattened ground-truth binary labels, shape (N,).
    n_bins : int
        Number of equally-spaced confidence bins.

    Returns
    -------
    ece : float
    bin_accuracies : np.ndarray of shape (n_bins,)
    bin_confidences : np.ndarray of shape (n_bins,)
    bin_counts : np.ndarray of shape (n_bins,)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (all_probs >= lo) & (all_probs <= hi)
        else:
            in_bin = (all_probs >= lo) & (all_probs < hi)

        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            bin_confidences[i] = all_probs[in_bin].mean()
            bin_accuracies[i] = all_labels[in_bin].mean()

    total_pixels = all_probs.shape[0]
    weights = bin_counts / max(total_pixels, 1)
    ece = (weights * np.abs(bin_accuracies - bin_confidences)).sum()

    return ece, bin_accuracies, bin_confidences, bin_counts


def per_class_ece(all_probs, all_labels, n_bins=15):
    """
    Per-class ECE (pECE): compute ECE separately for foreground (label=1)
    and background (label=0), then average. This avoids class imbalance
    masking poor foreground calibration.

    Returns
    -------
    pece : float
    ece_fg : float
    ece_bg : float
    """
    fg_mask = all_labels == 1
    bg_mask = all_labels == 0

    ece_fg = 0.0
    if fg_mask.sum() > 0:
        ece_fg, _, _, _ = pixel_ece(all_probs[fg_mask], all_labels[fg_mask], n_bins)

    ece_bg = 0.0
    if bg_mask.sum() > 0:
        ece_bg, _, _, _ = pixel_ece(all_probs[bg_mask], all_labels[bg_mask], n_bins)

    pece = (ece_fg + ece_bg) / 2.0
    return pece, ece_fg, ece_bg


def plot_reliability_diagram(
    bin_accuracies,
    bin_confidences,
    bin_counts,
    ece_value,
    title="Reliability Diagram",
    save_path=None,
):
    """
    Bar chart of per-bin accuracy vs confidence with a diagonal
    'perfect calibration' line and bin-count histogram below.
    """
    n_bins = len(bin_accuracies)
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    bar_width = 1.0 / n_bins

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_top.bar(
        bin_centers, bin_accuracies, width=bar_width * 0.9,
        color="steelblue", edgecolor="black", linewidth=0.5,
        label="Fraction of Positives", alpha=0.85,
    )
    ax_top.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax_top.set_ylabel("Fraction of Positives")
    ax_top.set_title(f"{title}  (ECE = {ece_value:.4f})")
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    ax_bot.bar(
        bin_centers, bin_counts, width=bar_width * 0.9,
        color="gray", edgecolor="black", linewidth=0.5, alpha=0.6,
    )
    ax_bot.set_xlabel("Confidence")
    ax_bot.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved to {save_path}")
    plt.close(fig)
    return fig