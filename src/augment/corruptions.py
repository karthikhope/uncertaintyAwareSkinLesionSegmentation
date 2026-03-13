"""
Image corruption engine for domain-shift experiments.

Implements 8 corruption types at 5 severity levels to simulate
smartphone-quality degradation of dermoscopic images.

Usage:
    from augment.corruptions import apply_corruption, CORRUPTION_TYPES

    corrupted = apply_corruption(image, "gaussian_blur", severity=3, seed=42)
"""

import io
import numpy as np
import cv2
from PIL import Image


SEVERITY_PARAMS = {
    "gaussian_blur":    {"sigma":   [1.0, 2.0, 3.0, 4.0, 5.0]},
    "motion_blur":      {"ksize":   [3, 5, 7, 9, 11]},
    "gaussian_noise":   {"std":     [0.02, 0.05, 0.10, 0.15, 0.20]},
    "speckle_noise":    {"std":     [0.02, 0.05, 0.10, 0.15, 0.20]},
    "jpeg_compression": {"quality": [80, 60, 40, 20, 10]},
    "brightness_shift": {"factor":  [0.1, 0.2, 0.3, 0.4, 0.5]},
    "contrast_shift":   {"factor":  [0.1, 0.2, 0.3, 0.4, 0.5]},
    "downscale":        {"factor":  [0.8, 0.6, 0.4, 0.3, 0.2]},
}

CORRUPTION_TYPES = list(SEVERITY_PARAMS.keys())


def _validate_inputs(image, severity):
    assert image.ndim == 3 and image.shape[2] == 3, \
        f"Expected HxWx3, got {image.shape}"
    assert 1 <= severity <= 5, f"Severity must be 1-5, got {severity}"


def apply_gaussian_blur(image, severity, seed=0):
    """Gaussian blur with increasing sigma."""
    _validate_inputs(image, severity)
    sigma = SEVERITY_PARAMS["gaussian_blur"]["sigma"][severity - 1]
    ksize = int(6 * sigma) | 1  # odd kernel size
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return np.clip(blurred, 0, 1).astype(np.float32)


def apply_motion_blur(image, severity, seed=0):
    """Horizontal motion blur with increasing kernel size."""
    _validate_inputs(image, severity)
    ksize = SEVERITY_PARAMS["motion_blur"]["ksize"][severity - 1]
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    blurred = cv2.filter2D(image, -1, kernel)
    return np.clip(blurred, 0, 1).astype(np.float32)


def apply_gaussian_noise(image, severity, seed=0):
    """Additive Gaussian noise with increasing standard deviation."""
    _validate_inputs(image, severity)
    rng = np.random.default_rng(seed)
    std = SEVERITY_PARAMS["gaussian_noise"]["std"][severity - 1]
    noise = rng.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1).astype(np.float32)


def apply_speckle_noise(image, severity, seed=0):
    """Multiplicative speckle noise: out = image + image * noise."""
    _validate_inputs(image, severity)
    rng = np.random.default_rng(seed)
    std = SEVERITY_PARAMS["speckle_noise"]["std"][severity - 1]
    noise = rng.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + image * noise, 0, 1).astype(np.float32)


def apply_jpeg_compression(image, severity, seed=0):
    """JPEG compression artifacts with decreasing quality."""
    _validate_inputs(image, severity)
    quality = SEVERITY_PARAMS["jpeg_compression"]["quality"][severity - 1]
    img_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer), dtype=np.float32) / 255.0
    return compressed


def apply_brightness_shift(image, severity, seed=0):
    """Random brightness increase/decrease."""
    _validate_inputs(image, severity)
    rng = np.random.default_rng(seed)
    factor = SEVERITY_PARAMS["brightness_shift"]["factor"][severity - 1]
    shift = rng.choice([-1, 1]) * factor
    return np.clip(image + shift, 0, 1).astype(np.float32)


def apply_contrast_shift(image, severity, seed=0):
    """Reduce contrast by pulling pixel values toward the mean."""
    _validate_inputs(image, severity)
    factor = SEVERITY_PARAMS["contrast_shift"]["factor"][severity - 1]
    mean_val = image.mean()
    adjusted = image * (1 - factor) + mean_val * factor
    return np.clip(adjusted, 0, 1).astype(np.float32)


def apply_downscale(image, severity, seed=0):
    """Downscale then upscale to simulate loss of fine detail."""
    _validate_inputs(image, severity)
    h, w = image.shape[:2]
    factor = SEVERITY_PARAMS["downscale"]["factor"][severity - 1]
    small_h, small_w = max(1, int(h * factor)), max(1, int(w * factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(restored, 0, 1).astype(np.float32)


_CORRUPTION_FN = {
    "gaussian_blur":    apply_gaussian_blur,
    "motion_blur":      apply_motion_blur,
    "gaussian_noise":   apply_gaussian_noise,
    "speckle_noise":    apply_speckle_noise,
    "jpeg_compression": apply_jpeg_compression,
    "brightness_shift": apply_brightness_shift,
    "contrast_shift":   apply_contrast_shift,
    "downscale":        apply_downscale,
}


def apply_corruption(image, corruption_type, severity, seed=0):
    """
    Apply a corruption to an image.

    Parameters
    ----------
    image : np.ndarray
        H x W x 3, float32 in [0, 1].
    corruption_type : str
        One of CORRUPTION_TYPES.
    severity : int
        1 (mild) to 5 (severe).
    seed : int
        For deterministic noise generation.

    Returns
    -------
    np.ndarray
        Corrupted image, same shape and range.
    """
    if corruption_type not in _CORRUPTION_FN:
        raise ValueError(
            f"Unknown corruption '{corruption_type}'. "
            f"Choose from: {CORRUPTION_TYPES}"
        )
    return _CORRUPTION_FN[corruption_type](image, severity, seed)
