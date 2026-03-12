"""
ISIC 2018 Task 1 dataset loader for skin lesion segmentation.

Returns (image, mask) pairs as float32 tensors:
    image: [3, H, W] normalised with ImageNet mean/std
    mask:  [1, H, W] binary {0.0, 1.0}

Supports optional corruption loading for domain-shift experiments
(see Task 2.3 in the work plan).

Usage:
    from datasets.isic import ISICDataset

    train_ds = ISICDataset("train", data_root="data/ISIC2018",
                           splits_root="data/splits", image_size=256)
    val_ds   = ISICDataset("val",   ...)
    test_ds  = ISICDataset("test",  ...)
"""

import csv
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ISICDataset(Dataset):
    """
    Parameters
    ----------
    split : str
        One of "train", "val", "test".
    data_root : str or Path
        Path to data/ISIC2018/ containing images/ and masks/.
    splits_root : str or Path
        Path to data/splits/ containing train.csv, val.csv, test.csv.
    image_size : int
        Resize both image and mask to (image_size, image_size).
    augment : bool
        Apply training augmentations (flips, rotation). Only used when split="train".
    corruption_type : str or None
        If set, load corrupted images from data/ISIC2018/corrupted/{type}/severity_{s}/
    corruption_severity : int or None
        Severity level 1-5. Required if corruption_type is set.
    normalize : bool
        Apply ImageNet mean/std normalization. Default True.
    """

    def __init__(
        self,
        split,
        data_root="data/ISIC2018",
        splits_root="data/splits",
        image_size=256,
        augment=False,
        corruption_type=None,
        corruption_severity=None,
        normalize=True,
    ):
        self.data_root = Path(data_root)
        self.splits_root = Path(splits_root)
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        self.normalize = normalize

        self.image_ids = self._load_split(split)

    def _load_split(self, split):
        csv_path = self.splits_root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {csv_path}. Run src/datasets/split.py first."
            )
        ids = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(row["image_id"])
        return ids

    def _get_image_path(self, image_id):
        if self.corruption_type and self.corruption_severity:
            return (
                self.data_root
                / "corrupted"
                / self.corruption_type
                / f"severity_{self.corruption_severity}"
                / f"{image_id}.jpg"
            )
        return self.data_root / "images" / f"{image_id}.jpg"

    def _get_mask_path(self, image_id):
        return self.data_root / "masks" / f"{image_id}_segmentation.png"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img = Image.open(self._get_image_path(image_id)).convert("RGB")
        mask = Image.open(self._get_mask_path(image_id)).convert("L")

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0   # H x W x 3, [0,1]
        mask = np.array(mask, dtype=np.float32) / 255.0  # H x W, [0,1]
        mask = (mask > 0.5).astype(np.float32)            # binarise

        if self.augment:
            img, mask = self._apply_augmentations(img, mask)

        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

        img = torch.from_numpy(img.transpose(2, 0, 1))          # [3, H, W]
        mask = torch.from_numpy(mask[np.newaxis, :, :])          # [1, H, W]

        return img, mask

    def _apply_augmentations(self, img, mask):
        """Random flips and 90-degree rotations (numpy-only, no extra deps)."""
        rng = np.random.default_rng()

        if rng.random() > 0.5:
            img = np.flip(img, axis=1).copy()   # horizontal flip
            mask = np.flip(mask, axis=1).copy()

        if rng.random() > 0.5:
            img = np.flip(img, axis=0).copy()   # vertical flip
            mask = np.flip(mask, axis=0).copy()

        k = rng.integers(0, 4)  # 0, 90, 180, 270 degrees
        if k > 0:
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return img, mask
