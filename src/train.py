import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import get_unet
from metrics.seg import dice_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_unet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = bce(logits, masks) + dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(loader):
    model.eval()
    total_dice = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            dice = dice_score(logits, masks)
            total_dice += dice.item()

    return total_dice / len(loader)


if __name__ == "__main__":
    # --- Replace these with your real DataLoaders ---
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # For sanity check: use synthetic data
    from torch.utils.data import TensorDataset
    import numpy as np

    np.random.seed(42)
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
            img[:, ellipse] = colour[:, 0, 0][:, None] * np.ones_like(img[:, ellipse])
            mask[0, ellipse] = 1.0
        imgs.append(torch.from_numpy(img))
        msks.append(torch.from_numpy(mask))

    all_imgs = torch.stack(imgs)
    all_msks = torch.stack(msks)
    train_dataset = TensorDataset(all_imgs[:16], all_msks[:16])
    val_dataset = TensorDataset(all_imgs[16:], all_msks[16:])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    num_epochs = 50
    best_dice = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader)
        val_dice = validate(val_loader)

        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                },
                "best_model.pth",
            )
            print("Checkpoint saved.")

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")