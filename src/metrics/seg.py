import torch

def dice_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()