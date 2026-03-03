import torch
from src.models.unet import get_unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet().to(device)

for i in range(5):
    dummy_images = torch.randn(4, 3, 256, 256).to(device)
    dummy_masks = torch.randint(0, 2, (4, 1, 256, 256)).float().to(device)

    logits = model(dummy_images)
    print(f'Batch {i+1}: logits.shape = {logits.shape}')

print(f'\nTotal images processed: 20')
print(f'Device: {device}')
