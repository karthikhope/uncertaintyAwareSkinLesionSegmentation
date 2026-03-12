import segmentation_models_pytorch as smp
import torch.nn as nn


class DropoutUnet(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        # Get features from encoder
        features = self.model.encoder(x)
        # Apply dropout on bottleneck features (512-ch, deepest layer)
        features[-1] = self.dropout(features[-1])
        # Decode
        decoder_output = self.model.decoder(features)
        logits = self.model.segmentation_head(decoder_output)
        return logits


def get_unet():
    return DropoutUnet(p=0.3)
