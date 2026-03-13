import segmentation_models_pytorch as smp
import torch.nn as nn


class DropoutUnet(nn.Module):
    def __init__(self, p=0.3, encoder_name="resnet34",
                 decoder_attention_type=None):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_attention_type=decoder_attention_type,
        )

        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        features = self.model.encoder(x)
        features[-1] = self.dropout(features[-1])
        decoder_output = self.model.decoder(features)
        logits = self.model.segmentation_head(decoder_output)
        return logits


def get_unet(encoder_name="resnet34", decoder_attention_type=None, p=0.3):
    return DropoutUnet(p=p, encoder_name=encoder_name,
                       decoder_attention_type=decoder_attention_type)


def get_attention_unet(p=0.3):
    return get_unet(encoder_name="resnet34", decoder_attention_type="scse", p=p)


def get_resunet(p=0.3):
    return get_unet(encoder_name="resnet50", p=p)
