# Multi-Model Comparison Report

## Clean Test Set Performance

| Model | Dice | IoU | ECE | pECE | MI |
|-------|------|-----|-----|------|----|
| U-Net (ResNet34) | 0.8940 | 0.8266 | 0.0242 | 0.0742 | 0.0007 |
| Attention U-Net | 0.9007 | 0.8348 | 0.0187 | 0.0642 | 0.0006 |
| ResUNet (ResNet50) | 0.8979 | 0.8306 | 0.0202 | 0.0717 | 0.0003 |

## Dice at Severity 5 (All Corruptions)

| Corruption | U-Net (ResNet34) | Attention U-Net | ResUNet (ResNet50) | Best |
|------------|------|------|------|------|
| Gaussian Blur | 0.8272 | 0.8498 | 0.8492 | Attention U-Net |
| Motion Blur | 0.8694 | 0.8776 | 0.8701 | Attention U-Net |
| Gaussian Noise | 0.0017 | 0.1761 | 0.0001 | Attention U-Net |
| Speckle Noise | 0.4664 | 0.6636 | 0.4350 | Attention U-Net |
| JPEG Compression | 0.8348 | 0.8622 | 0.8483 | Attention U-Net |
| Brightness Shift | 0.6173 | 0.6875 | 0.6316 | Attention U-Net |
| Contrast Shift | 0.8742 | 0.8772 | 0.8829 | ResUNet (ResNet50) |
| Downscale | 0.8739 | 0.8649 | 0.8586 | U-Net (ResNet34) |

## Mean Dice Drop (Clean → Severity 5)

| Model | Clean Dice | Mean S5 Dice | Mean Drop | Relative Drop |
|-------|-----------|-------------|-----------|---------------|
| U-Net (ResNet34) | 0.8940 | 0.6706 | 0.2234 | 25.0% |
| Attention U-Net | 0.9007 | 0.7324 | 0.1683 | 18.7% |
| ResUNet (ResNet50) | 0.8979 | 0.6720 | 0.2259 | 25.2% |

## ECE at Severity 5

| Corruption | U-Net (ResNet34) | Attention U-Net | ResUNet (ResNet50) | Best |
|------------|------|------|------|------|
| Gaussian Blur | 0.0446 | 0.0370 | 0.0331 | ResUNet (ResNet50) |
| Motion Blur | 0.0336 | 0.0339 | 0.0273 | ResUNet (ResNet50) |
| Gaussian Noise | 0.2248 | 0.1965 | 0.2250 | Attention U-Net |
| Speckle Noise | 0.1570 | 0.0982 | 0.1505 | Attention U-Net |
| JPEG Compression | 0.0386 | 0.0355 | 0.0372 | Attention U-Net |
| Brightness Shift | 0.0903 | 0.0874 | 0.0983 | Attention U-Net |
| Contrast Shift | 0.0275 | 0.0248 | 0.0233 | ResUNet (ResNet50) |
| Downscale | 0.0388 | 0.0562 | 0.0447 | U-Net (ResNet34) |

## Key Findings

1. **Best clean accuracy:** Attention U-Net (Dice 0.9007)
2. **Best calibration:** Attention U-Net (ECE 0.0187)
3. **Most robust under corruption:** Attention U-Net (mean severity-5 Dice 0.7324)
4. **All models fail on Gaussian noise s5** — this is a shared vulnerability from training on clean data only

