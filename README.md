# Uncertainty-Aware Skin Lesion Segmentation

> **Bridging the Clinical Gap:** A Bayesian U-Net that outputs both segmentation masks and reliability maps, flagging when it doesn't know enough to be trusted.

---

## Overview

Standard deep learning models for skin lesion segmentation are **overconfident** — they predict with 100% certainty even on blurry, out-of-distribution smartphone photos. In clinical settings, this silent failure is dangerous.

This project implements **Monte Carlo (MC) Dropout** on a U-Net architecture to estimate **epistemic uncertainty** at the pixel level. The system produces:

- A **segmentation mask** (lesion vs background)
- An **uncertainty map** highlighting regions where the model is unsure
- **Decomposed uncertainty** into epistemic (model) and aleatoric (data) components

When presented with corrupted or out-of-distribution images, the model's uncertainty visibly increases — enabling a clinical triage system that knows when to defer to a human expert.

---

## Architecture

```
Input Image (3 x 256 x 256)
        |
   [ResNet34 Encoder] ----skip connections--->
        |                                     |
   [Bottleneck + Dropout2d(p=0.3)]            |
        |                                     |
   [U-Net Decoder] <----concat----------------
        |
   [Segmentation Head] --> Logits (1 x 256 x 256)
```

At inference time, dropout remains **active**. Each of T=20 stochastic forward passes produces a slightly different prediction. The variance across passes reveals where the model is uncertain.

### Uncertainty Decomposition

| Metric | Formula | Captures |
|--------|---------|----------|
| Predictive Entropy | H(p-bar) | Total uncertainty |
| Expected Entropy | E[H(p_t)] | Aleatoric-like (data noise) |
| Mutual Information | H(p-bar) - E[H(p_t)] | Epistemic (model uncertainty) |

---

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── unet.py              # DropoutUnet (ResNet34 + MC Dropout)
│   ├── metrics/
│   │   ├── seg.py                # Dice score
│   │   └── uncertainty.py        # Predictive entropy, expected entropy, MI
│   ├── train.py                  # Training loop (BCE + Dice loss)
│   ├── infer.py                  # MC Dropout inference + visualization
│   └── utils.py                  # Dropout control utilities
├── test_unet.py                  # Forward pass shape tests
├── test_overfit.py               # Sanity check: overfit on synthetic data
├── test_mc_questions.py          # MC Dropout diagnostic experiments
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (Synthetic Sanity Check)

```bash
cd src
python train.py
```

Trains on 20 synthetic images with random ellipses. Saves `best_model.pth` when validation Dice improves.

### 3. Run MC Dropout Inference

```bash
cd src
python infer.py
```

Generates `mc_dropout_uncertainty.png` — a 2x5 grid comparing clean vs corrupted images across prediction, epistemic uncertainty, and total uncertainty channels.

### 4. Run Tests

```bash
python test_unet.py        # Verify forward pass and output shapes
python test_overfit.py      # Confirm model can overfit (Dice >= 0.9)
python test_mc_questions.py # MC Dropout experiments (Q2-Q4)
```

---

## Key Results

The inference pipeline demonstrates that:

- **Clean images:** Low epistemic uncertainty; predictions are consistent across MC passes
- **Corrupted images (Gaussian noise):** Epistemic uncertainty (mutual information) **increases** — the model correctly signals reduced confidence
- **Lesion borders:** Highest uncertainty concentration, matching the inherent ambiguity of boundary delineation

---

## Team Structure

| Role | Focus |
|------|-------|
| **Pavankumar Kulkarni** | Model zoo, MC Dropout pipeline, uncertainty decomposition, OOD triage |
| **V Karthikkumar** | Dataset loaders, corruption suite, calibration metrics (ECE/pECE), statistical tests, plots |

---

## Reference Papers

| Paper | Why It Matters |
|-------|----------------|
| [U-Net (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) | Base segmentation architecture |
| [MC Dropout (Gal & Ghahramani, 2016)](https://arxiv.org/abs/1506.02142) | Theoretical foundation for dropout-based uncertainty |
| [Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977) | Epistemic + aleatoric decomposition framework |

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** + torchvision
- **segmentation_models_pytorch** (ResNet34 U-Net encoder)
- **Matplotlib** (visualization)
- **NumPy** (data manipulation)

---

## Roadmap

- [x] Baseline U-Net with MC Dropout
- [x] Uncertainty metrics (entropy, MI)
- [x] Synthetic data training pipeline
- [x] Clean vs corrupted comparison
- [ ] ISIC 2018 dataset integration
- [ ] Corruption suite with severity ladder
- [ ] Pixel-wise ECE / calibration metrics
- [ ] Reliability diagrams
- [ ] Attention U-Net / ResUNet baselines
- [ ] Aleatoric uncertainty modeling
- [ ] Statistical significance tests
- [ ] Final report and presentation

---

## License

Academic project for IISc Applied AI in Healthcare course.

---

## Acknowledgments

Built upon [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) for the U-Net backbone and inspired by the MC Dropout framework of Gal & Ghahramani (2016).
