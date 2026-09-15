# Face Aging with Identity Preservation by using CycleGAN

**A deep learning project that ages faces realistically across 8 age groups while keeping the person's identity intact.** Built with PyTorch · Trained on UTKFace · Evaluated on AgeDB-30

## Overview

Given an input face image, the model produces a realistically aged version of that same person while preserving recognizable identity features. It learns the structural and textural patterns of aging across decades — without requiring paired before/after images of the same individual.

- **Input:** A face photo
- **Output:** The same person rendered in a target age group (e.g., 20s → 60s)

## Architecture

A custom GAN pipeline built from several components:

| Component | Role |
|---|---|
| FaceEncoder (ResNet50 backbone) | Disentangles facial content from identity |
| AgeEmbedder (MLP) | Learns a style vector representing the target age group |
| U-Net Generator + AdaIN | Generates the aged face via style injection |
| Attention Gates | Directs generation toward age-relevant facial regions |
| ArcFace (frozen ResNet50) | Enforces identity preservation |
| Multi-Scale Discriminator + Spectral Norm | Stabilizes adversarial training |
| CycleGAN loss | Enables training without paired data |
| Perceptual loss (VGG16) | Preserves fine texture detail |

Total objective: WGAN-GP adversarial loss + age classification loss + cycle-consistency loss + identity loss + perceptual loss + feature matching loss.

## Datasets

| Dataset | Purpose | Link |
|---|---|---|
| UTKFace | Training (~14,500 images across 8 age groups) | Kaggle |
| AgeDB-30 | Evaluation (500 images) | Kaggle |

Images are grouped into 8 age brackets: 0–10, 11–20, 21–30, 31–40, 41–50, 51–60, 61–70, 71+. The training set was balanced across groups and deduplicated via perceptual hashing.

## Results

| Metric | Score |
|---|---|
| CSIM (Cosine Identity Similarity) | ~0.88 |
| FID (Fréchet Inception Distance) | ~51 |

A CSIM of 0.88 suggests strong identity retention across age transformations. The FID of ~51 reflects reasonable visual fidelity given the training constraints below.

## Limitations

This was built as a semester project under real resource constraints:

- **Reduced training data** — UTKFace was downsampled per group due to time/GPU limits; the full dataset would likely improve output quality.
- **Limited training epochs** — trained for ~70–100 epochs on a free Colab GPU; the model was still improving when training was cut off.
- **128×128 resolution** — outputs can look soft at larger display sizes; higher resolutions would need more compute.
- **Discrete age buckets** — the model works on 8 age groups rather than continuous age, so transitions between groups aren't perfectly smooth.
- **No checkpoint included** — trained weights exceed GitHub's size limits; re-train using the provided notebooks.
- **Unpaired training** — UTKFace has no same-person before/after pairs, so CycleGAN-style unpaired training is used.

## Setup

All notebooks are designed to run on Google Colab with a GPU runtime.

```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch timm tqdm imagehash
pip install torch-fidelity Pillow==10.2.0
```

Run notebooks in order: `PREPROCESSING` → `TRAINING` → `TESTING`
