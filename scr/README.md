# IPA-CycleGAN — Face Aging with Identity Preservation

A deep learning project that realistically ages faces across 8 age groups while preserving the person's identity.
Built with PyTorch · Trained on UTKFace · Evaluated on AgeDB-30

---

## Overview

Given a face image, the model generates a realistic aged version of that person while keeping their identity recognizable. It learns the structural and textural changes that happen across decades of aging, without needing paired before/after images of the same person.

**Input:** A face photo  
**Output:** The same person aged into a target age group (e.g., 20s → 60s)

---

## Architecture

This is a custom GAN architecture combining several components:

| Component | Role |
|---|---|
| **FaceEncoder** (ResNet50 backbone) | Separates facial content from identity |
| **AgeEmbedder** (MLP) | Learns an age-group style vector |
| **U-Net Generator** + AdaIN | Synthesizes aged face with style injection |
| **Attention Gates** | Focuses generation on age-relevant facial regions |
| **ArcFace** (frozen ResNet50) | Identity preservation loss |
| **Multi-Scale Discriminator** + Spectral Norm | Stable adversarial training |
| **CycleGAN loss** | Enables unpaired training |
| **Perceptual loss** (VGG16) | Preserves high-frequency texture |

The full loss combines: adversarial (WGAN-GP) + age classification + cycle-consistency + identity + perceptual + feature matching.

---

## Datasets

| Dataset | Purpose | Link |
|---|---|---|
| **UTKFace** | Training (~14,500 images across 8 age groups) | [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) |
| **AgeDB-30** | Evaluation (500 images) | [Kaggle](https://www.kaggle.com/datasets/nitingandhi/agedb-database) |

Images are grouped into 8 age ranges: 0–10, 11–20, 21–30, 31–40, 41–50, 51–60, 61–70, 71+.  
The training set was balanced across groups and deduplicated using perceptual hashing.

---

## Results

| Metric | Score |
|---|---|
| CSIM (Cosine Identity Similarity) | ~0.88 |
| FID (Fréchet Inception Distance) | ~51 |

CSIM of 0.88 indicates identity is well-preserved across age transformations. FID of ~51 reflects reasonable image quality given the training constraints.

---

## Limitations

This is a semester project built under real constraints:

- **Reduced training data** — UTKFace was downsampled per group due to time and GPU limitations. The full dataset would likely improve output quality.
- **Limited epochs** — training was run to ~70–100 epochs on free Colab GPU. The model was still improving at cutoff; more compute would reduce FID further.
- **128×128 resolution** — outputs can appear soft at larger display sizes. Higher resolution training requires more compute than was available.
- **Discrete age groups** — the model operates on 8 age buckets rather than continuous age values, so transitions between groups are not perfectly smooth.
- **Checkpoint not included** — trained weights are too large for GitHub. Re-train using the provided notebooks.
- **No paired training data** — UTKFace has no before/after pairs of the same person, so CycleGAN-style unpaired training is used.

---

## Setup

All notebooks run on Google Colab with a GPU runtime.

```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch timm tqdm imagehash
pip install torch-fidelity Pillow==10.2.0
```

Run in order: `PREPROCESSING` → `TRAINING` → `TESTING`

---

## About

Built as a semester project for a Bachelor of Science in Artificial Intelligence at Bahria University, Karachi.
