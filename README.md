---
title: "From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST"
layout: single
author_profile: true
---

<p align="center">
  <img src="images/bloodmnist-mlp-cnn.png"
       width="960"
       alt="BloodMNIST: MLP and CNN architecture comparison"/>
</p>

# From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST

**An educational empirical study of biomedical image classification, spatial feature learning, and classifier-head parameter efficiency.**

**PyTorch · BloodMNIST / MedMNIST v2 · MLP · CNN · Global Average Pooling**

[Jupyter notebook](notebooks/bloodmnist-mlp-cnn-uyentran-study-v7.ipynb) · [Technical report (PDF)](reports/bloodmnist_technical_report.pdf) · [M1 learning curves](figures/bloodmnist_M1_learning_curves.png) · [M2 confusion matrix](figures/bloodmnist_M2_confusion_matrix.png)

## Overview

This project compares a multilayer perceptron (MLP) baseline with two small convolutional neural networks (CNNs) for eight-class blood cell image classification using **BloodMNIST**. The goal is to understand how convolutional feature extraction and the choice of classifier head relate to classification performance and trainable parameter counts.

The study uses three predefined models, one random seed, and the same dataset splits and training settings. It is an **educational, single-seed comparison**, not a multi-seed benchmark or a clinically validated diagnostic system.

## Research Questions

**RQ1 — MLP versus CNN.** Under the same official dataset splits and shared training settings, how do the MLP and CNN architectures compare in validation and test classification performance?

**RQ2 — Flatten versus GAP.** With the same CNN feature-extractor design, how does replacing a Flatten classifier head with Global Average Pooling (GAP) affect trainable parameters and classification performance?

## Model Architectures

All models accept RGB images of shape `(B, 3, 28, 28)` and produce eight **raw logits** of shape `(B, 8)`.

| Model | Architecture | Trainable parameters |
|---|---|---:|
| **M0 — MLP** | Flatten → Linear(2352, 256) → ReLU → Dropout → Linear(256, 128) → ReLU → Dropout → Linear(128, 8) | 636,296 |
| **M1 — CNN + BN + Flatten** | Two Conv–BatchNorm–ReLU–MaxPool blocks (32, 64 channels) → Flatten → Dropout → Linear(3136, 8) | 44,680 |
| **M2 — CNN + BN + GAP** | Same two-block CNN design → Global Average Pooling → Flatten → Dropout → Linear(64, 8) | 20,104 |

M1 and M2 have the **same feature-extractor architecture**, but they are initialized and trained independently; they do not share trained weights. Their final feature maps have shape `(B, 64, 7, 7)`. M1 flattens these maps into 3,136 features; M2 averages each channel into one value, leaving 64 features. This reduces M2's total trainable parameter count by **24,576 (55.0%)** relative to M1.

The hidden MLP Linear weights and CNN convolutional weights use He/Kaiming Uniform initialization for ReLU. The output Linear layers retain PyTorch's default initialization. The CNNs use Batch Normalization; M0 does not.

## Dataset and Preprocessing

[BloodMNIST](https://medmnist.com/) contains **17,092 RGB images** at **28 × 28** resolution across eight blood cell classes. The official splits are used without additional splitting:

| Split | Images | Role |
|---|---:|---|
| Training | 11,959 | Update model parameters |
| Validation | 1,712 | Track performance and select each best checkpoint |
| Test | 3,421 | Final evaluation after checkpoint selection |

Images are converted using `transforms.ToTensor()`, which scales the original pixel values to `[0, 1]`. Per-channel RGB means and standard deviations are then calculated **from the training split only**:

| Statistic | Red | Green | Blue |
|---|---:|---:|---:|
| Mean | 0.794348 | 0.659659 | 0.696191 |
| Standard deviation | 0.215630 | 0.241603 | 0.117889 |

The same `transforms.Normalize(mean, std)` values are applied to training, validation, and test images. No data augmentation or resizing is used. Computing normalization statistics from training data only avoids validation/test leakage.

## Experimental Protocol

| Setting | Value |
|---|---|
| Official models | M0, M1, M2 — one training run each |
| Random seed | 42, reset before each model |
| Optimizer | Adam |
| Learning rate | 0.001, fixed for all models |
| Loss | Unweighted `nn.CrossEntropyLoss()` |
| Batch size | 128 |
| Training | 12 full epochs per model; no early stopping |
| Initialization | He/Kaiming Uniform for designated hidden Linear and Conv weights |
| Regularization | Dropout 0.2; BatchNorm in M1/M2 |
| Model selection | Highest validation macro-F1; retain earlier epoch on a tie |
| Final test metrics | Macro-F1 (primary) and accuracy (secondary) |

Each run starts with a fresh model, Adam optimizer, and seed-controlled shuffled training DataLoader. The best model `state_dict` is saved using validation macro-F1. The official test set is not used for training, tuning, or checkpoint selection.

## Results

| Model | Parameters | Best epoch | Best val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| **M0 — MLP** | 636,296 | 12 | 0.8366 | 0.8381 | 0.8150 |
| **M1 — CNN + Flatten** | 44,680 | 7 | **0.9156** | **0.9152** | **0.9051** |
| **M2 — CNN + GAP** | 20,104 | 11 | 0.8596 | 0.8702 | 0.8452 |

**RQ1 — MLP versus CNN.** Both CNNs achieved higher test macro-F1 than M0 under the tested protocol. M1 exceeded M0 by **9.01 percentage points**, and M2 exceeded M0 by **3.02 percentage points**. Because these models differ in more than the use of convolution (including BatchNorm and model capacity), the comparison does not isolate the causal effect of convolution.

**RQ2 — Flatten versus GAP.** M1 achieved higher test macro-F1 than M2 (**0.9051 versus 0.8452**), while M2 used **55.0% fewer trainable parameters**. This is an observed performance–parameter trade-off in the single-seed experiment, not evidence that Flatten is always better than GAP. Parameter reduction also does not establish an equivalent reduction in runtime or computational cost.

### Learning Curves and Error Analysis

Training histories and learning curves for all three models are generated in the notebook. The best validation checkpoints were selected at epochs 12 (M0), 7 (M1), and 11 (M2). See the [M1 learning curves](figures/bloodmnist_M1_learning_curves.png) for an example of the recorded training/validation loss and validation macro-F1 trajectories.

A predefined [confusion-matrix analysis of M2](figures/bloodmnist_M2_confusion_matrix.png) found **2,977 correct** and **444 incorrect** predictions among 3,421 test images. Monocytes had the lowest class recall (**0.5739**); **114 of 284** true monocytes were predicted as immature granulocytes. The confusion matrix describes these errors but does not establish why they occur.

## Interpretation and Limitations

The results suggest that the tested CNN architectures are more effective than the tested MLP baseline for this BloodMNIST task. M1 produced the strongest classification scores; M2 provided a smaller parameter count with lower macro-F1. The eight-class evaluation also shows why macro-F1 is useful alongside overall accuracy when class-level performance differs.

The main limitations are:

- **Single seed:** One run per model does not measure seed-to-seed variability or establish statistical significance.
- **Architectural differences:** M0 differs from the CNNs in BatchNorm, capacity, and feature extraction, so the effect of convolution alone cannot be isolated.
- **Image resolution:** The 28 × 28 images may not retain fine cellular details or represent higher-resolution imaging tasks.
- **Fixed settings:** A common learning rate and 12-epoch budget support consistent comparison but may not be optimal for each architecture.
- **Clinical applicability:** Results on this benchmark do not demonstrate reliability on real-world clinical data or suitability for diagnosis.

These are limitations of the study's scope, not claims about the universal ranking of MLP, Flatten, or GAP architectures.

## Reproducibility

The notebook uses standard Python/PyTorch libraries: `torch`, `torchvision`, `medmnist`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. A compatible Jupyter or VS Code environment is needed to run it.

```bash
# Clone the project
git clone https://github.com/uyentran9/bloodmnist-mlp-cnn.git
cd bloodmnist-mlp-cnn

# Create the local data and results folders
mkdir -p data results

# Install the main Python dependencies
python -m pip install torch torchvision medmnist numpy pandas matplotlib scikit-learn notebook
```

Keep the notebook in `notebooks/` and create `data/` and `results/` at the project root (as shown above). The notebook's `find_project_root()` function locates a parent directory containing **both `notebooks/` and `data/`**. BloodMNIST is downloaded by `medmnist` when needed; dataset files are not redistributed in this repository.

Open the [Jupyter notebook](notebooks/bloodmnist-mlp-cnn-uyentran-study-v7.ipynb) and execute **Parts 0–6 in order** with a fresh kernel. A full run recreates three independently trained models, their best validation-selected checkpoints, histories, learning curves, and the final test analysis. Hardware backends (CPU, Apple Silicon MPS, NVIDIA CUDA) and dependency versions can affect exact floating-point results.

| Notebook part | Purpose |
|---|---|
| Part 0 | Imports, device selection, fixed settings, and output paths |
| Part 1 | BloodMNIST splits, EDA, training-derived RGB normalization, and DataLoaders |
| Part 2 | M0/M1/M2 architectures, initialization, and forward checks |
| Part 3 | Training, validation, and checkpoint-selection functions |
| Part 4 | Three official runs, histories, and learning curves |
| Part 5 | Held-out test evaluation and M2 confusion matrix |
| Part 6 | Research question answers, limitations, and conclusion |

## Research Materials

| Resource | Description |
|---|---|
| [Jupyter notebook](notebooks/bloodmnist-mlp-cnn-uyentran-study-v7.ipynb) | Complete implementation, saved outputs, and explanations |
| [Technical report (PDF)](reports/bloodmnist_technical_report.pdf) | Three-page summary of methods, results, and limitations |
| [M1 learning curves](figures/bloodmnist_M1_learning_curves.png) | Training/validation loss and validation macro-F1 |
| [M2 confusion matrix](figures/bloodmnist_M2_confusion_matrix.png) | Eight-class error analysis of the selected M2 model |
| [BloodMNIST / MedMNIST](https://medmnist.com/) | Dataset source and benchmark documentation |

## References and Attribution

BloodMNIST is based on the peripheral blood-cell image dataset of **Acevedo et al. (2020)** and distributed through the **MedMNIST v2** benchmark (**Yang et al., 2023**). See the [MedMNIST website](https://medmnist.com/) and the technical report for dataset provenance and references.

**Author:** Victoria Tran (My-Uyen Tran)  
**Research-use notice:** This is an educational classification study, not a clinically validated diagnostic system. It should not be used for medical decision-making.
