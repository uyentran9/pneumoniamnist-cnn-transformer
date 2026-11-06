---
title: "From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency aon BloodMNIST"
layout: single
author_profile: true
---

<p align="center">
  <img src="{{ site.baseurl }}/images/bloodmnist-mlp-cnn.png"
       width="960"
       alt="From MLP to CNN: BloodMNIST model comparison"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>

---

# From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST

An educational empirical study comparing **multilayer perceptron (MLP) and convolutional neural network (CNN) architectures**, with a focus on spatial feature learning, classifier-head design, and trainable parameter efficiency in BloodMNIST image classification.

> **Project:** From MLP to CNN, individual empirical study  
> **Research area:** Deep learning and biomedical image classification  
> **Dataset:** BloodMNIST (MedMNIST v2)  
> **Framework:** PyTorch  
> **Experimental scope:** Three architectures (M0, M1, M2); one random seed; three independent training runs

**Project materials:** [Jupyter notebook]({{ site.baseurl }}/files/bloodmnist-mlp-cnn-uyentran.ipynb) · [Technical report (PDF)]({{ site.baseurl }}/files/bloodmnist_technical_report_uyentran.pdf) · [GitHub repository](https://github.com/uyentran9/bloodmnist-mlp-cnn)

## Research overview

A multilayer perceptron (MLP) classifies images after flattening their pixel values into a one-dimensional vector. A convolutional neural network (CNN) instead uses convolutional filters to learn features from local spatial patterns. These approaches differ in how they process image information and in the number of parameters required for classification.

This project compares an **MLP baseline (M0)** with two small CNNs: **M1**, which uses a Flatten classifier head, and **M2**, which uses Global Average Pooling (GAP). The goal is to examine classification performance and parameter efficiency under shared data splits and fixed training settings, not to develop a clinical diagnostic system.

### Research questions

1. **RQ1 — MLP vs. CNN:** How do MLP and CNN architectures compare in BloodMNIST classification performance under the same official data splits and training settings?
2. **RQ2 — Flatten vs. GAP:** How does replacing a Flatten classifier head with GAP affect trainable parameter counts and classification performance when the CNN feature-extractor design is kept the same?

The study reports observed results for these two questions. It does not attempt to establish a universally superior architecture or perform statistical hypothesis testing.

## Dataset and preprocessing

[BloodMNIST](https://medmnist.com/) contains **17,092 RGB blood-cell images** of size **28 × 28 pixels**, representing eight classes. The official dataset splits were used without additional splitting or data augmentation.

| Split | Images | Purpose |
|---|---:|---|
| Training | 11,959 | Learn model parameters |
| Validation | 1,712 | Monitor performance and select the best checkpoint |
| Test | 3,421 | Evaluate selected checkpoints once after model selection |

**Preprocessing:** Images were converted to floating-point tensors using `transforms.ToTensor()`. Per-channel RGB means and standard deviations were then computed **from the training split only**:

- **RGB mean:** `(0.794348, 0.659659, 0.696191)`
- **RGB standard deviation:** `(0.215630, 0.241603, 0.117889)`

The same training-derived statistics were used with `transforms.Normalize()` for all three splits. Validation and test images were not used to estimate normalization statistics. No resizing or data augmentation was applied.

## Experimental design

Three architectures were implemented in PyTorch. Every model receives an RGB image tensor of shape `(B, 3, 28, 28)` and produces eight raw classification logits.

| Model | Architecture | Trainable parameters |
|---|---|---:|
| **M0 — MLP** | Flatten → Linear(2352, 256) → Linear(256, 128) → Linear(128, 8); ReLU and Dropout after hidden layers | 636,296 |
| **M1 — CNN + BN + Flatten** | Two Conv–BatchNorm–ReLU–MaxPool blocks (32 and 64 channels) → Flatten → Dropout → Linear(3136, 8) | 44,680 |
| **M2 — CNN + BN + GAP** | Same two-block CNN design → GAP → Flatten → Dropout → Linear(64, 8) | 20,104 |


<h3>Model architecture overview</h3>

<p>The diagram below illustrates the three model architectures, their intermediate tensor shapes, weight initialization, classifier heads, and trainable parameter counts.</p>

<p align="center">
  <img src="{{ site.baseurl }}/images/BloodMNIST%20Model%20Architectures%20Infographic.png"
       width="960"
       alt="BloodMNIST model architecture diagram comparing M0 MLP, M1 CNN with Flatten, and M2 CNN with Global Average Pooling"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>


M0 has two fully connected hidden layers and does not use Batch Normalization. M1 and M2 use the same **CNN backbone design** but have different classifier heads; they were initialized and trained independently rather than sharing trained weights.

### Shared training protocol

| Setting | Configuration |
|---|---|
| Random seed | 42; reset before each model run |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Unweighted cross-entropy |
| Batch size | 128 |
| Training duration | 12 epochs per model; no early stopping |
| Activation / Dropout | ReLU / `p = 0.2` |
| Weight initialization | He/Kaiming Uniform for M0 hidden Linear weights and CNN convolutional weights |
| Checkpoint selection | Highest validation macro-F1; earlier epoch retained on a tie |
| Final evaluation | Test macro-F1 (primary), test accuracy (secondary) |
| Error analysis | One predefined confusion matrix for M2 |

All three models used the same dataset preprocessing and fixed training settings. Each run used a fresh model, a fresh Adam optimizer, and a seed-controlled shuffled training DataLoader. No learning-rate pilot or hyperparameter search was performed.

## What I investigated

- **Architecture comparison:** Compared an MLP that flattens images before classification with CNNs that learn spatial features using convolutional layers.
- **Classifier-head design:** Compared Flatten and GAP after the same two-block CNN feature-extractor design.
- **Parameter efficiency:** Counted trainable parameters and measured the reduction associated with the GAP head.
- **Training and validation behavior:** Recorded cross-entropy loss, validation accuracy, and validation macro-F1 across 12 epochs.
- **Held-out test performance:** Evaluated each model once using its best validation-selected checkpoint.
- **Classification errors:** Examined M2's test confusion matrix to identify which blood-cell classes were most frequently confused.

## Models and methods

`PyTorch` · `MLP` · `CNN` · `Batch Normalization` · `ReLU` · `Dropout` · `He/Kaiming Uniform` · `Adam` · `Flatten` · `Global Average Pooling` · `Macro-F1` · `Confusion Matrix`

## Experimental results

The following results use each model's **best validation macro-F1 checkpoint**. Macro-F1 is the primary evaluation metric because it gives equal importance to the eight classes when calculating the average class F1 score.

| Model | Parameters | Best epoch | Best validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| **M0 — MLP** | 636,296 | 12 | 0.8366 | 0.8381 | 0.8150 |
| **M1 — CNN + Flatten** | 44,680 | 7 | **0.9156** | **0.9152** | **0.9051** |
| **M2 — CNN + GAP** | 20,104 | 11 | 0.8596 | 0.8702 | 0.8452 |

The ranking by both validation and test macro-F1 was **M1 > M2 > M0**. The selected best epochs differed across models, so final evaluation used the saved checkpoints rather than necessarily the weights from epoch 12.

### RQ1 — MLP vs. CNN

Both CNN models performed better than the MLP baseline on the held-out test set:

- **M1 vs. M0:** Test macro-F1 increased from 0.8150 to 0.9051, a difference of **0.0901 (9.01 percentage points)**.
- **M2 vs. M0:** Test macro-F1 increased from 0.8150 to 0.8452, a difference of **0.0302 (3.02 percentage points)**.

Both CNNs also had substantially fewer trainable parameters than M0. These observations favor the tested CNN designs for this task, but they do not isolate the effect of convolution because the models also differ in Batch Normalization and parameter counts.

### RQ2 — Flatten vs. GAP

M1's Flatten head passes **3,136 features** to its final Linear layer, whereas M2's GAP head reduces the classifier input to **64 features**.

- **Parameter reduction:** M2 uses **24,576 fewer trainable parameters** than M1, a reduction of **55.0%**.
- **Test performance difference:** M2's test macro-F1 (0.8452) is **0.0599 (5.99 percentage points)** lower than M1's (0.9051).

Under this experimental protocol, GAP provided a smaller model at the cost of lower classification performance. Fewer parameters do not, by themselves, establish faster training or inference.

### Training and validation curves

The three figures below show training/validation loss and validation macro-F1 across the 12 training epochs. Dashed markers identify the selected best epochs.

**Figure 1 — M0 (MLP):** Training/validation loss and validation macro-F1; best validation checkpoint at epoch 12.

<p align="center">
  <img src="{{ site.baseurl }}/images/m0-TrainValLoss-ValMacroF1.png"
       width="960"
       alt="M0 MLP training and validation loss and validation macro-F1 across 12 epochs"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>

**Figure 2 — M1 (CNN + Flatten):** Training/validation loss and validation macro-F1; best validation checkpoint at epoch 7.

<p align="center">
  <img src="{{ site.baseurl }}/images/m1-TrainValLoss-ValMacroF1.png"
       width="960"
       alt="M1 CNN with Flatten training and validation loss and validation macro-F1 across 12 epochs"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>

**Figure 3 — M2 (CNN + GAP):** Training/validation loss and validation macro-F1; best validation checkpoint at epoch 11.

<p align="center">
  <img src="{{ site.baseurl }}/images/m2-TrainValLoss-ValMacroF1.png"
       width="960"
       alt="M2 CNN with Global Average Pooling training and validation loss and validation macro-F1 across 12 epochs"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>

Training loss decreased in all three runs. Validation performance differed across architectures and did not always improve with additional epochs, supporting the use of validation-selected checkpoints.

### M2 confusion matrix and error analysis

The M2 confusion matrix uses predictions from the selected checkpoint on the **3,421-image official test split**. Rows represent true classes and columns represent predicted classes.

<p align="center">
  <img src="{{ site.baseurl }}/images/m2-test-confusion-matrix.png"
       width="900"
       alt="M2 BloodMNIST test confusion matrix showing eight blood-cell classes"
       style="max-width: 100%; height: auto; border-radius: 14px;"/>
</p>

**Figure 4 — M2 test confusion matrix.** Diagonal cells are correct predictions; off-diagonal cells are misclassifications.

M2 correctly classified **2,977 of 3,421 test images** (87.02% accuracy). Monocytes had the lowest class recall (**163/284 = 57.39%**), with **114 monocyte images** misclassified as immature granulocytes. Platelets (**469/470**) and eosinophils (**610/624**) had higher recall. These results identify where prediction errors occurred but do not establish their causes.

## What the project demonstrates

This study shows how MLP and CNN architecture choices relate to classification performance and model size in a small biomedical image-classification task. Under the tested settings, **M1 achieved the highest test macro-F1 (0.9051)**, while **M2 used the fewest trainable parameters (20,104)**.

The comparison illustrates a practical trade-off: replacing Flatten with GAP substantially reduced parameters, but the smaller CNN achieved lower classification performance in this single-seed experiment. These findings describe the observed results, not a universal ranking of architectures or classifier heads.

## Experimental workflow and reproducibility

1. **Prepare the data:** Load official BloodMNIST splits; inspect image shapes and class counts; compute training-only RGB mean/std; apply shared normalization.
2. **Build the models:** Implement M0 and the two-head CNN design; initialize specified weights; verify output shapes and parameter counts.
3. **Train and validate:** Train each model for 12 epochs using Adam; record training loss and validation metrics; save the best validation macro-F1 checkpoint.
4. **Summarize results:** Export training histories and create learning curves for all three models.
5. **Evaluate on test data:** Restore each selected checkpoint and calculate test accuracy and macro-F1 once.
6. **Analyze errors:** Produce the predefined M2 confusion matrix and discuss the two research questions and experimental limitations.

**Reproduction notes:** Download the [Jupyter notebook]({{ site.baseurl }}/files/bloodmnist-mlp-cnn-uyentran.ipynb) and run its cells from Part 0 through Part 6 in order. The notebook installs/imports the required Python libraries and downloads BloodMNIST through `medmnist` when needed. Its `find_project_root()` helper expects a local project directory containing `notebooks/` and `data/`; place the downloaded notebook in `notebooks/` and create `data/` before running it. Output artifacts are written to `results/` under the detected project root.

The original experiment was executed in a Python/PyTorch environment with an Apple Silicon MPS backend. Numerical results may differ across hardware backends, package versions, and execution environments.

**Project materials:** [Executable notebook]({{ site.baseurl }}/files/bloodmnist-mlp-cnn-uyentran.ipynb) · [Technical report (PDF)]({{ site.baseurl }}/files/bloodmnist_technical_report_uyentran.pdf) · [Source repository](https://github.com/uyentran9/bloodmnist-mlp-cnn)

## Limitations and future work

- **Single seed:** Each model was trained once; seed-to-seed variability and statistical significance were not measured.
- **Architectural differences:** M0 differs from the CNNs in convolution, Batch Normalization, and parameter counts. Their results do not isolate the causal effect of convolution.
- **Image resolution:** BloodMNIST uses 28 × 28 images; findings may not generalize to higher-resolution biomedical images.
- **Fixed training settings:** All models used one learning rate and a 12-epoch training budget, without hyperparameter tuning or data augmentation. These settings may not be optimal for every model.
- **Clinical scope:** Test results on this benchmark do not establish performance on real-world clinical data or suitability for medical diagnosis.

Future work could evaluate additional random seeds, training settings, and biomedical image datasets. These extensions were **not** performed in the current study.

## Data and intended use

BloodMNIST is part of the [MedMNIST benchmark](https://medmnist.com/). This educational study uses the dataset for neural-network architecture comparison. It is **not a clinically validated diagnostic system** and is not intended for medical decision-making.

## Author

**Victoria Tran (My-Uyen Tran)**  
Individual project

Thank you for visiting — and for being part of my journey 🌱
