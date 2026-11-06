---
layout: single
title: "Experimental Design and Pipeline"
permalink: /pipeline/
author_profile: true
toc: true
toc_label: "On This Page"
toc_icon: "cogs"
toc_sticky: true
---

**From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST**

*An educational empirical study comparing MLP and CNN architectures, with a focus on classification performance and trainable parameter efficiency.*

| Study attribute | Implementation |
|---|---|
| Dataset | BloodMNIST (MedMNIST v2); eight-class, 28 × 28 RGB blood-cell images |
| Models | M0: two-hidden-layer MLP; M1: CNN + BatchNorm + Flatten; M2: CNN + BatchNorm + GAP |
| Preprocessing | `ToTensor()` followed by RGB normalization using training-set statistics only |
| Training | Adam, learning rate 0.001, batch size 128, 12 epochs per model |
| Experimental runs | Three independent model runs; one predefined random seed (42) |
| Model selection | Best validation macro-F1 checkpoint for each model |
| Final evaluation | Held-out test macro-F1 (primary), test accuracy, and one M2 confusion matrix |

This page explains the study's **experimental design, data pipeline, model architectures, training protocol, evaluation, and limitations**. It complements the [executed notebook](../code/) and [technical report (PDF)](../files/bloodmnist_technical_report_uyentran.pdf).

---

## 1. Research Objective and Scope

A multilayer perceptron (MLP) classifies an image after flattening its pixels into a vector. A convolutional neural network (CNN) processes the image with convolutional filters, allowing it to learn spatial features. This project compares these approaches on BloodMNIST and examines how the final CNN classifier head affects model size and classification performance.

### Research questions

**RQ1 — MLP vs. CNN.** How do the MLP baseline and CNN models compare in classification performance under the same official data splits and shared training settings?

**RQ2 — Flatten vs. GAP.** How does replacing a Flatten classifier head with Global Average Pooling (GAP) affect trainable parameter counts and classification performance when the CNN feature-extractor design is kept the same?

The study uses **three predefined architectures**, **one random seed**, and **one fixed training protocol**. It does not include separate BatchNorm or Dropout ablations, hyperparameter searches, multiple-seed comparisons, or statistical hypothesis tests. BloodMNIST is used as an educational image-classification benchmark, not as a clinically validated diagnostic system.

---

## 2. End-to-End Experimental Workflow

```text
Research questions: RQ1 (MLP vs. CNN), RQ2 (Flatten vs. GAP)
                           |
                           v
               BloodMNIST official splits
               Train / Validation / Test
                           |
                           v
           Inspect images and class distributions
                           |
                           v
          Compute RGB mean/std from TRAIN only
                           |
                           v
         ToTensor() + shared RGB Normalize()
                           |
                           v
             Define and verify M0, M1, M2
                           |
                           v
          For each model: reset seed to 42
          Fresh model + He initialization
          Fresh Adam + shuffled train loader
                           |
                           v
                 Train for 12 epochs
          Record train loss and validation metrics
                           |
                           v
            Save best validation macro-F1
                    checkpoint/model
                           |
                           v
             Evaluate three checkpoints
             once on the held-out test set
                           |
                           v
            M2 test confusion matrix
                           |
                           v
         Answer RQ1/RQ2; discuss limitations
```

The notebook implements this workflow in **Parts 0–6**. The test set is used only after checkpoint selection; test results are not used to change the models or training settings.

---

## 3. Dataset and Preprocessing

### Official dataset partitions

BloodMNIST contains **17,092 RGB images** of size **28 × 28**, representing eight blood-cell classes. The study uses the official MedMNIST partitions without making new splits.

| Partition | Images | Role |
|---|---:|---|
| Training | 11,959 | Update model parameters; estimate RGB normalization statistics |
| Validation | 1,712 | Evaluate loss, accuracy, and macro-F1 after each epoch; select checkpoints |
| Test | 3,421 | Evaluate the three selected models once; analyze M2 errors |

### Training-derived RGB normalization

1. Load the official training images with `transforms.ToTensor()`, converting pixel values to floating-point tensors in `[0, 1]`.
2. Calculate mean and population standard deviation for each RGB channel using **training images only**.
3. Apply the same transformation, `ToTensor()` followed by `Normalize(train_mean, train_std)`, to the training, validation, and test partitions.
4. Create DataLoaders with batch size 128. Shuffle training data using a seeded generator; do not shuffle validation or test data.

The RGB statistics calculated in the notebook are:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| Red | 0.794348 | 0.215630 |
| Green | 0.659659 | 0.241603 |
| Blue | 0.696191 | 0.117889 |

For each channel, normalization uses:

```text
normalized_pixel = (pixel - training_channel_mean) / training_channel_std
```

Values after normalization are **not necessarily between 0 and 1**. Calculating the statistics from the training split alone prevents validation or test images from contributing to preprocessing estimates. No resizing or data augmentation is used.

Images enter the models with shape `(B, 3, 28, 28)`. The classification labels are converted to shape `(B,)` with integer class IDs 0–7. The notebook also checks data shapes, label ranges, and finite pixel values.

For the dataset source, class definitions, and access instructions, see the [Dataset page](../dataset/).

---

## 4. Model Architectures and Initialization

All models receive normalized BloodMNIST images and return **eight raw logits** per image, with output shape `(B, 8)`. The logits are passed directly to `nn.CrossEntropyLoss()`; there is no separate output Softmax layer.

| Model | Architecture | Trainable parameters |
|---|---|---:|
| **M0 — MLP** | Flatten (2,352) → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout → Linear(8) | **636,296** |
| **M1 — CNN + Flatten** | Two Conv–BN–ReLU–MaxPool blocks (32, 64 channels) → Flatten (3,136) → Dropout → Linear(8) | **44,680** |
| **M2 — CNN + GAP** | Same two-block CNN design → GAP → Flatten (64) → Dropout → Linear(8) | **20,104** |

### M0 — Two-hidden-layer MLP

M0 flattens each RGB image into a vector of **2,352 values**. It uses hidden layers of 256 and 128 units, with ReLU and Dropout (`p = 0.2`) after each hidden layer. M0 does **not** use Batch Normalization.

### M1 and M2 — Two-block CNNs

Both CNNs use the same *feature-extractor design*:

```text
Input: (B, 3, 28, 28)
    |
    v
Conv2d(3, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    |
    v
(B, 32, 14, 14)
    |
    v
Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    |
    v
(B, 64, 7, 7)
```

The classifier heads differ:

- **M1 — Flatten:** Converts the `(64, 7, 7)` feature maps to **3,136 features**, followed by Dropout and `Linear(3136, 8)`.
- **M2 — GAP:** Uses `AdaptiveAvgPool2d(1)` to reduce the feature maps to `(64, 1, 1)`, then flattens to **64 features**, followed by Dropout and `Linear(64, 8)`.

M2 has **24,576 fewer trainable parameters**, or **55.0% fewer**, than M1. M1 and M2 use the same backbone *architecture*, but their weights are initialized and trained independently; they do not share a trained backbone.

### Weight initialization

- **He/Kaiming Uniform:** Applied to M0's two hidden Linear weight matrices and to the CNN convolutional weights, using `fan_in` mode for ReLU.
- **PyTorch defaults:** Retained for output Linear layers, convolutional biases, and BatchNorm parameters.
- **Dropout:** Uses `p = 0.2` where specified and does not add trainable parameters.

The notebook verifies each model's output shape and trainable parameter count before training.

---

## 5. Fixed Training and Validation Protocol

| Setting | Value |
|---|---|
| Random seed | 42, reset before creating each model |
| Optimizer | `torch.optim.Adam` (fresh optimizer per model) |
| Learning rate | 0.001 for all models |
| Loss | Unweighted `nn.CrossEntropyLoss()` |
| Batch size | 128 for train, validation, and test |
| Training duration | 12 complete epochs per model; no early stopping |
| Training DataLoader | Fresh seed-controlled shuffled loader for each run |
| Validation | Loss, accuracy, and macro-F1 after every epoch |
| Best checkpoint | Highest validation macro-F1; earlier epoch retained on an exact tie |
| Final test | Macro-F1 (primary) and accuracy (secondary) |

### Training and validation loop

For each training mini-batch:

```text
Load images and labels -> optimizer.zero_grad()
                       -> forward pass (logits)
                       -> cross-entropy loss
                       -> loss.backward()
                       -> optimizer.step()
                       -> accumulate sample-weighted training loss
```

After every epoch, the model switches to evaluation mode with `model.eval()` and uses `torch.no_grad()` to calculate validation metrics. **Validation macro-F1 is computed across the complete validation set**, not by averaging mini-batch F1 scores.

A model's `state_dict()` is saved whenever validation macro-F1 improves strictly. Training continues for all 12 epochs even if the best checkpoint occurred earlier. The recorded training loss is the sample-weighted average of mini-batch losses observed during training.

### Official model runs

Exactly **three independent runs** are performed: M0, M1, and M2. Before each run, the code resets the seed and creates a fresh model, initialization, Adam optimizer, and training DataLoader. The same dataset splits, normalization statistics, and training settings are used for all models.

This is a shared experimental protocol, **not a claim that the models have identical capacity or that the learning rate is optimal for each architecture**.

---

## 6. Checkpoints, Learning Curves, and Final Test Evaluation

### Validation-selected checkpoints

The best epochs obtained in the notebook are **12 for M0**, **7 for M1**, and **11 for M2**. The notebook saves each best checkpoint, along with a CSV training history and learning-curve figure. Learning curves show training/validation cross-entropy and validation macro-F1 across all 12 epochs.

### Held-out test evaluation

After all three checkpoints are selected, the notebook reconstructs the model architectures, loads each saved checkpoint, and evaluates it **once** on the official test set.

| Model | Parameters | Best epoch | Best val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| **M0** | 636,296 | 12 | 0.8366 | 0.8381 | 0.8150 |
| **M1** | 44,680 | 7 | **0.9156** | **0.9152** | **0.9051** |
| **M2** | 20,104 | 11 | 0.8596 | 0.8702 | 0.8452 |

The ordering is **M1 > M2 > M0** by both validation and test macro-F1.

### Answers to the research questions

**RQ1 — MLP vs. CNN:** Both CNNs achieved higher test macro-F1 than the MLP baseline. Compared with M0, test macro-F1 was **9.01 percentage points higher for M1** and **3.02 points higher for M2**. However, M0 and the CNNs also differ in Batch Normalization, parameter count, and overall design. The experiment does not isolate the effect of convolution alone.

**RQ2 — Flatten vs. GAP:** M2 used **55.0% fewer trainable parameters** than M1, but its test macro-F1 was **5.99 percentage points lower**. This is an observed performance–parameter trade-off in the tested models, not evidence that Flatten is always better than GAP. Parameter count alone does not measure training speed or computational cost.

---

## 7. M2 Confusion Matrix and Error Analysis

M2 was selected for the confusion-matrix analysis **before inspecting test performance**. Its confusion matrix uses the predictions already collected during final test evaluation; the model is not evaluated a second time.

- M2 correctly classified **2,977 of 3,421** test images and misclassified **444**.
- **Monocyte** had the lowest class recall: **163/284 (57.39%)**.
- The largest directed error was **monocyte → immature granulocytes**, with **114** test images.
- **Platelet** and **eosinophil** had high recall: **469/470** and **610/624**, respectively.

These results show that classification performance varies across classes, even when overall accuracy is relatively high. The confusion matrix describes **which errors occurred**, not the underlying reasons for them.

The figure is available in the repository at [`images/m2-test-confusion-matrix.png`](../images/m2-test-confusion-matrix.png).

---

## 8. Limitations and Further Investigation

The findings are limited by the following design choices:

- **Single seed:** Each model was trained once. The study does not quantify variability across random seeds or establish statistical significance.
- **Architectural differences:** M0 differs from M1/M2 in more than convolution, including BatchNorm and parameter count.
- **Low resolution:** Images are only 28 × 28 pixels; results may not extend to higher-resolution biomedical images.
- **Fixed settings:** One learning rate and a 12-epoch budget were used without augmentation or hyperparameter optimization.
- **Clinical scope:** Benchmark classification results do not establish clinical reliability or diagnostic suitability.

Future work could examine repeated random seeds, additional architectures, or training-setting sensitivity. **These are possible extensions, not experiments conducted in this project.**

---

## 9. Reproducibility and Notebook-to-Pipeline Map

### Running the source notebook

The public repository stores the source notebook at [`files/bloodmnist-mlp-cnn-uyentran.ipynb`](../files/bloodmnist-mlp-cnn-uyentran.ipynb) and its browser-readable export on the [Code page](../code/).

**Important local-directory requirement:** The current notebook's `find_project_root()` function expects a project directory containing both `notebooks/` and `data/`. To run the downloaded notebook locally, place it under `notebooks/` within a working directory that also contains `data/`. The code creates `results/` for experiment outputs.

```text
WorkingProjects/
├── notebooks/
│   └── bloodmnist-mlp-cnn-uyentran.ipynb
├── data/
└── results/             # Created by the notebook if needed
```

Install the Python dependencies used in the notebook, restart the kernel, and execute **Parts 0–6 in order**. The `medmnist` package downloads BloodMNIST when needed. A complete reproduction retrains three models, regenerates their checkpoints and training histories, and reevaluates the selected checkpoints on the test split. Numerical results can vary across software versions and hardware backends.

### Notebook-to-pipeline map

| Notebook part | Role |
|---|---|
| **Part 0 — Setup & Reproducibility** | Import libraries, select device, define fixed configuration and random seed |
| **Part 1 — Load, Inspect & Normalize BloodMNIST** | Load official splits, inspect images/class counts, compute training RGB statistics, create DataLoaders |
| **Part 2 — Define M0, M1, M2 & Initialize Weights** | Build the MLP/CNN architectures, apply He initialization, verify shapes and parameters |
| **Part 3 — Basic Training & Validation** | Implement `train_one_epoch()`, `evaluate()`, and `train_model()` |
| **Part 4 — Official Training: Only Three Runs** | Train three models, save best checkpoints, training histories, and learning curves |
| **Part 5 — Final Test Evaluation & Simple Error Analysis** | Evaluate three selected checkpoints and generate the M2 confusion matrix |
| **Part 6 — Results, Discussion & Conclusion** | Answer RQ1/RQ2, explain limitations, and summarize findings |

---

## Research Materials

- [Dataset and access instructions](../dataset/)
- [Jupyter Notebook and HTML preview](../code/)
- [Executable notebook (.ipynb)](../files/bloodmnist-mlp-cnn-uyentran.ipynb)
- [Technical report (PDF)](../files/bloodmnist_technical_report_uyentran.pdf)
- [GitHub repository](https://github.com/uyentran9/bloodmnist-mlp-cnn)



<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
