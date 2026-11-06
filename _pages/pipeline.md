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

This page describes the experimental design used to compare **ImageNet-pretrained ResNet-18 (CNN)** and **DeiT-Tiny (Vision Transformer)** for pediatric pneumonia classification on **PneumoniaMNIST+ 224**. It focuses on preprocessing, model setup, full fine-tuning, checkpoint selection, and final evaluation.

![PneumoniaMNIST+ 224 model architectures and experimental pipeline]({{ '/images/vision-transformer-pneumonia4.png' | relative_url }})

*Figure 1. Shared data pipeline, model architectures, training settings, and evaluation workflow.*

## 1. Research Question and Study Design

**Research question:** How do ImageNet-pretrained ResNet-18 and DeiT-Tiny compare in classification performance and training time when fully fine-tuned under the same experimental protocol?

| Study component | Design |
|---|---|
| Dataset | PneumoniaMNIST+ 224; binary pediatric chest X-ray classification |
| Models | E1: ResNet-18 (CNN); E2: DeiT-Tiny (Vision Transformer) |
| Initialization | ImageNet-1K pretrained weights; two-class classification heads |
| Training | Full fine-tuning with shared hyperparameters |
| Random seeds | 42, 43, 44 for each model (six runs total) |
| Checkpoint selection | Highest validation macro-F1 within each run |
| Final evaluation | Official held-out test set; mean ± sample SD across seeds |

The study compares two practical pretrained modeling pipelines. It does **not** isolate the effect of convolution versus self-attention, because the models also differ in parameter count and pretraining details.

## 2. Dataset and Preprocessing

The study uses the official MedMNIST splits without creating new partitions.

| Split | Images | Purpose |
|---|---:|---|
| Training | 4,708 | Update model parameters |
| Validation | 524 | Select the best checkpoint for each model and seed |
| Test | 624 | Final evaluation after checkpoint selection |

The labels are **Normal (0)** and **Pneumonia (1)**. Pneumonia represents approximately 74.2% of the training and validation images, compared with 62.5% of the test images.

Both models use the same preprocessing:

1. Load the official **224 × 224 grayscale** images.
2. Repeat the grayscale channel to create a **three-channel image**.
3. Convert the image to a `float32` tensor using `ToTensor()`.
4. Normalize each channel using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

The resulting input shape is **`(B, 3, 224, 224)`**, where `B` is the batch size. No random data augmentation is used. Training batches are shuffled with a seed-controlled generator; validation and test batches are not shuffled.

For sample images, class counts, and official data access, see the [Dataset page]({{ '/dataset/' | relative_url }}).

## 3. Model Architectures

Both models are loaded with `timm` and configured to output **two raw logits** per image. All parameters are trainable during full fine-tuning; no pretrained layers are frozen.

| Model | Main architecture | Parameters (approx.) |
|---|---|---:|
| **E1 — ResNet-18** | Convolutional stem, four residual stages, global average pooling, linear classifier | 11.18 million |
| **E2 — DeiT-Tiny** | Patch embeddings, 12 Transformer encoder blocks, CLS token, linear classifier | 5.52 million |

### E1 — ResNet-18

ResNet-18 learns image representations through convolutional layers and residual connections. Its final feature vector has **512 dimensions**, followed by a linear classifier that produces two logits.

### E2 — DeiT-Tiny

DeiT-Tiny divides a 224 × 224 image into **196 non-overlapping 16 × 16 patches**. Each patch is represented by a **192-dimensional embedding**. After adding a CLS token and positional embeddings, the model processes **197 tokens** through 12 Transformer encoder blocks, each with three attention heads. The classification head maps the 192-dimensional representation to two logits.

Both architectures start from pretrained ImageNet-1K weights, but their original classification heads are replaced for the two-class task.

## 4. Full Fine-Tuning and Checkpoint Selection

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Loss | Unweighted `CrossEntropyLoss` |
| Batch size | 16 |
| Training duration | 5 epochs per run; no early stopping |
| Random seeds | 42, 43, 44 |
| Training DataLoader | Fresh seed-controlled shuffled loader for each run |
| Validation | Cross-entropy loss and macro-F1 after every epoch |
| Checkpoint rule | Save weights when validation macro-F1 strictly improves |

For each model and seed, the notebook resets the random seed, creates a fresh model from pretrained weights, initializes a new optimizer, and creates a new training DataLoader. It then:

1. Trains the model for five epochs, updating **all model parameters**.
2. Records training loss, validation loss, and validation macro-F1 after each epoch.
3. Saves the checkpoint with the **highest validation macro-F1**; an exact tie retains the earlier checkpoint.
4. Saves the training history and a validation summary for the run.

This produces **six validation-selected checkpoints**. The same training settings are applied to both models for consistency; they are not claimed to be optimal for either architecture.

## 5. Final Test Evaluation

The notebook reconstructs each model, loads its selected checkpoint, and evaluates it on the **624-image official test set**. Test results are not used for checkpoint selection or hyperparameter tuning.

The evaluation reports accuracy, **macro-F1 (primary metric)**, sensitivity (recall for Pneumonia), specificity (recall for Normal), and ROC-AUC. Results are summarized as **mean ± sample standard deviation** over three seeds for each model.

| Test metric | ResNet-18 | DeiT-Tiny |
|---|---:|---:|
| Accuracy | 0.9209 ± 0.0148 | 0.9215 ± 0.0100 |
| **Macro-F1** | **0.9114 ± 0.0176** | **0.9121 ± 0.0119** |
| Sensitivity | 0.9974 ± 0.0000 | 0.9974 ± 0.0026 |
| Specificity | 0.7934 ± 0.0395 | 0.7949 ± 0.0299 |
| ROC-AUC | 0.9907 ± 0.0027 | 0.9900 ± 0.0005 |
| Training time per run | ~1.37 min | ~1.36 min |

The mean test macro-F1 scores and training times are similar under this protocol. Both models show high sensitivity but lower specificity. The three-seed comparison is too limited to support a broad claim that one architecture is superior.

The [test results summary]({{ '/files/test_results_summary.csv' | relative_url }}) and [per-seed results]({{ '/files/test_results_per_seed.csv' | relative_url }}) provide the exact reported values.

### Training Behavior and Classification Errors

![Training and validation loss across six runs]({{ '/images/training_curves.png' | relative_url }})

*Figure 2. Training and validation cross-entropy loss for both models and all three seeds.*

ResNet-18's validation loss generally decreased or stabilized. DeiT-Tiny's validation loss rose in later epochs for seeds 42 and 44 while its training loss continued to decrease, suggesting possible overfitting in those runs.

![Test confusion matrices for ResNet-18 and DeiT-Tiny]({{ '/images/confusion_matrices.png' | relative_url }})

*Figure 3. Test confusion matrices for the six validation-selected checkpoints.*

Across the runs, false positives (Normal predicted as Pneumonia) were more frequent than false negatives (Pneumonia predicted as Normal). These figures describe observed errors; they do not establish their underlying causes.

## 6. Limitations

- **Small evaluation set:** The official test split contains 624 images; no confidence intervals or significance tests were performed.
- **Three seeds:** Mean ± SD summarizes run-to-run variation but does not measure uncertainty across new test datasets.
- **Fixed training settings:** Both architectures use the same five-epoch budget without model-specific hyperparameter tuning or data augmentation.
- **Model differences:** Architecture, parameter count, and pretrained weights differ, so performance differences cannot be attributed solely to convolution or self-attention.
- **Limited external validity:** No external-hospital evaluation was performed. The results are for research and education, not clinical diagnosis.

## 7. Reproducibility and Research Materials

The executed notebook documents the implementation in **Parts 0–6**:

| Notebook part | Content |
|---|---|
| **Part 0** | Setup, configuration, device, seeds, output directories |
| **Part 1** | Official dataset splits, preprocessing, inspection, DataLoaders |
| **Part 2** | Pretrained model loading and architecture verification |
| **Part 3** | Training, validation, and checkpoint-selection functions |
| **Part 4** | Six full fine-tuning runs and validation summaries |
| **Part 5** | Final test metrics, training curves, confusion matrices |
| **Part 6** | Results, interpretation, limitations, and conclusion |

The notebook writes training histories to `results/`, checkpoints to `checkpoints/`, and figures to `images/`. Its working directory determines the project root when the expected `notebooks/` and `data/` directories are not found. Check `cfg.project_root` before running it locally. Reproducing the full experiment retrains all six runs; published checkpoint files are not required.

- [Jupyter notebook and executed HTML preview]({{ '/code/' | relative_url }})
- [Validation summary (CSV)]({{ '/files/validation_summary.csv' | relative_url }})
- [Technical report]({{ '/report/' | relative_url }})
- [GitHub repository](https://github.com/uyentran9/pneumoniamnist-cnn-transformer)

---  

<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
