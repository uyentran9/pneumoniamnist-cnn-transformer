---
title: "Comparative Evaluation of ImageNet-Pretrained ResNet-18 and DeiT-Tiny for Pediatric Pneumonia Classification"
layout: single
author_profile: true
---

<p align="center">
  <img src="{{ site.baseurl }}/images/vision-transformer-pneumonia5.png"
       width="960"
       alt="ResNet-18 versus DeiT-Tiny for pediatric pneumonia classification on PneumoniaMNIST+ 224"
       style="max-width: 100%; height: auto; border-radius: 14px;" />
</p>


An independent educational study comparing **a convolutional neural network (ResNet-18)** and **a Vision Transformer (DeiT-Tiny)** for pediatric chest X-ray classification using **PneumoniaMNIST+ 224**.

> **Research area:** Deep learning · Biomedical computer vision · Transfer learning  
> **Dataset:** PneumoniaMNIST+ 224 (Normal vs. Pneumonia)  
> **Framework:** PyTorch · `timm` · MedMNIST  
> **Experimental scope:** Two ImageNet-pretrained models · Three random seeds · Six full fine-tuning runs

**Project materials:** [Jupyter notebook]({{ site.baseurl }}/code/) · [Notebook (HTML)]({{ site.baseurl }}/code/) · [Technical report (PDF)]({{ site.baseurl }}/report/) · [GitHub repository](https://github.com/uyentran9/pneumoniamnist-cnn-transformer)

## Research overview

Convolutional neural networks learn image features using local convolutional filters. Vision Transformers divide images into patches and model relationships among patch representations using self-attention. Both approaches can be adapted to biomedical images through transfer learning.

This project compares ImageNet-pretrained ResNet-18 and DeiT-Tiny using the **same dataset splits, preprocessing, and fixed fine-tuning settings**. Each model is trained separately with seeds **42, 43, and 44**. The best checkpoint from each run is selected by validation macro-F1 and evaluated on the held-out test set.

**Research question:** How do ImageNet-pretrained ResNet-18 and DeiT-Tiny compare in pediatric pneumonia classification performance and training time under a shared full fine-tuning protocol?

The goal is to study these two pretrained model pipelines, not to establish a universally better architecture or build a clinical diagnostic system.

## Dataset and preprocessing

[PneumoniaMNIST+ 224](https://medmnist.com/) provides **5,856 grayscale chest X-ray images** at **224 × 224** resolution, labeled **Normal (0)** or **Pneumonia (1)**. The official splits were used without additional partitioning.

| Split | Normal | Pneumonia | Total | Purpose |
|---|---:|---:|---:|---|
| Training | 1,214 | 3,494 | 4,708 | Update model weights |
| Validation | 135 | 389 | 524 | Select the best checkpoint |
| Test | 234 | 390 | 624 | Final evaluation only |

**Preprocessing:** Each grayscale image was converted into three identical channels, transformed to a tensor, and normalized using ImageNet channel means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Both models used this same deterministic preprocessing. No random data augmentation or class reweighting was applied.

<p align="center">
  <img src="{{ site.baseurl }}/images/class_distribution.png"
       width="860"
       alt="Percentage of Normal and Pneumonia images in the official training, validation, and test splits"
       style="max-width: 100%; height: auto;" />
</p>

*Figure 1. Class distribution in the official dataset splits.* Pneumonia represents about **74.2%** of training and validation images but **62.5%** of test images. See also [sample training X-rays]({{ site.baseurl }}/images/sample_xrays.png).

## Models and experimental design

Both models were loaded with ImageNet-1K pretrained weights through `timm`, configured for two output classes, and **fully fine-tuned**. The model input is `(B, 3, 224, 224)` and the output is `(B, 2)` logits.

| Model | Main architecture | Trainable parameters |
|---|---|---:|
| **E1 — ResNet-18 (CNN)** | Residual convolutional blocks and global average pooling | 11,177,538 |
| **E2 — DeiT-Tiny (Vision Transformer)** | 16 × 16 image patches, 12 Transformer encoder blocks, 3 attention heads per block | 5,524,802 |

DeiT-Tiny uses 196 image patches plus a CLS token, with embedding dimension 192. This study uses the non-distilled DeiT-Tiny model; no knowledge-distillation training was performed.

<p align="center">
  <img src="{{ site.baseurl }}/images/vision-transformer-pneumonia4.png"
       width="1100"
       alt="PneumoniaMNIST+ preprocessing, ResNet-18 and DeiT-Tiny architectures, and the shared training and evaluation pipeline"
       style="max-width: 100%; height: auto; border-radius: 10px;" />
</p>

*Figure 2. Model architectures and experimental pipeline.* Both models use the same input preparation, training settings, checkpoint-selection rule, and test protocol.

### Shared training protocol

| Setting | Configuration |
|---|---|
| Random seeds | 42, 43, 44 for each model |
| Optimizer | AdamW |
| Learning rate / weight decay | `2e-5` / `0.01` |
| Loss | Unweighted cross-entropy |
| Batch size | 16 |
| Training | 5 epochs per run; 6 runs in total |
| Checkpoint selection | Highest validation macro-F1; earlier epoch kept on a tie |
| Test metrics | Macro-F1 (primary), accuracy, sensitivity, specificity, ROC-AUC |
| Reporting | Mean ± sample standard deviation across three seeds |

Every run starts with its own pretrained model initialization, optimizer, and seed-controlled training DataLoader. No model-specific hyperparameter search or early stopping was used. **The test set was not used for model selection or training decisions.**

## Experimental results

### Held-out test performance

Each validation-selected checkpoint was evaluated on the **same official test set of 624 images**. Values below are **mean ± sample standard deviation across three training seeds**.

| Metric | ResNet-18 | DeiT-Tiny |
|---|---:|---:|
| **Macro-F1 (primary)** | **0.9114 ± 0.0176** | **0.9121 ± 0.0119** |
| Accuracy | 0.9209 ± 0.0148 | 0.9215 ± 0.0100 |
| Sensitivity (Pneumonia recall) | 0.9974 ± 0.0000 | 0.9974 ± 0.0026 |
| Specificity (Normal recall) | 0.7934 ± 0.0395 | 0.7949 ± 0.0299 |
| ROC-AUC | 0.9907 ± 0.0027 | 0.9900 ± 0.0005 |

The mean test macro-F1 difference was only **0.0007** (0.07 percentage points). DeiT-Tiny scored higher for seed 42; ResNet-18 scored higher for seeds 43 and 44. **Neither model was consistently better across seeds.**

ResNet-18 achieved higher mean best-validation macro-F1 (**0.9884** vs. **0.9809**), but not a clear test advantage. Recorded training times were also close: approximately **1.37 min/run** for ResNet-18 and **1.36 min/run** for DeiT-Tiny on Apple Silicon MPS. These are five-epoch training runtimes, not inference-speed measurements.

### Training and validation behavior

<p align="center">
  <img src="{{ site.baseurl }}/images/training_curves.png"
       width="1000"
       alt="Training and validation cross-entropy loss over five epochs for ResNet-18 and DeiT-Tiny across seeds 42, 43, and 44"
       style="max-width: 100%; height: auto;" />
</p>

*Figure 3. Training and validation loss for all six runs.* ResNet-18 validation loss decreased or stabilized across the three seeds. For DeiT-Tiny, validation loss rose in later epochs for seeds 42 and 44 while training loss continued to fall, suggesting possible overfitting in those runs. Checkpoints were selected using **validation macro-F1**, not minimum validation loss.

### Confusion matrices and classification errors

<p align="center">
  <img src="{{ site.baseurl }}/images/confusion_matrices.png"
       width="1000"
       alt="Six held-out test confusion matrices comparing ResNet-18 and DeiT-Tiny for seeds 42, 43, and 44"
       style="max-width: 100%; height: auto;" />
</p>

*Figure 4. Test confusion matrices for the six validation-selected checkpoints.* Each matrix includes 234 Normal and 390 Pneumonia images.

There were **0–2 false negatives** (Pneumonia predicted as Normal) and **40–59 false positives** (Normal predicted as Pneumonia) per run. Both models therefore showed high Pneumonia sensitivity but lower Normal specificity at the evaluated decision rule.

**Detailed results:** [Per-seed test metrics]({{ site.baseurl }}/files/test_results_per_seed.csv) · [Test summary]({{ site.baseurl }}/files/test_results_summary.csv) · [Validation and runtime summary]({{ site.baseurl }}/files/validation_summary.csv)

## Interpretation and limitations

Under this fixed protocol, the two **pretrained pipelines** achieved similar held-out test macro-F1 and recorded training times. The comparison does not isolate the causal effect of CNN versus Transformer architecture: the models also differ in parameter count and pretrained checkpoints.

Important limitations include:

- **Small, fixed test set:** All seeds use the same 624 images. The reported seed standard deviations are not confidence intervals for new patients or hospitals.
- **Differences between splits:** Pneumonia prevalence and dataset provenance differ across training/validation and test data. Their contribution to the validation–test gap is not established.
- **Fixed training budget:** Five epochs and shared hyperparameters may not be optimal for either architecture.
- **Limited generalization evidence:** Three training seeds, no statistical equivalence test, and no independent external-hospital evaluation.

The results describe an educational benchmark experiment, **not clinical diagnostic performance**.

## Reproducibility and research materials

The experiment is organized in a Jupyter notebook covering **Parts 0–6**: setup, data preparation, architecture inspection, training/validation functions, six full fine-tuning runs, test evaluation, and discussion. It uses PyTorch, `timm`, `medmnist`, NumPy, Pandas, Matplotlib, and scikit-learn.

The recorded environment used **Python 3.13.5**, **PyTorch 2.14.1**, **timm 1.0.31**, and **MedMNIST 3.0.2** on Apple Silicon MPS. Results may vary across environments. Run the notebook in order to recreate checkpoints and output files; the full notebook and setup instructions are available in the repository README.

| Material | Link |
|---|---|
| Executable Jupyter notebook | [Code page]({{ site.baseurl }}/code/) |
| Notebook with saved outputs (HTML) | [Code page]({{ site.baseurl }}/code/) |
| Technical report (PDF) | [Report page]({{ site.baseurl }}/report/) |
| Source code and reproduction instructions | [GitHub repository](https://github.com/uyentran9/pneumoniamnist-cnn-transformer) |
| Dataset and benchmark documentation | [MedMNIST / MedMNIST+](https://medmnist.com/) |

The technical report includes the complete method, per-run results, limitations, and references to the source dataset and model architectures.

## Author and intended use

**Victoria Tran (My-Uyen Tran)**  
Independent educational research project · 2026  
[Academic portfolio](https://uyentran9.github.io/uyentran-ai-blog/)

**Research-use notice:** This project is not a clinically validated diagnostic system and must not be used for medical diagnosis or medical decision-making.  

Thank you for visiting — and for being part of my journey 🌱
