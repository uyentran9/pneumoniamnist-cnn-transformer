---
layout: single
title: "Dataset"
permalink: /dataset/
author_profile: true
---

## PneumoniaMNIST+ 224

This project uses **PneumoniaMNIST+ 224**, a standardized dataset of **5,856 pediatric chest X-ray images** for binary classification: **Normal (0)** or **Pneumonia (1)**. It is the 224 × 224 version of PneumoniaMNIST in the MedMNIST+ benchmark. The higher-resolution version retains the same images and official train/validation/test splits as PneumoniaMNIST.

### Dataset Overview

| Property | Description |
|---|---|
| Modality | Pediatric chest X-rays |
| Task | Binary image classification |
| Images | 5,856 |
| Original dataset format | Grayscale, 224 × 224 pixels (1 channel) |
| Labels | Normal (0), Pneumonia (1) |
| Source | PneumoniaMNIST+, MedMNIST+; originally based on Kermany et al. (2018) |

### Sample X-ray Images

The examples below are from the training split and illustrate both classes.

![Examples of Normal and Pneumonia chest X-rays from the training set]({{ '/images/sample_xrays.png' | relative_url }})

## Official Splits and Class Distribution

The study uses the official splits without creating a new train/test partition.

| Split | Normal (0) | Pneumonia (1) | Total |
|---|---:|---:|---:|
| Train | 1,214 (25.79%) | 3,494 (74.21%) | 4,708 |
| Validation | 135 (25.76%) | 389 (74.24%) | 524 |
| Test | 234 (37.50%) | 390 (62.50%) | 624 |

![Class distribution across the official training, validation, and test splits]({{ '/images/class_distribution.png' | relative_url }})

Pneumonia is the majority class in all three splits, but its proportion decreases from approximately **74%** in training and validation to **62.5%** in test. Because accuracy alone can obscure class-specific errors, the study uses **macro-F1** as its primary model-selection and comparison metric and also reports sensitivity, specificity, accuracy, and ROC-AUC.

## Data Loading and Preprocessing

ResNet-18 and DeiT-Tiny use the **same preprocessing pipeline**:

1. Convert each grayscale image to three identical channels, as required by the ImageNet-pretrained models.
2. Apply `ToTensor()` to convert pixel values to `float32` in the range `[0, 1]`.
3. Normalize using the ImageNet channel mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

```python
from torchvision import transforms

shared_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Each processed image has shape **`(3, 224, 224)`**. The experiment uses a batch size of **16**, shuffles training data with a seeded generator, and does not shuffle validation or test data. **No random data augmentation** is applied. The complete loading and training code is available on the [Code page]({{ '/code/' | relative_url }}).

## Data Access

The dataset is **not redistributed in this repository**. It can be downloaded through the official MedMNIST distribution:

- [PneumoniaMNIST+ 224 dataset (.npz) — Zenodo](https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1)
- [MedMNIST website](https://medmnist.com/)
- [MedMNIST Python package and documentation](https://github.com/MedMNIST/MedMNIST)

The project notebook loads the official dataset using `medmnist.PneumoniaMNIST` with `size=224` and `download=True`.

## Dataset Usage and Limitations

- **Training set:** Used to update model parameters during full fine-tuning.
- **Validation set:** Used to select the best checkpoint for each model and random seed based on macro-F1.
- **Test set:** Used only for final evaluation after checkpoint selection.

MedMNIST creates its training and validation splits from the original source training data and uses the source validation data as its test split. The dataset is limited to a particular source and does not establish performance across other hospitals, patient populations, or imaging conditions. **This experiment is for research and education, not clinical use.**

## Attribution and References

PneumoniaMNIST is distributed under **CC BY 4.0**. Please acknowledge both MedMNIST and the original source dataset when reusing it.

1. **Yang, J., et al. (2023).** “MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification.” *Scientific Data*, 10, 41. [DOI](https://doi.org/10.1038/s41597-022-01721-8).
2. **Kermany, D. S., et al. (2018).** “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.” *Cell*, 172(5), 1122–1131.e9. [DOI](https://doi.org/10.1016/j.cell.2018.02.010).


---

<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
