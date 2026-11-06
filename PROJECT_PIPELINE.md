---
layout: single
title: "Experimental Design and Pipeline — Vanishing Gradients and Early-Layer Trainability in Deep MLPs"
permalink: /pipeline/
author_profile: true
toc: true
toc_label: "Experimental pipeline"
toc_icon: "cogs"
toc_sticky: true
---

> **Project:** Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST  
> **Author:** Victoria Tran (My-Uyen Tran) 
> **Research Area:** Deep Learning · Neural Network Optimization · Gradient Analysis  
> **Dataset:** BloodMNIST (MedMNIST v2)  
> **Framework:** PyTorch  
> **Research Objective:** Investigate how network depth and hidden-layer activation functions affect gradient attenuation, early-layer trainability, and optimization behavior in deep multilayer perceptrons.  
> **Experimental Design:** Two controlled stages — network-depth comparison (D2, D5, D10, D20) and activation-function comparison (Sigmoid, Tanh, ReLU at D5).  
> **Training Protocol:** Xavier Uniform initialization · SGD (learning rate 0.1) · 20 epochs · Three random seeds (0, 1, 2).  
> **Experimental Scope:** 18 unique main training runs, reusing the D5–Sigmoid configuration across both stages.  
> **Primary Measurements:** Layer-wise RMS weight gradients, first-to-last hidden-layer gradient ratios, normalized first-layer weight displacement, and training-loss trajectories.  
> **Secondary Evaluation:** Validation cross-entropy loss, classification accuracy, and macro-F1.  
> **Documentation Basis:** Final executed PyTorch notebook, Project Outline Version 4, and the accompanying 19-page technical report.  
> **Scope:** This document explains the implemented experimental pipeline, methodological decisions, measurements, and interpretation. The Jupyter notebook contains the executable implementation, while the technical report provides the complete scientific analysis.

---

**Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST**

*An empirical investigation of neural-network optimization using controlled architecture and activation-function comparisons.*

| Study attribute | Implementation |
|---|---|
| Dataset | BloodMNIST (MedMNIST v2); eight-class, 28 × 28 RGB microscopy images |
| Architecture | Configurable constant-width multilayer perceptron (MLP), 256 hidden units per layer |
| Stage A | Sigmoid MLPs with 2, 5, 10, or 20 hidden layers |
| Stage B | Sigmoid, Tanh, and ReLU MLPs at a fixed depth of 5 hidden layers |
| Controlled training protocol | Xavier Uniform initialization; plain SGD (learning rate 0.1); 20 epochs |
| Replication | Random seeds 0, 1, and 2; 18 unique main training runs |
| Primary measurements | Layer-wise RMS weight gradients, first-to-last gradient ratios, first-layer weight displacement, and training-loss progression |
| Secondary measurements | Validation cross-entropy, accuracy, and macro-F1 |

This page documents **how the study was implemented and evaluated**, connecting the research questions to the experimental controls, measurements, figures, and reported evidence. It is a methodological companion to the [executed notebook]({{ site.baseurl }}/code/) and [technical report]({{ site.baseurl }}/report/), not a substitute for either.

---

## 1. Research Objective and Scope

Vanishing gradients can limit the optimization of deep neural networks when gradients associated with earlier layers become extremely small. This study examines the relationship between **network depth, hidden-layer weight-gradient magnitude, early-layer parameter movement, and optimization progress** in fully connected MLPs trained on BloodMNIST.

BloodMNIST is used as a **controlled empirical testbed**. The aim is to investigate training dynamics, not to establish a state-of-the-art blood-cell classifier or a clinical diagnostic system.

### Research questions

**RQ1 — Network depth.** How does increasing network depth affect layer-wise weight-gradient magnitudes and the trainability of early layers in Sigmoid MLPs on BloodMNIST?

**RQ2 — Activation functions.** Under a common initialization and training protocol, how do Sigmoid, Tanh, and ReLU differ in relative gradient attenuation, optimization progress, and classification performance at a fixed network depth?

### Research hypotheses

- **H1 — Depth effect:** Increasing depth is expected to produce stronger attenuation of early-layer RMS weight gradients relative to later hidden layers under settings susceptible to vanishing gradients.
- **H2 — Early-layer trainability effect:** Conditions exhibiting stronger early-layer gradient attenuation are expected to show smaller normalized first-layer weight displacement and slower optimization progress.
- **H3 — Activation effect:** Under a common controlled initialization, Sigmoid and Tanh were hypothesized to exhibit stronger depth-dependent early-layer gradient attenuation than ReLU, with the magnitude of such differences potentially depending on initialization. Stage B tests activation-dependent differences **only at D5**, not activation-by-depth or activation-by-initialization interactions.

The two-stage design distinguishes the primary depth comparison from the narrower, fixed-depth activation comparison. This distinction is essential when interpreting the evidence for H3.

---

## 2. End-to-End Experimental Workflow

```text
Research questions and hypotheses
             |
             v
BloodMNIST official splits -> train-only normalization statistics
             |
             v
Configurable constant-width MLP + controlled initialization
             |
             v
Training, validation, and gradient-recording implementation
             |
             v
Smoke tests + D2/D20 learning-rate sanity pilot
             |
             v
Freeze learning rate and operational loss threshold
             |
             +-------------------------------+
             |                               |
             v                               v
Stage A: depth comparison          Stage B: activation comparison
D2 / D5 / D10 / D20                 D5: Sigmoid / Tanh / ReLU
Sigmoid, 3 seeds                    Reuse D5-Sigmoid, 3 seeds
             |                               |
             +---------------+---------------+
                             |
                             v
Layer-wise gradients + parameter displacement + loss trajectories
                             |
                             v
Run verification -> mean / sample SD -> figures and tables
                             |
                             v
Interpretation, limitations, and technical reporting
```

For direct correspondence with the executable implementation, the notebook is organized into **Parts 0–8**, summarized in Section 11 below.

---

## 3. Dataset and Leakage-Controlled Preprocessing

### Official BloodMNIST partitions

BloodMNIST contains **17,092 microscopic peripheral blood-cell images**, each represented by three RGB channels at **28 × 28** resolution, across **eight classes**. The official MedMNIST partitions were retained:

| Partition | Images | Use in this study |
|---|---:|---|
| Training | 11,959 | Optimize network parameters and record training-time gradient measurements |
| Validation | 1,712 | Evaluate cross-entropy loss, accuracy, and macro-F1 after each epoch |
| Test | 3,421 | Dataset verification only; not used for reported classification performance |

No additional data split or augmentation was used. Test-set classification metrics are **not** presented in the technical report.

### Preprocessing sequence

1. Load the raw **training partition** and compute channel-wise RGB means and standard deviations from its pixels only.
2. Convert images to PyTorch tensors with pixel intensities in `[0, 1]`.
3. Normalize each RGB channel using the **training-derived** mean and standard deviation. Apply these same statistics to training, validation, and test images.
4. Preserve input tensors with shape `[batch_size, 3, 28, 28]` in the DataLoader; flatten them inside the MLP into vectors of length **2,352**.
5. Shuffle the training DataLoader using a seeded PyTorch generator; do not shuffle validation or test data.

Training uses a mini-batch size of **128**; validation and test DataLoaders use **1,024**. Computing preprocessing statistics only on the training split avoids using held-out images to estimate the normalization parameters.

See the dedicated [Dataset page]({{ site.baseurl }}/dataset/) for data access and provenance.

---

## 4. MLP Architecture and Initialization

The PyTorch implementation uses a configurable, constant-width architecture:

```text
RGB image [3, 28, 28]
       |
       v
Flatten -> 2,352 input features
       |
       v
D fully connected hidden layers, 256 units each
       |  same activation after every hidden Linear layer
       v
Linear output layer -> 8 class logits
       |
       v
Multiclass cross-entropy loss
```

The number of hidden layers **D excludes the output layer**. D2, for example, has two hidden Linear layers followed by one eight-logit output layer. There is **no output Softmax layer**: `CrossEntropyLoss` receives the raw logits directly.

The architecture is intentionally simple. It includes **no convolution, batch normalization, dropout, or residual connections**, which would introduce additional sources of variation into the main comparisons.

### Initialization and controlled variables

All Linear weight matrices, **including the output layer**, are initialized with **Xavier Uniform, gain = 1**; biases are initialized to zero. The same rule is used in Stage A and across all Stage B activations.

This facilitates a comparison under a common initialization **rule**. It does not imply that Xavier initialization is optimal for every activation, particularly ReLU. Identical random-seed settings are used for corresponding configurations; outcomes are still interpreted as experimental comparisons rather than exact theoretical isolation of every effect.

Although hidden width remains fixed, **the total number of trainable parameters increases with depth**. The depth experiment therefore controls width but does not hold parameter count constant.

---

## 5. Training Protocol and Pilot Decisions

### Fixed main-experiment configuration

| Component | Final setting |
|---|---|
| Loss function | Multiclass cross-entropy |
| Optimizer | SGD without momentum or weight decay |
| Learning rate | 0.1, fixed across the main conditions |
| Epochs | 20 complete epochs per run |
| Training batch size | 128 |
| Random seeds | 0, 1, 2 |
| Weight initialization | Xavier Uniform, gain 1; zero biases |
| Device | Apple Silicon MPS in the recorded environment; CPU supported |

Plain SGD was selected because, without adaptive gradient rescaling, each parameter update is proportional to its gradient: `parameter_new = parameter_old - learning_rate × gradient`. This supports interpretation of gradient magnitude alongside parameter movement. It does **not** make weight-gradient magnitude a complete explanation of learning behavior.

### Preliminary learning-rate sanity pilot

Before the main experiment, the shallowest and deepest candidate Sigmoid models (**D2 and D20**) were each trained for **three full epochs using seed 0**, at a provisional learning rate of **0.1**. The pilot checked for non-finite training losses and obvious optimization instability. Neither architecture showed these failures, so the learning rate was retained rather than subjected to a broader search.

The same pilot trajectories informed a **training-loss threshold of 1.50**, which was fixed **before** the main 20-epoch experiments. The threshold is an **operational indicator of progress within the training budget**, not a mathematical convergence criterion.

### One training mini-batch

```text
Load mini-batch -> clear accumulated gradients
           -> forward pass -> compute cross-entropy
           -> backward pass (compute gradients)
           -> record selected hidden-layer gradients
           -> SGD parameter update
           -> accumulate sample-weighted training loss
```

The implementation clears gradients before backpropagation and records them **after `loss.backward()` but before `optimizer.step()`**. After the epoch, it records net first-layer weight displacement and evaluates the updated model on the full validation partition without gradient computation.

The epoch-level training loss is the **sample-weighted average of mini-batch losses observed during parameter updates**, rather than a second evaluation of one fixed end-of-epoch model on the training partition.

---

## 6. Gradient and Trainability Measurements

Measurements were designed to connect gradient attenuation with parameter movement and observable optimization progress. The numerical values refer specifically to **weight gradients of hidden Linear layers**; bias and output-layer gradients are excluded from the layer-wise comparison.

### 6.1 Layer-wise RMS weight gradients

For a hidden layer with weight-gradient tensor `G` containing `N` elements:

```text
RMS(G) = sqrt( sum(G_i^2) / N ) = ||G||_2 / sqrt(N)
```

This provides a per-parameter-scale gradient magnitude and reduces the dependence of the raw Euclidean norm on tensor element count. RMS still depends on layer dimensions and the distribution of activations entering the layer; it is **not a direct measurement of the activation-gradient Jacobian**.

The implementation uses a rescaled norm computation for numerical stability when gradients are extremely small.

### 6.2 Gradient logging schedule

A shared `GradientRecorder` collects:

- **Early-step records:** gradients for the **first 100 global training steps**.
- **Epoch-level records:** the **first 10 training mini-batches of every epoch**, averaged separately for each hidden layer and epoch.
- **First-layer weight displacement:** one measurement **after each epoch**.

The epoch-level aggregation is the **mean of 10 mini-batch RMS values**, not an RMS computed from the mean of 10 gradient tensors. The sampling scheme is identical across compared conditions but does not observe every gradient at every training step.

### 6.3 First-to-last hidden-layer gradient ratio

For each seed and epoch, the relative gradient magnitude is calculated as:

```text
R_epoch = mean_RMS_gradient(hidden layer 1, epoch)
          / mean_RMS_gradient(hidden layer D, epoch)
```

The principal cross-condition comparison uses **epoch 1**. Smaller values indicate weaker first-hidden-layer **weight gradients** relative to the last hidden layer. Ratios are computed **for each seed first**, then aggregated; the reported mean ratio is not obtained by dividing two across-seed mean gradients.

### 6.4 Normalized first-layer weight displacement

For the first hidden layer, the net parameter movement from initialization is:

```text
C_1(epoch) = ||W_1(epoch) - W_1(0)||_F / ||W_1(0)||_F
```

This is the **net displacement from initial weights**, not the sum of the magnitudes of all SGD updates. Opposing updates can partially cancel. In deep models, floating-point precision can also make extremely small updates unobservable in stored weights.

### 6.5 Training progress and validation performance

Optimization progress is assessed through the training-loss trajectory and the **first epoch whose mean training loss is at most 1.50**, if that threshold is reached in 20 epochs. Runs that do not meet it are reported as **Not reached**, not as proof that training could never improve under different settings.

Validation cross-entropy, accuracy, and **macro-F1** are recorded after each epoch. Macro-F1 assigns equal weight to the eight blood-cell classes. Final validation results refer to **epoch 20**.

All run-level metrics are summarized as **mean ± sample standard deviation across the three seeds**, where defined. No formal significance testing was performed.

---

## 7. Stage A — Controlled Network-Depth Comparison

**Research focus:** RQ1, H1, and H2.

The independent experimental variable is the number of hidden layers. Four **Sigmoid** architectures are evaluated:

| Configuration | Hidden layers | Activation | Independent training runs |
|---|---:|---|---:|
| D2 | 2 | Sigmoid | 3 seeds |
| D5 | 5 | Sigmoid | 3 seeds |
| D10 | 10 | Sigmoid | 3 seeds |
| D20 | 20 | Sigmoid | 3 seeds |

Dataset partitions, preprocessing, hidden width, initialization rule, optimizer, learning rate, training duration, and random-seed set remain fixed. **Stage A comprises 12 main training runs.**

The analysis uses three central figures:

1. **Figure 1 — Layer-wise gradient magnitude:** absolute and last-hidden-layer-normalized RMS weight gradients at epoch 1 across D2, D5, D10, and D20.
2. **Figure 2 — Gradient evolution:** a layer-by-epoch heatmap of `log10` mean RMS weight gradients in D20 over the full 20 epochs.
3. **Figure 3 — Training-loss trajectories:** mean training cross-entropy across depths, with across-seed variability and the fixed loss threshold of 1.50.

The figures, their captions, and the complete numerical interpretation appear in the [technical report]({{ site.baseurl }}/report/).

### Stage A: selected results

| Depth | First-to-last RMS gradient ratio, epoch 1 | First-layer displacement, epoch 20 | Training loss, epoch 20 | Epochs to loss ≤ 1.50 | Validation accuracy, epoch 20 |
|---|---:|---:|---:|---|---:|
| D2 | 0.4070 | 0.3279 | 0.3963 | 2.00 ± 0.00 (3/3) | 0.8123 |
| D5 | 0.006369 | 0.1090 | 1.2555 | 14.67 ± 0.58 (3/3) | 0.4663 |
| D10 | 4.837 × 10⁻⁶ | 2.740 × 10⁻⁵ | 2.0114 | Not reached (0/3) | 0.1820 |
| D20 | 2.711 × 10⁻¹² | 0.0000* | 2.0117 | Not reached (0/3) | 0.1820 |

All numeric point estimates above are **means across three seeds**. The technical report's **Table 1A** includes the corresponding sample standard deviations and validation macro-F1 scores. The threshold count in parentheses indicates seeds reaching the criterion. **\*D20's recorded zero displacement is limited by numerical precision and must not be interpreted as mathematically zero gradient or zero updates.**

The first-to-last hidden-layer RMS gradient ratio decreased from **0.4070 (D2)** to approximately **2.711 × 10⁻¹² (D20)**. The deepest Sigmoid models also exhibited much smaller first-layer weight displacement, slower training-loss reduction, and lower final validation performance under the **tested Sigmoid–Xavier–SGD protocol**. This observation supports H1 under the experimental conditions and H2 as an **association**, not a unique causal account.

---

## 8. Stage B — Fixed-Depth Activation Comparison

**Research focus:** RQ2 and H3.

Stage A showed that **D5** retained measurable optimization progress while exhibiting appreciable gradient attenuation. D5 was therefore selected for the follow-up activation comparison. Only the hidden-layer activation was deliberately varied:

- **Sigmoid**, the reused D5 baseline from Stage A;
- **Tanh**;
- **ReLU**.

All Stage B configurations used the same D5 architecture, initialization rule, SGD settings, data pipeline, 20-epoch budget, and seed set. The corresponding seeds align initialization and shuffling sequences across activation conditions.

There are **nine Stage B result entries**, but the three D5–Sigmoid entries were **reused**, not retrained. Thus:

```text
Stage A: 4 depths × 3 seeds                    = 12 main runs
Stage B: 2 additional activations × 3 seeds   =  6 new runs
                                                   ------
Total unique main runs                        = 18
```

### Stage B: selected results at D5

| Activation | First-to-last RMS gradient ratio, epoch 1 | First-layer displacement, epoch 20 | Training loss, epoch 20 | Validation accuracy, epoch 20 | Validation macro-F1, epoch 20 |
|---|---:|---:|---:|---:|---:|
| Sigmoid | 0.006369 | 0.1090 | 1.2555 | 0.4663 | 0.3377 |
| Tanh | 0.4750 | 0.3981 | 0.1934 | **0.8621** | **0.8441** |
| ReLU | 0.4954 | 0.4033 | 0.1992 | 0.8170 | 0.7946 |

All values are **means across three seeds**; sample standard deviations and the loss-threshold results appear in the technical report's **Table 1B**. All Tanh and ReLU runs reached the loss threshold by epoch 1, whereas the Sigmoid runs reached it at epochs 14–15.

Under the shared protocol, Tanh and ReLU exhibited substantially larger relative early-layer weight gradients and faster optimization than Sigmoid. **Tanh achieved the highest observed mean validation accuracy and macro-F1**, whereas **ReLU had the slightly higher mean relative gradient ratio**. Those measurements need not rank architectures identically. The comparison was descriptive and limited to three seeds, with **no statistical significance test** and **no activation-by-depth or initialization-interaction experiment**.

---

## 9. Experiment Verification and Reproducibility

The notebook separates smoke tests and pilot checks from the **full 20-epoch main runs**. Each main run is independently initialized and executed on the complete training DataLoader for the specified number of epochs. Python, NumPy, and PyTorch random seeds are controlled; the training loader uses a separately seeded generator.

Before final reporting, the implementation checks that:

- Stage A contains **12/12** expected depth–seed combinations.
- Stage B contains **9/9** expected activation–seed entries.
- Each main run has **20 epoch records**, and the epoch-level hidden-layer gradient records correspond to the configured depth and logging schedule.
- Epoch-level gradient records contain **10 contributing batches** for each hidden layer and epoch.
- Run configurations match the fixed optimizer, initialization, learning-rate, batch-size, and training-budget settings.
- Stage B's three D5–Sigmoid results match the Stage A baselines that were reused.
- Accounting for baseline reuse yields **18 unique main training runs**.
- The Stage A and B summary tables and the three figures are derived from the recorded experiments.

The final recorded environment used **Python 3.13, PyTorch 2.14.1, and MedMNIST 3.0.2**, with Apple Silicon **MPS** for computation and a CPU fallback in the implementation. Floating-point behavior and numerical results may vary with hardware, library versions, and execution backend.

### Reproduction procedure

1. Obtain BloodMNIST through the `medmnist` Python package. The raw dataset is not redistributed in this repository.
2. Install dependencies using the repository's `requirements.txt`, in a compatible Python environment.
3. Open the [executable Jupyter notebook]({{ site.baseurl }}/code/) and verify the relative input and output directories.
4. Restart the kernel and run the notebook **in order**, from Parts 0 through 8, using the official partitions and documented seeds.
5. Verify the pilot settings, the 18 unique main training runs, the expected gradient record counts, and the final Stage A/B summary tables.
6. Compare generated figures and reported metrics with the [technical report]({{ site.baseurl }}/report/).

The saved notebook outputs allow a reader to inspect the completed analyses without rerunning all models; they are **not** a substitute for executing the underlying source code when independently reproducing the work.

---

## 10. Scientific Interpretation and Limitations

The experiments provide evidence for **depth-dependent early-layer weight-gradient attenuation** in Sigmoid MLPs under the evaluated conditions, and an associated reduction in **net first-layer parameter movement** and **optimization progress**. They also identify substantial activation-dependent differences at the selected fixed depth.

The results warrant the following distinctions:

- **H1 — Supported within the tested configuration.** Stage A identifies progressively smaller early-layer RMS weight gradients relative to later hidden layers as Sigmoid MLP depth increases.
- **H2 — Supported as an empirical association.** Smaller early-layer weight gradients coincide with smaller net displacement and slower training-loss reduction; the observations do **not** isolate vanishing gradients as the sole cause.
- **H3 — Partially supported.** Activation-dependent differences are observed at D5, but the original hypothesis's depth- and initialization-dependent interactions remain untested.

The study has several limitations. It uses **one low-resolution image dataset**, fully connected **constant-width MLPs**, a limited set of depths and activations, **one shared Xavier initialization**, **plain SGD at one learning rate**, **20 epochs**, and **three seeds**. Increasing depth also increases parameter count. RMS **weight** gradients are not direct measurements of activation-gradient propagation, saturation, or Jacobian spectra. D20's recorded zero displacement is subject to floating-point precision. Only validation performance is reported, with **no independent test-set classification estimate**, confidence intervals, or formal significance tests.

These boundaries are important: the results support **empirical relationships under controlled conditions**, not universal claims about deep neural networks or clinical deployment.

### Directions for further investigation

Future experiments could investigate depth-by-activation interactions, Xavier versus activation-specific initialization, activation saturation and inactive ReLU units, residual connections, normalization, learning-rate and optimizer sensitivity, and higher-resolution data or CNN architectures. **These are proposed extensions, not completed results of this study.**

---

## 11. Notebook-to-Pipeline Map

The notebook's parts provide a direct audit trail from implementation to reported findings:

| Notebook part | Role in the experimental workflow |
|---|---|
| **Part 0 — Setup & Library Imports** | Prepare the software environment and core dependencies |
| **Part 1 — Data Pipeline & Experimental Configuration** | Load BloodMNIST, estimate training-only normalization statistics, configure official splits, inspect inputs and seeded loaders |
| **Part 2 — Configurable Constant-Width MLP** | Construct variable-depth models, initialize weights/biases, and verify shapes and reproducibility |
| **Part 3 — Training & Validation Pipeline** | Define cross-entropy, SGD, training and validation loops, and initial training checks |
| **Part 4 — Gradient & Trainability Measurements** | Implement stable RMS gradients, first-layer displacement, and the `GradientRecorder` logging protocol |
| **Part 5 — Pipeline Verification & Protocol Finalization** | Run smoke checks and D2/D20 learning-rate pilot; freeze learning rate and loss threshold |
| **Part 6 — Stage A: Network Depth Experiment** | Execute D2/D5/D10/D20 comparisons; produce Figures 1–3 and depth results |
| **Part 7 — Stage B: Activation Function Experiment** | Select D5, compare Sigmoid/Tanh/ReLU, and aggregate activation-dependent measurements |
| **Part 8 — Final Analysis & Scientific Reporting** | Verify run completeness and baseline reuse; assemble results tables, interpretations, and scientific limitations |

The figures and summary tables are explained in full in the **19-page technical report**, together with the supporting theoretical literature and discussion of the strength and limits of the evidence.

---

## Research Materials

- [Dataset and data-access instructions]({{ site.baseurl }}/dataset/)
- [Jupyter notebook, HTML preview, and source code]({{ site.baseurl }}/code/)
- [Technical report (PDF)]({{ site.baseurl }}/report/)
- [GitHub repository](https://github.com/uyentran9/gradient-vanishing)

<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  

