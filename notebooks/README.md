# Experimental Notebook

This directory contains the Jupyter notebook for *Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST*.

The notebook provides the PyTorch implementation, experimental procedures, recorded training results, and analyses underlying the accompanying technical report.

## Experimental Design

The study consists of two controlled experimental stages:

**Stage A — Network Depth Comparison**

Compares multilayer perceptrons (MLPs) with 2, 5, 10, and 20 hidden layers under Sigmoid activation to examine depth-dependent gradient attenuation and early-layer trainability.

**Stage B — Activation Function Comparison**

Compares Sigmoid, Tanh, and ReLU at a fixed depth of five hidden layers under a shared initialization and optimization protocol.

All configurations use Xavier Uniform initialization, stochastic gradient descent (SGD) with a learning rate of 0.1, and 20 training epochs. Experiments use three random seeds (0, 1, and 2), yielding 18 unique main training runs after reusing the D5–Sigmoid baseline across stages.

## Measurements and Evaluation

The notebook records:

- Layer-wise root-mean-square (RMS) weight-gradient magnitudes.
- First-to-last hidden-layer RMS gradient ratios.
- Normalized first-layer weight displacement from initialization.
- Training-loss trajectories and epochs required to reach a predefined loss threshold of 1.50.
- Validation cross-entropy loss, classification accuracy, and macro-F1.

Reported results are aggregated across three random seeds using the mean and sample standard deviation.

## Dataset

Experiments use BloodMNIST from the MedMNIST v2 benchmark, consisting of 28 × 28 RGB images across eight blood-cell classes. The official training, validation, and test partitions are retained.

The test partition is loaded for verification but is not used for final classification-performance reporting.

## Reproducibility

The notebook includes the model definitions, initialization procedures, training loops, gradient-recording logic, and analysis code. Recorded outputs are retained to allow inspection of the experimental results without rerunning training.

Dependencies are documented in the repository-level `requirements.txt`. Dataset loading uses the MedMNIST package.

Reproducing the experiments requires executing the notebook in a compatible Python environment. Numerical results may vary across hardware backends and software versions.

## Technical Report

The complete methodology, experimental results, discussion, limitations, and references are documented in the accompanying technical report located in the repository's `reports/` directory.
