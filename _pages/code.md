---
layout: single
title: "Code"
permalink: /code/
author_profile: true
---

This page provides the PyTorch implementation and executed Jupyter Notebook for **From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST**.

The notebook contains the code, explanations, and recorded results for three models: an MLP baseline (M0), a CNN with a Flatten classifier head (M1), and a CNN with Global Average Pooling (M2). The study compares classification performance and trainable parameter counts using the official BloodMNIST splits.

---

## Jupyter Notebook — Executable Source

The original `.ipynb` file includes dataset inspection, training-derived RGB normalization, model definitions, training and validation, test evaluation, learning curves, and the M2 confusion matrix. The models were trained using one predefined random seed and shared training settings.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ '/files/bloodmnist-mlp-cnn-uyentran.ipynb' | relative_url }}"
     download>
    ⬇ Download .ipynb
  </a>

  <a class="code-btn code-secondary-btn"
     href="https://github.com/uyentran9/bloodmnist-mlp-cnn/blob/HEAD/files/bloodmnist-mlp-cnn-uyentran.ipynb"
     target="_blank"
     rel="noopener noreferrer">
    ↗ View on GitHub
  </a>
</div>

---

## Executed Notebook — HTML Preview

The HTML export displays the notebook's saved Markdown explanations, code, outputs, tables, and figures directly in the browser. You can read the results without rerunning the training experiments. To reproduce the study, use the executable `.ipynb` file above.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ '/files/bloodmnist-mlp-cnn-uyentran.html' | relative_url }}"
     download>
    ⬇ Download HTML
  </a>

  <a class="code-btn code-secondary-btn"
     href="{{ '/files/bloodmnist-mlp-cnn-uyentran.html' | relative_url }}"
     target="_blank"
     rel="noopener noreferrer">
    ↗ Open full-screen
  </a>
</div>

<div class="notebook-wrap">
  <iframe
    src="{{ '/files/bloodmnist-mlp-cnn-uyentran.html' | relative_url }}"
    title="BloodMNIST MLP vs. CNN — Executed Jupyter Notebook"
    loading="lazy">
  </iframe>
</div>

<style>
/* Action buttons */
.code-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75em;
  margin: 1em 0 1.5em;
}

.code-actions .code-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 4px;
  text-decoration: none !important;
  transition: background-color 0.2s ease;
}

/* Fixed button colors remain legible in light and dark themes. */
.code-actions .code-primary-btn,
.code-actions .code-primary-btn:visited {
  background: #1f4d7a !important;
  color: #ffffff !important;
  border: 1px solid #1f4d7a !important;
}

.code-actions .code-secondary-btn,
.code-actions .code-secondary-btn:visited {
  background: #eef5fb !important;
  color: #1f4d7a !important;
  border: 1px solid #1f4d7a !important;
}

.code-actions .code-primary-btn:hover,
.code-actions .code-primary-btn:focus-visible {
  background: #173b5e !important;
  color: #ffffff !important;
  border-color: #173b5e !important;
}

.code-actions .code-secondary-btn:hover,
.code-actions .code-secondary-btn:focus-visible {
  background: #dcebf7 !important;
  color: #173b5e !important;
  border-color: #173b5e !important;
}

.code-actions .code-btn:focus-visible {
  outline: 2px solid #1f4d7a;
  outline-offset: 3px;
}

/* Notebook preview */
.notebook-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background: #ffffff;
}

.notebook-wrap iframe {
  display: block;
  width: 100%;
  height: 85vh;
  min-height: 700px;
  border: 0;
  background: #ffffff;
}

@media (max-width: 600px) {
  .notebook-wrap iframe {
    height: 75vh;
    min-height: 450px;
  }
}
</style>

---


<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
