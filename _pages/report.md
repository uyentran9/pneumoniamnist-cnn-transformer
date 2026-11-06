---
layout: single
title: "Technical Report"
permalink: /report/
author_profile: true
---

The technical report documents the experimental design, model architectures, training protocol, results, and limitations of **Comparative Evaluation of ImageNet-Pretrained ResNet-18 and DeiT-Tiny for Pediatric Pneumonia Classification**.

This study compares two ImageNet-pretrained models on **PneumoniaMNIST+ 224** using full fine-tuning and the official dataset splits. Each model is trained with three random seeds (42, 43, 44). The report presents validation-based checkpoint selection, held-out test evaluation, training curves, confusion matrices, and a discussion of the findings and limitations.

<div class="report-actions">
  <a class="report-btn report-primary-btn"
     href="{{ '/files/pneumoniamnist_cnn_transformer_uyentran.pdf' | relative_url }}"
     download>
    ⬇ Download PDF
  </a>

  <a class="report-btn report-secondary-btn"
     href="{{ '/files/pneumoniamnist_cnn_transformer_uyentran.pdf' | relative_url }}"
     target="_blank"
     rel="noopener noreferrer">
    ↗ Open full-screen
  </a>
</div>

<div class="report-wrap">
  <iframe
    src="{{ '/files/pneumoniamnist_cnn_transformer_uyentran.pdf' | relative_url }}"
    title="PneumoniaMNIST+ 224 — ResNet-18 vs. DeiT-Tiny Technical Report (PDF)"
    loading="lazy">
  </iframe>
</div>

<style>
/* PDF action buttons */
.report-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75em;
  margin: 1em 0 1.5em;
}

.report-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 4px;
  text-decoration: none !important;
}

/* Colors remain readable in light and dark themes. */
.report-primary-btn,
.report-primary-btn:visited {
  background: #1f4d7a !important;
  color: #ffffff !important;
  border: 1px solid #1f4d7a !important;
}

.report-secondary-btn,
.report-secondary-btn:visited {
  background: #eef5fb !important;
  color: #1f4d7a !important;
  border: 1px solid #1f4d7a !important;
}

.report-primary-btn:hover,
.report-primary-btn:focus-visible {
  background: #173b5e !important;
  border-color: #173b5e !important;
  color: #ffffff !important;
}

.report-secondary-btn:hover,
.report-secondary-btn:focus-visible {
  background: #dcebf7 !important;
  border-color: #173b5e !important;
  color: #173b5e !important;
}

.report-btn:focus-visible {
  outline: 2px solid #1f4d7a;
  outline-offset: 3px;
}

/* Embedded PDF preview */
.report-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background: #ffffff;
}

.report-wrap iframe {
  display: block;
  width: 100%;
  height: 90vh;
  min-height: 700px;
  border: 0;
}

@media (max-width: 600px) {
  .report-wrap iframe {
    height: 75vh;
    min-height: 450px;
  }
}
</style>


<!--  
## Presentation Slides

<p>Slides summarizing the project's motivation, methodology, and key results.</p>
<div class="report-actions">
  <a class="report-btn" href="{{ site.baseurl }}/files/UyenTran_Hepatic_Risk_Prediction_Slides.pptx" download>⬇ Download the .pptx file</a>
  <a class="report-btn report-secondary-btn" href="https://view.officeapps.live.com/op/embed.aspx?src=https%3A%2F%2Fuyentran9.github.io%2Fhepatic-risk-prediction%2Ffiles%2FUyenTran_Hepatic_Risk_Prediction_Slides.pptx" target="_blank" rel="noopener">↗ Open full-screen</a>
</div>
<div class="report-wrap">
  <iframe
    src="https://view.officeapps.live.com/op/embed.aspx?src=https%3A%2F%2Fuyentran9.github.io%2Fhepatic-risk-prediction%2Ffiles%2FUyenTran_Hepatic_Risk_Prediction_Slides.pptx"
    loading="lazy"
    frameborder="0">
  </iframe>
</div> 

<!--
<iframe
  src="https://view.officeapps.live.com/op/embed.aspx?src=https%3A%2F%2Fuyentran9.github.io%2Fhepatic-risk-prediction%2Ffiles%2FUyenTran_Hepatic_Risk_Prediction_Slides.pptx"
  width="100%"
  height="600"
  frameborder="0">
</iframe>

[⬇ Download the .pptx file](/hepatic-risk-prediction/files/UyenTran_Hepatic_Risk_Prediction_Slides.pptx)
-->

---  

<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
