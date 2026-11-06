---
layout: single
title: "Technical Report"
permalink: /report/
author_profile: true
---

## Technical Report

This three-page technical report documents the experimental design, model architectures, training and evaluation protocol, results, and limitations of **From MLP to CNN: Spatial Feature Learning and Classifier-Head Efficiency on BloodMNIST**.

The study compares an MLP baseline (M0) with two CNN architectures (M1 with Flatten and M2 with Global Average Pooling). It examines classification performance and trainable parameter efficiency using the official BloodMNIST splits and a fixed, single-seed training protocol. The report includes validation and test results, M1 learning curves, and M2 test confusion-matrix analysis.

<div class="report-actions">
  <a class="report-btn"
     href="{{ '/files/bloodmnist_technical_report_uyentran.pdf' | relative_url }}"
     download>
    ⬇ Download PDF
  </a>

  <a class="report-btn report-secondary-btn"
     href="{{ '/files/bloodmnist_technical_report_uyentran.pdf' | relative_url }}"
     target="_blank"
     rel="noopener noreferrer">
    ↗ Open full-screen
  </a>
</div>

<div id="report-wrap" class="report-wrap">
  <iframe
    id="report-frame"
    src="{{ '/files/bloodmnist_technical_report_uyentran.pdf' | relative_url }}"
    title="BloodMNIST MLP vs. CNN — Technical Report (PDF)"
    loading="lazy">
  </iframe>
</div>

<style>
/* Button layout */
.report-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75em;
  margin: 1em 0 1.5em;
}

/* Shared button styling */
.report-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 4px;
  text-decoration: none !important;
}

/* Primary button: Download PDF */
.report-btn:not(.report-secondary-btn),
.report-btn:not(.report-secondary-btn):visited {
  background: #1f4d7a !important;
  color: #ffffff !important;
  border: 1px solid #1f4d7a !important;
}

/* Secondary button: Open full-screen */
.report-secondary-btn,
.report-secondary-btn:visited {
  background: #eef5fb !important;
  color: #1f4d7a !important;
  border: 1px solid #1f4d7a !important;
}

/* Primary button hover and keyboard focus */
.report-btn:not(.report-secondary-btn):hover,
.report-btn:not(.report-secondary-btn):focus-visible {
  background: #173b5e !important;
  color: #ffffff !important;
  border-color: #173b5e !important;
}

/* Secondary button hover and keyboard focus */
.report-secondary-btn:hover,
.report-secondary-btn:focus-visible {
  background: #dcebf7 !important;
  color: #173b5e !important;
  border-color: #173b5e !important;
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

/* Shown only if the PDF is unavailable */
.report-missing {
  padding: 3em 1.5em;
  text-align: center;
  color: var(--global-text-color-light, #888);
}
</style>

<script>
(function () {
  var url = "{{ '/files/bloodmnist_technical_report_uyentran.pdf' | relative_url }}";

  fetch(url, { method: 'HEAD' })
    .then(function (res) {
      if (!res.ok) throw new Error('PDF unavailable');
    })
    .catch(function () {
      document.getElementById('report-wrap').innerHTML =
        '<div class="report-missing">The technical report PDF is not available yet. ' +
        'Please check that the file has been uploaded to <code>files/</code>.</div>';
    });
})();
</script>


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


<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
