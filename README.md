# PRISM: Training-Free Video Anomaly Detection via Intrinsic Statistical Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of:

**PRISM: Training-Free Video Anomaly Detection via Intrinsic Statistical Modeling**

PRISM (**P**arameter-less **R**ecognition based on **I**ntrinsic **S**tatistical **M**odeling) is a training-free, text-guided framework for video anomaly detection. It uses frozen multimodal embedding models and constructs decorrelated semantic anomaly axes through text-side statistical whitening.

PRISM does not require video-level anomaly labels, model fine-tuning, or high-latency generative VLM inference. Instead, it performs anomaly scoring by projecting video embeddings onto calibrated text-derived semantic axes.

---

## 🌟 Highlights

- **Training-Free and Zero-Shot**

  PRISM does not train task-specific detection heads and does not fine-tune the multimodal encoder. All video and text embeddings are extracted using frozen pretrained models.

- **Text-Side Semantic Whitening**

  In a strict zero-shot setting, target-domain video distributions are not assumed to be available. Therefore, PRISM estimates covariance only from text-side semantic descriptions and uses the whitening transform to construct decorrelated anomaly axes.

- **Efficient Inference**

  After video embeddings are extracted, PRISM only requires lightweight vector operations, including centering, dot-product projection, softmax-weighted semantic aggregation, and temporal smoothing.

- **Interpretable Semantic Axes**

  Anomaly scores are computed through alignment with text-derived anomaly directions, making the detection process transparent and easy to analyze.

---

## 🔍 Method Overview

Given a frozen multimodal embedding model, PRISM performs anomaly detection in the following steps:

1. Encode normal and anomaly text descriptions.
2. Compute the normal semantic center.
3. Estimate text-side covariance from the description pool.
4. Construct a regularized whitening matrix.
5. Build anomaly-specific semantic axes from whitened anomaly-normal directions.
6. Extract clip-level video embeddings using a frozen encoder.
7. Center video embeddings using the normal text center.
8. Project video embeddings onto the calibrated semantic axes.
9. Aggregate multi-axis anomaly logits with temperature-controlled semantic weighting.
10. Apply Gaussian temporal smoothing to obtain final frame-level anomaly scores.

Importantly, PRISM applies whitening only to the text-derived semantic axes. Video embeddings are not whitened because no target-domain video distribution is assumed available in the zero-shot setting.

---

## 📁 Repository Structure

```text
ICML2026-PRISM/
├── Code/
│   ├── Qwen_ucf.py
│   ├── Qwen_xd.py
│   ├── Qwen_msad.py
│   ├── PE_ucf.py
│   ├── PE_xd.py
│   ├── qwen_tsnet&cos_visual_ucf.py
│   └── qwen_tsnet&cos_visual_xd.py
│
├── feature_extraction/
│   ├── extract_qwen_ucf.py
│   ├── extract_qwen_xd.py
│   ├── extract_pe_ucf.py
│   └── extract_pe_xd.py
│
├── semantic_axis/
│   ├── ucf_descriptions.json
│   ├── xd_descriptions.json
│   └── msad_descriptions.json
│
├── picture/
├── README.md
└── LICENSE
