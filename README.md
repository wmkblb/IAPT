# Instance-Visible Attribute-Guided Prompt Tuning for Vision-Language Models

Official implementation of:

**Instance-Visible Attribute-Guided Prompt Tuning for Vision-Language Models**

This repository provides the implementation of IAPT, a reliable vision-language representation learning framework that introduces instance-visible attribute evidence to reduce the mismatch between static textual priors and instance-level visual evidence.

## Overview

Vision-language models (VLMs) provide strong transferable representations through large-scale image-text pre-training. However, existing prompt tuning methods mainly rely on class-level semantic priors, which may contain attributes that are not visually supported by individual images.

IAPT addresses this issue by:
- retrieving instance-visible attributes through global-to-local visual evidence verification;
- calibrating visual representations with attribute-guided residual learning;
- storing cross-instance attribute evidence through Visible Attribute Evidence Memory (VEM);
- enhancing textual representations with evidence-aware calibration.

## Installation

The code is implemented with Python and PyTorch.

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3

Install dependencies:

```bash
pip install -r requirements.txt
