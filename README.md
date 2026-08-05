# Instance-Visible Attribute-Guided Prompt Tuning for Vision-Language Models

This repository contains the implementation of the paper:

**Instance-Visible Attribute-Guided Prompt Tuning for Vision-Language Models**

## Overview

Vision-language models (VLMs) achieve strong transferability through large-scale image-text pre-training. However, existing prompt tuning methods mainly rely on static class-level semantic priors, which may contain attributes that are not visually supported by individual instances.

This work proposes Instance-Visible Attribute-Guided Prompt Tuning (IAPT), which introduces instance-visible attribute evidence to improve the reliability of vision-language representations.

The main components include:

- Global-to-local instance-visible attribute retrieval;
- Evidence-guided visual representation calibration;
- Visible Attribute Evidence Memory (VEM) for cross-instance evidence aggregation;
- Evidence-guided textual representation calibration.

## Environment

The code is implemented with Python and PyTorch.

Required packages:

```bash
pip install -r requirements.txt
