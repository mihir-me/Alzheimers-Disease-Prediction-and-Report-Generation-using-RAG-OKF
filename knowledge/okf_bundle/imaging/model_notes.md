---
type: Imaging Biomarker
generated: { by: opencode/big-pickle, at: 2026-10-03T13:00:04Z }
title: "Practical Notes for This Model"
description: "The four MRI models in this system (ResNet-152, VGG16, EfficientNet-B4, and"
tags: ["dementia", "cdr", "atrophy", "imaging", "mri"]
sources:
  - knowledge/imaging_biomarkers.md
---
# Practical Notes for This Model

The four MRI models in this system (ResNet-152, VGG16, EfficientNet-B4, and
Vision Transformer) were trained on 2D brain MRI slices and classify each slice
into one of four classes: No Impairment, Very Mild, Mild, or Moderate
Impairment. The ensemble combines the four softmax outputs with a logistic
regression meta-learner to produce the final MRI stage prediction.

Related: [MRI Atrophy](/imaging/mri_biomarkers.md), [Dementia Staging](/overview/clinical_stages_cdr.md).
