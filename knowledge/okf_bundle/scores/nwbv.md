---
type: Clinical Score
generated: { by: opencode/big-pickle, at: 2026-10-03T13:00:04Z }
title: "nWBV (Normalized Whole Brain Volume)"
description: "nWBV is the whole brain volume normalized by the subject's total intracranial"
tags: ["cdr", "mmse", "nwbv", "atrophy", "imaging"]
sources:
  - knowledge/clinical_scores.md
---
# nWBV (Normalized Whole Brain Volume)

nWBV is the whole brain volume normalized by the subject's total intracranial
volume (from the OASIS dataset). It is expressed as a fraction (roughly 0.65 to
0.85 in healthy older adults).

- **Lower nWBV** indicates greater brain atrophy, which is a hallmark of
  Alzheimer's disease.
- Normalization corrects for differences in head size, making the measure
  comparable between individuals.

## How the Model Uses These

The clinical fusion branch (ClinicalFusionNet) receives three inputs:

1. `mmse_norm = MMSE / 30`
2. `cdr_norm` (standardized CDR)
3. `nwbv_norm` (standardized nWBV)

and outputs a DEMENTED vs NON-DEMENTED probability. Demented (class 0) is
associated with lower MMSE, higher CDR, and lower nWBV.

See also: [MMSE](/scores/mmse.md), [CDR](/scores/cdr_scale.md), [MRI Atrophy](/imaging/mri_biomarkers.md).