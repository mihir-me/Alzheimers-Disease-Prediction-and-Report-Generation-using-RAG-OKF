---
type: Clinical Score
generated: { by: opencode/big-pickle, at: 2026-10-03T13:00:04Z }
title: "MMSE (Mini-Mental State Examination)"
description: "The MMSE is a 30-point cognitive screening instrument that assesses orientation,"
tags: ["dementia", "cdr", "mmse", "staging"]
sources:
  - knowledge/clinical_scores.md
---
# MMSE (Mini-Mental State Examination)

The MMSE is a 30-point cognitive screening instrument that assesses orientation,
registration, attention/calculation, recall, and language.

- **30 points** is the maximum (best) score.
- Scores are commonly interpreted as:
  - 26-30: normal
  - 21-25: mild cognitive impairment
  - 10-20: moderate impairment
  - 0-9: severe impairment
- In this system the MMSE is normalized as `MMSE / 30` before being fed to the
  clinical model. A lower MMSE generally indicates greater cognitive decline.

Related: [CDR Scale](/scores/cdr_scale.md), [Dementia Staging](/overview/clinical_stages_cdr.md).

