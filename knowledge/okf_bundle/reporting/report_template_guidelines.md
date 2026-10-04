---
type: Report Guideline
generated: { by: opencode/big-pickle, at: 2026-10-03T13:00:04Z }
title: "Report Structure Guidelines"
description: "This document defines the required structure for the auto-generated clinical"
tags: ["dementia", "cdr", "mmse", "nwbv", "atrophy"]
sources:
  - knowledge/report_template.md
---
# Report Structure Guidelines

This document defines the required structure for the auto-generated clinical
report. Always follow these sections in order and use the exact section
headings.

## Required Sections

1. **Clinical Summary** — a 2-3 sentence plain-language summary of the predicted
   cognitive stage and overall impression.

2. **Imaging (MRI) Analysis** — summarize the MRI findings and the model's
   per-class probabilities for No Impairment, Very Mild, Mild, and Moderate.
   Mention hippocampal/whole-brain atrophy only as supported by the knowledge
   base. Do not invent numeric findings.

3. **Clinical Scores Interpretation** — interpret the provided MMSE, CDR, and
   nWBV values against standard thresholds. State what the combination suggests.

4. **Risk Interpretation** — connect the prediction to relevant risk factors and
   epidemiology from the knowledge base.

5. **Recommended Next Steps** — concrete, actionable recommendations consistent
   with the predicted stage (specialist referral, cognitive testing, vascular
   risk control, caregiver support, medication discussion).

6. **References / Knowledge Sources** — list the knowledge base documents that
   were retrieved and used.

7. **Disclaimer** — a short disclaimer that this is a decision-support output,
   not a clinical diagnosis, and that a qualified clinician must confirm.

## Style Rules

- Use clear markdown headings and bullet points.
- Be conservative and accurate; never state probabilities you do not have.
- Tailor recommendations to the predicted stage.
- Keep the whole report readable by both clinicians and patients.
- Cite the retrieved knowledge chunks by source document name.

Related: [MMSE](/scores/mmse.md), [CDR Scale](/scores/cdr_scale.md), [nWBV](/scores/nwbv.md), [MRI Atrophy](/imaging/mri_biomarkers.md), [Dementia Staging](/overview/clinical_stages_cdr.md), [APOE4 Risk](/risk/risk_factors.md).
