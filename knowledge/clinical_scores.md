# Clinical Scores: MMSE, CDR, nWBV

## MMSE (Mini-Mental State Examination)

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

## CDR (Clinical Dementia Rating)

The CDR is a 5-point scale used to stage the severity of dementia.

- **CDR 0** — No dementia (normal).
- **CDR 0.5** — Questionable / very mild dementia (MCI).
- **CDR 1** — Mild dementia.
- **CDR 2** — Moderate dementia.
- **CDR 3** — Severe dementia.

The CDR combines ratings across six domains: memory, orientation, judgment and
problem solving, community affairs, home and hobbies, and personal care. Memory
is weighted most heavily.

## nWBV (Normalized Whole Brain Volume)

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
