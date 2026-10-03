# MRI Imaging Biomarkers of Alzheimer's Disease

## Medial Temporal Lobe Atrophy

Medial temporal lobe (MTL) atrophy, especially of the hippocampus and entorhinal
cortex, is the earliest and most reproducible structural MRI finding in
Alzheimer's disease. It can be rated visually (e.g., the Scheltens scale) or
measured volumetrically.

## Whole-Brain Atrophy and Ventricular Enlargement

As the disease progresses, cortical atrophy spreads to the temporal, parietal,
and later frontal lobes. The lateral ventricles enlarge (ex-vacuo dilation) as
surrounding tissue is lost. Whole-brain atrophy correlates with nWBV, which
decreases with disease severity.

## Pattern of Atrophy by Stage

- **No Impairment:** normal or near-normal brain volume; no significant
  hippocampal atrophy.
- **Very Mild (MCI):** early hippocampal volume loss; subtle whole-brain
  atrophy.
- **Mild dementia:** obvious MTL atrophy, mild cortical atrophy, slight
  ventricular enlargement.
- **Moderate dementia:** pronounced bilateral hippocampal and cortical atrophy,
  marked ventricular enlargement, reduced white matter volume.

## Other Imaging Features

- White matter hyperintensities (small vessel disease) are common and are a
  modifier, though not a specific AD biomarker.
- FDG-PET and amyloid PET (Pittsburgh compound B / florbetapir) are functional
  and molecular complements to structural MRI but are not part of this system.

## Practical Notes for This Model

The four MRI models in this system (ResNet-152, VGG16, EfficientNet-B4, and
Vision Transformer) were trained on 2D brain MRI slices and classify each slice
into one of four classes: No Impairment, Very Mild, Mild, or Moderate
Impairment. The ensemble combines the four softmax outputs with a logistic
regression meta-learner to produce the final MRI stage prediction.
