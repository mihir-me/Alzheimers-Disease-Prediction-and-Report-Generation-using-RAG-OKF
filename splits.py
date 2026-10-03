"""Build subject-disjoint train/val/test splits for retraining.

- OASIS is used for validation and test (subject-disjoint per class).
- KAGGLE train images (no subject info available) are all used for training.
- A subject-disjoint slice of OASIS is also added to training so the models
  learn the OASIS domain (user requirement: fine-tune + OASIS in train).
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).resolve().parent
DATASETS_PATH = BASE_PATH / "datasets"
KAGGLE_TRAIN = DATASETS_PATH / "KAGGLE DATASET" / "Combined Dataset" / "train"
OASIS_PATH = DATASETS_PATH / "OASIS DATASET" / "Data"
RESULTS_PATH = BASE_PATH / "results"
MANIFEST_PATH = RESULTS_PATH / "split_manifest.json"
CLINICAL_DATA_FILENAMES = ("brainscore.csv", "alzheimer.csv")


def clinical_data_path() -> Path:
    for filename in CLINICAL_DATA_FILENAMES:
        path = DATASETS_PATH / filename
        if path.is_file():
            return path
    expected = ", ".join(str(DATASETS_PATH / filename) for filename in CLINICAL_DATA_FILENAMES)
    raise FileNotFoundError(f"Clinical dataset not found; checked: {expected}")

SEED = 42

# canonical class order (index 0..3)
CLASSES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

# OASIS folder -> canonical class
OASIS_FOLDER_MAP = {
    "Mild Dementia": "Mild Impairment",
    "Moderate Dementia": "Moderate Impairment",
    "Non Demented": "No Impairment",
    "Very mild Dementia": "Very Mild Impairment",
}

SUBJECT_RE = re.compile(r"(OAS\d+_\d+)_")

# per-class train slice cap from OASIS (keep the set balanced + bounded)
OASIS_TRAIN_CAP = 1600
OASIS_VAL_CAP = 600
OASIS_TEST_CAP = 600

# per-subject slice caps: slices from one subject are near-identical, so cap
# them to force the model to learn across subjects instead of memorising one scan
MAX_TRAIN_SLICES_PER_SUBJECT = 40
MAX_EVAL_SLICES_PER_SUBJECT = 100

# The OASIS dataset has only 2 Moderate Dementia subjects total. If both stay
# in val/test the model never sees an OASIS-domain Moderate scan and cannot
# classify one. Move one subject into training (sacrificing an independent
# Moderate test subject, which is unavoidable with only 2 subjects).
SPLIT_OVERRIDES = {
    "Moderate Dementia": {
        "train_subjects": ["OAS1_0308"],
        "val_subjects": ["OAS1_0351"],
        "test_subjects": [],
    },
}


def subject_id(filename):
    m = SUBJECT_RE.match(filename)
    return m.group(1) if m else None


def list_images(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def build_oasis_split(seed=SEED):
    """Subject-disjoint split of OASIS -> train/val/test subject lists per class."""
    rng = random.Random(seed)
    result = {}
    for folder, cls in OASIS_FOLDER_MAP.items():
        folder_path = os.path.join(OASIS_PATH, folder)
        images = list_images(folder_path)
        subjects = {}
        for img in images:
            sid = subject_id(img)
            if sid is not None:
                subjects.setdefault(sid, []).append(img)
        subject_ids = sorted(subjects.keys())
        rng.shuffle(subject_ids)

        n = len(subject_ids)
        if n < 3:
            # too few subjects: keep 1 for val and 1 for test, none for train
            # (the class is still covered in training via the KAGGLE split)
            n_val = 1 if n >= 1 else 0
            n_test = 1 if n >= 2 else 0
            val_ids = subject_ids[:n_val]
            test_ids = subject_ids[n_val:n_val + n_test]
            train_ids = subject_ids[n_val + n_test:]
        else:
            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.15))
            if n_val < 1:
                n_val = 1
            n_test = max(n - n_train - n_val, 0)
            # adjust so test gets at least 1 when possible
            if n_test == 0 and n_train >= 2:
                n_train -= 1
                n_test = 1
            train_ids = subject_ids[:n_train]
            val_ids = subject_ids[n_train:n_train + n_val]
            test_ids = subject_ids[n_train + n_val:]

        result[cls] = {
            "train_subjects": sorted(train_ids),
            "val_subjects": sorted(val_ids),
            "test_subjects": sorted(test_ids),
            "all_images": {s: subjects[s] for s in subject_ids},
        }

    for folder, override in SPLIT_OVERRIDES.items():
        cls = OASIS_FOLDER_MAP[folder]
        result[cls]["train_subjects"] = override["train_subjects"]
        result[cls]["val_subjects"] = override["val_subjects"]
        result[cls]["test_subjects"] = override["test_subjects"]

    return result


def sample_slices(image_lists, cap, rng, per_subject_cap=None):
    """Sample up to `cap` images from a list of per-subject image lists."""
    if per_subject_cap is not None:
        capped = []
        for imgs in image_lists:
            picked = imgs
            if len(imgs) > per_subject_cap:
                rng.shuffle(imgs)
                picked = imgs[:per_subject_cap]
            capped.extend(picked)
        flat = capped
    else:
        flat = []
        for imgs in image_lists:
            flat.extend(imgs)
    if len(flat) <= cap:
        return sorted(flat)
    rng.shuffle(flat)
    return sorted(flat[:cap])


def build_manifest(seed=SEED):
    rng = random.Random(seed + 1)
    split = build_oasis_split(seed)

    manifest = {
        "seed": seed,
        "classes": CLASSES,
        "train": {c: [] for c in CLASSES},
        "val": {c: [] for c in CLASSES},
        "test": {c: [] for c in CLASSES},
        "train_subjects": {c: [] for c in CLASSES},
        "val_subjects": {c: [] for c in CLASSES},
        "test_subjects": {c: [] for c in CLASSES},
    }

    # ---- OASIS-based val / test / train ----
    folder_of = {cls: folder for folder, cls in OASIS_FOLDER_MAP.items()}

    def prefix(cls, imgs):
        folder = folder_of[cls]
        return [f"{folder}/{f}" for f in imgs]

    for cls in CLASSES:
        info = split[cls]

        def subject_lists(subjects):
            return [list(info["all_images"][s]) for s in subjects]

        val_imgs = subject_lists(info["val_subjects"])
        test_imgs = subject_lists(info["test_subjects"])
        train_imgs = subject_lists(info["train_subjects"])

        manifest["val"][cls] = prefix(cls, sample_slices(val_imgs, OASIS_VAL_CAP, rng, per_subject_cap=MAX_EVAL_SLICES_PER_SUBJECT))
        manifest["test"][cls] = prefix(cls, sample_slices(test_imgs, OASIS_TEST_CAP, rng, per_subject_cap=MAX_EVAL_SLICES_PER_SUBJECT))
        manifest["train"][cls] = prefix(cls, sample_slices(train_imgs, OASIS_TRAIN_CAP, rng, per_subject_cap=MAX_TRAIN_SLICES_PER_SUBJECT))

        manifest["val_subjects"][cls] = info["val_subjects"]
        manifest["test_subjects"][cls] = info["test_subjects"]
        manifest["train_subjects"][cls] = info["train_subjects"]

    # ---- KAGGLE train (all images) ----
    for cls in CLASSES:
        folder = os.path.join(KAGGLE_TRAIN, cls)
        imgs = list_images(folder)
        kaggle_paths = [f"KAGGLE/{cls}/{f}" for f in imgs]
        manifest["train"][cls] = manifest["train"][cls] + kaggle_paths

    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def resolve_path(rel):
    """Resolve a manifest-relative path to an absolute path."""
    if rel.startswith("KAGGLE"):
        return os.path.join(KAGGLE_TRAIN, rel.split("/", 1)[1])
    parts = rel.split("/")
    return os.path.join(OASIS_PATH, parts[0], "/".join(parts[1:]))


def summary(manifest):
    lines = []
    for split_name in ("train", "val", "test"):
        counts = {c: len(manifest[split_name][c]) for c in CLASSES}
        subj = {c: len(manifest[f"{split_name}_subjects"][c]) for c in CLASSES}
        lines.append(f"{split_name}: total={sum(counts.values())} classes={counts}")
        lines.append(f"   subjects={subj}")
    return "\n".join(lines)


if __name__ == "__main__":
    m = build_manifest()
    print(summary(m))
