"""Train a true multimodal fusion model (MRI features + clinical scores).

Input  : 16 MRI probabilities (4-class softmax x 4 base models, averaged per
         subject) + 3 clinical features (mmse_norm, cdr_norm, nwbv_norm).
Output : 2 classes - DEMENTED (0) / NON-DEMENTED (1).

Pairing note: brainscore.csv contains no subject IDs, so exact per-subject
pairing with the MRI images is impossible. Each sample therefore uses a real
per-subject MRI feature vector plus a clinical triple sampled from the real
OASIS clinical records of the SAME diagnosis (deterministic per subject).
Splits are subject-disjoint (no subject appears in more than one split).

The single-image MRI probabilities and clinical scalers used here match the
runtime path in app.py, so the trained model works directly in the app.
"""

import json
import os
import random
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from train_base import MODEL_BUILDERS, make_transforms, DEVICE
from splits import build_manifest, clinical_data_path

BASE_PATH = Path(__file__).resolve().parent
DATASETS_PATH = BASE_PATH / "datasets"
OASIS_PATH = DATASETS_PATH / "OASIS DATASET" / "Data"
MODELS_PATH = BASE_PATH / "models"
RESULTS_PATH = BASE_PATH / "results"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SUBJECT_RE = re.compile(r"(OAS\d+_\d+)_")

OASIS_FOLDER_LABEL = {
    "Mild Dementia": 0,
    "Moderate Dementia": 0,
    "Very mild Dementia": 0,
    "Non Demented": 1,
}

MAX_SLICES_PER_SUBJECT = 12
FEATURE_CACHE = os.path.join(RESULTS_PATH, "fusion_mri_features.npz")


class MultimodalFusionNet(nn.Module):
    """MRI (16) + clinical (3) -> DEMENTED / NON-DEMENTED."""

    def __init__(self, n_mri=16, n_clinical=3, n_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_mri + n_clinical, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.network(x)


def subject_id(filename):
    m = SUBJECT_RE.match(filename)
    return m.group(1) if m else None


def list_images(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def collect_subject_mri_features(base_models, use_cache=True):
    """Per-subject 16-dim MRI features (mean softmax over sampled slices)."""
    if use_cache and os.path.exists(FEATURE_CACHE):
        try:
            d = np.load(FEATURE_CACHE, allow_pickle=False)
            required = {"subject_ids", "mri_features", "labels"}
            if required.issubset(d.files):
                subject_ids = d["subject_ids"].tolist()
                mri_features = d["mri_features"]
                labels = d["labels"]
                if (
                    len(subject_ids) == len(mri_features) == len(labels)
                    and len(set(subject_ids)) == len(subject_ids)
                    and mri_features.ndim == 2
                    and mri_features.shape[1] == 16
                    and np.isfinite(mri_features).all()
                    and set(np.unique(labels)).issubset({0, 1})
                ):
                    subject_ids = [str(subject) for subject in subject_ids]
                    print(f"Loaded MRI feature cache: {FEATURE_CACHE}")
                    return subject_ids, mri_features, labels
        except (OSError, ValueError, KeyError):
            pass

    print("Computing per-subject MRI features (no valid cache found)...")
    if not base_models:
        raise ValueError("At least one base model is required")
    model_list = [(k, m.to(DEVICE).eval()) for k, (m, size) in base_models.items()]
    sizes = {k: size for k, (_, size) in base_models.items()}

    subject_ids, mri_features, labels = [], [], []
    t0 = time.time()
    for folder, label in OASIS_FOLDER_LABEL.items():
        folder_path = os.path.join(OASIS_PATH, folder)
        images = list_images(folder_path)
        subjects = {}
        for img in images:
            sid = subject_id(img)
            if sid is not None:
                subjects.setdefault(sid, []).append(img)

        for sid, imgs in sorted(subjects.items()):
            rng = random.Random(SEED + len(subject_ids))
            picked = imgs
            if len(imgs) > MAX_SLICES_PER_SUBJECT:
                picked = rng.sample(imgs, MAX_SLICES_PER_SUBJECT)

            per_model = {}
            for key, model in model_list:
                size = sizes[key]
                transform = make_transforms(size, train=False)
                batch = torch.stack([transform(Image.open(os.path.join(folder_path, f)).convert("RGB"))
                                     for f in picked]).to(DEVICE)
                with torch.no_grad():
                    with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                        out = model(batch)
                    per_model[key] = torch.softmax(out, dim=1).mean(dim=0).cpu().numpy()

            feats = np.hstack([per_model[k] for k in base_models])
            subject_ids.append(sid)
            mri_features.append(feats)
            labels.append(label)
            if len(subject_ids) % 50 == 0:
                print(f"  {len(subject_ids)} subjects | {time.time()-t0:.0f}s")

    mri_features = np.asarray(mri_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if not subject_ids or not np.isfinite(mri_features).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("MRI feature collection produced invalid data")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    np.savez(FEATURE_CACHE, subject_ids=subject_ids, mri_features=mri_features, labels=labels)
    return subject_ids, mri_features, labels


def load_clinical_triples():
    """Per-diagnosis clinical triples + shared scalers (matches app.py)."""
    df = pd.read_csv(clinical_data_path())
    required_columns = {"Group", "MMSE", "CDR", "nWBV"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Clinical dataset is missing columns: {sorted(missing_columns)}")
    df["diagnosis"] = df["Group"].apply(
        lambda g: 0 if str(g).strip() in ("Demented", "Converted") else (1 if str(g).strip() == "Nondemented" else np.nan)
    )
    df = df.dropna(subset=["diagnosis"]).copy()
    for col in ("MMSE", "CDR", "nWBV"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if df[col].notna().sum() == 0:
            raise ValueError(f"Clinical feature {col} contains no numeric values")
        df[col] = df[col].fillna(df[col].median())

    cdr_scaler = StandardScaler()
    nwbv_scaler = StandardScaler()
    mmse_norm = df["MMSE"].values / 30.0
    cdr_norm = cdr_scaler.fit_transform(df[["CDR"]].values).ravel()
    nwbv_norm = nwbv_scaler.fit_transform(df[["nWBV"]].values).ravel()

    triples = {}
    for label in (0, 1):
        mask = df["diagnosis"].values == label
        if not np.any(mask):
            raise ValueError(f"Clinical dataset has no samples for class {label}")
        triples[label] = np.stack(
            [mmse_norm[mask], cdr_norm[mask], nwbv_norm[mask]], axis=1
        ).astype(np.float32)
    return triples, cdr_scaler, nwbv_scaler


def sample_clinical_triple(rng, triples, label):
    pool = triples[label]
    return pool[rng.randint(0, len(pool) - 1)]


def main():
    print("=" * 70)
    print("MULTIMODAL FUSION (MRI + CLINICAL)")
    print("=" * 70)

    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    base_models = {}
    for key in MODEL_BUILDERS:
        model = MODEL_BUILDERS[key][0]().to(DEVICE)
        ckpt = os.path.join(MODELS_PATH, f"{key}_best.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        base_models[key] = (model, MODEL_BUILDERS[key][1])
    print("Base models loaded.")

    subject_ids, mri_features, labels = collect_subject_mri_features(base_models)
    print(f"Subjects: {len(subject_ids)} | MRI features: {mri_features.shape}")

    triples, cdr_scaler, nwbv_scaler = load_clinical_triples()
    print(f"Clinical pools: demented={len(triples[0])} non_demented={len(triples[1])}")

    clinical_features = np.zeros((len(subject_ids), 3), dtype=np.float32)
    for i, sid in enumerate(subject_ids):
        rng = random.Random(SEED + int("".join(ch for ch in sid if ch.isdigit())) % (2**31))
        clinical_features[i] = sample_clinical_triple(rng, triples, int(labels[i]))

    X = np.hstack([mri_features, clinical_features]).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Fusion features contain non-finite values")
    y = labels.astype(np.int64)

    manifest = build_manifest()
    split_subjects = {
        split: {
            subject
            for cls in manifest["classes"]
            for subject in manifest[f"{split}_subjects"][cls]
            if subject is not None
        }
        for split in ("train", "val", "test")
    }
    split_ids = {"train": [], "val": [], "test": []}
    for index, subject in enumerate(subject_ids):
        if subject in split_subjects["test"]:
            split_ids["test"].append(index)
        elif subject in split_subjects["val"]:
            split_ids["val"].append(index)
        elif subject in split_subjects["train"]:
            split_ids["train"].append(index)
        else:
            raise ValueError(f"Subject {subject!r} is not present in the split manifest")
    train_idx = np.asarray(split_ids["train"], dtype=np.int64)
    val_idx = np.asarray(split_ids["val"], dtype=np.int64)
    test_idx = np.asarray(split_ids["test"], dtype=np.int64)
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("Fusion splits must contain train, validation, and test subjects")
    if np.unique(y[train_idx]).size != 2:
        raise ValueError("Fusion training split must contain both classes")

    print(f"Split (subject-disjoint): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"  train class counts: {np.bincount(y[train_idx])}")

    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))
    test_ds = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    train_counts = np.bincount(y[train_idx], minlength=2)
    if np.any(train_counts == 0):
        raise ValueError("Fusion training split must contain both classes")
    weights = torch.tensor(1.0 / train_counts, dtype=torch.float32)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    model = MultimodalFusionNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    def evaluate(loader):
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb.to(DEVICE))
                preds.extend(out.argmax(dim=1).cpu().tolist())
                truth.extend(yb.tolist())
        return accuracy_score(truth, preds) * 100, preds, truth

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, 121):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc, _, _ = evaluate(val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch % 10 == 0:
            print(f"  epoch {epoch:03d} | val_acc={val_acc:.2f}%")

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state")
    model.load_state_dict(best_state)
    test_acc, test_preds, test_truth = evaluate(test_loader)
    val_acc, val_preds, val_truth = evaluate(val_loader)
    print("=" * 70)
    print(f"FUSION RESULTS (subject-disjoint OASIS)")
    print("=" * 70)
    print(f"best val acc (epoch {best_epoch}): {best_val:.2f}%")
    print(f"test accuracy: {test_acc:.2f}%")
    print("\nClassification report (test):")
    print(classification_report(test_truth, test_preds, labels=[0, 1], target_names=["DEMENTED", "NON-DEMENTED"], digits=4, zero_division=0))

    torch.save(best_state, os.path.join(MODELS_PATH, "fusion_model.pth"))
    joblib.dump(cdr_scaler, os.path.join(MODELS_PATH, "fusion_cdr_scaler.pkl"))
    joblib.dump(nwbv_scaler, os.path.join(MODELS_PATH, "fusion_nwbv_scaler.pkl"))
    np.savez(
        os.path.join(RESULTS_PATH, "fusion_dataset.npz"),
        subject_ids=subject_ids, X=X, y=y,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )

    info = {
        "model": "MultimodalFusionNet",
        "input_features": int(X.shape[1]),
        "mri_features": 16,
        "clinical_features": ["mmse_norm", "cdr_norm", "nwbv_norm"],
        "num_classes": 2,
        "classes": {"0": "DEMENTED", "1": "NON-DEMENTED"},
        "seed": SEED,
        "subjects": len(subject_ids),
        "train_subjects": int(len(train_idx)),
        "val_subjects": int(len(val_idx)),
        "test_subjects": int(len(test_idx)),
        "best_validation_accuracy": float(best_val),
        "test_accuracy": float(test_acc),
        "pairing": (
            "clinical triples sampled per diagnosis from brainscore.csv "
            "(no subject IDs available for exact pairing); MRI features are "
            "per-subject real averages; splits are subject-disjoint"
        ),
    }
    with open(os.path.join(MODELS_PATH, "fusion_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("\nSaved fusion_model.pth + fusion scalers + fusion_info.json")


if __name__ == "__main__":
    main()
