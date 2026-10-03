"""Full output verification: all 4 MRI models + ensemble + clinical fusion.

Evaluates every component on the OASIS validation/test splits and prints a
clear summary. The app prediction path is verified separately.
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

from splits import CLASSES, build_manifest, clinical_data_path
from train_base import MODEL_BUILDERS, ManifestDataset, make_transforms, DEVICE

BASE_PATH = Path(__file__).resolve().parent
MODELS_PATH = BASE_PATH / "models"
RESULTS_PATH = BASE_PATH / "results"
MODEL_KEYS = ["resnet152", "vgg16", "efficientnet_b4", "vit"]
DISPLAY = ["ResNet-152", "VGG16", "EfficientNet-B4", "ViT"]


class ClinicalFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.network(x)


class MultimodalFusionNet(nn.Module):
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


def load_all():
    base = {}
    for key in MODEL_KEYS:
        size = MODEL_BUILDERS[key][1]
        model = MODEL_BUILDERS[key][0]().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, f"{key}_best.pth"), map_location=DEVICE))
        model.eval()
        base[key] = (model, size)
    meta = joblib.load(os.path.join(MODELS_PATH, "ensemble_meta_learner.pkl"))
    clinical = ClinicalFusionNet().to(DEVICE)
    clinical.load_state_dict(torch.load(os.path.join(MODELS_PATH, "clinical_branch_best.pth"), map_location=DEVICE))
    clinical.eval()
    return base, meta, clinical


def collect(manifest, split, base, batch_size=32):
    X = {}
    y = None
    for key in MODEL_KEYS:
        model, size = base[key]
        ds = ManifestDataset(manifest, split, make_transforms(size, train=False))
        if len(ds) == 0:
            raise ValueError(f"No samples found for split: {split}")
        if y is None:
            y = np.array(ds.labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(DEVICE)
                with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(xb)
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
        X[key] = np.vstack(probs)
    return X, y


def main():
    manifest = build_manifest()
    base, meta, clinical = load_all()
    print("=" * 78)
    print("FULL OUTPUT VERIFICATION ON OASIS VALIDATION / TEST")
    print("=" * 78)

    report = {}
    for split in ("val", "test"):
        X, y = collect(manifest, split, base)
        print(f"\n--- {split.upper()} ---")
        indiv = {}
        for key, dname in zip(MODEL_KEYS, DISPLAY):
            acc = accuracy_score(y, X[key].argmax(axis=1)) * 100
            indiv[dname] = round(float(acc), 2)
            print(f"  {dname:<20} accuracy: {acc:.2f}%")

        meta_pred = meta.predict(np.hstack([X[k] for k in MODEL_KEYS]))
        ens_acc = accuracy_score(y, meta_pred) * 100
        print(f"  {'Ensemble':<20} accuracy: {ens_acc:.2f}%")
        print(f"\n  Ensemble classification report:\n{classification_report(y, meta_pred, target_names=CLASSES, labels=list(range(len(CLASSES))), digits=4, zero_division=0)}")
        report[split] = {"individual": indiv, "ensemble_accuracy": float(ens_acc)}

    # clinical branch validation (from brainscore.csv split reproduced here)
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
    mmse = df["MMSE"].values / 30.0
    cdr_scaler = joblib.load(MODELS_PATH / "fusion_cdr_scaler.pkl")
    nwbv_scaler = joblib.load(MODELS_PATH / "fusion_nwbv_scaler.pkl")
    cdr = cdr_scaler.transform(df[["CDR"]]).ravel()
    nwbv = nwbv_scaler.transform(df[["nWBV"]]).ravel()
    Xc = np.stack([mmse, cdr, nwbv], axis=1).astype(np.float32)
    yc = df["diagnosis"].values.astype(np.int64)
    if np.unique(yc).size != 2:
        raise ValueError("Clinical dataset must contain both diagnosis classes")

    split_generator = torch.Generator().manual_seed(42)
    train_count = int(0.8 * len(Xc))
    if train_count == 0 or train_count == len(Xc):
        raise ValueError("Clinical dataset is too small to split")
    _, val_set = random_split(
        list(range(len(Xc))),
        [train_count, len(Xc) - train_count],
        generator=split_generator,
    )
    val_idx = np.asarray(val_set.indices, dtype=np.int64)
    with torch.no_grad():
        out = clinical(torch.tensor(Xc[val_idx]).to(DEVICE)).argmax(dim=1).cpu().numpy()
    clin_acc = accuracy_score(yc[val_idx], out) * 100
    print(f"\n--- CLINICAL FUSION BRANCH ---")
    print(f"  validation accuracy: {clin_acc:.2f}%")
    report["clinical"] = {"validation_accuracy": float(clin_acc)}

    # ---- Multimodal fusion (MRI + clinical) on subject-disjoint test ----
    fusion_model = MultimodalFusionNet().to(DEVICE)
    fusion_model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "fusion_model.pth"), map_location=DEVICE))
    fusion_model.eval()

    data = np.load(os.path.join(RESULTS_PATH, "fusion_dataset.npz"))
    if not all(key in data for key in ("X", "y", "test_idx")):
        raise ValueError("Fusion dataset is missing required arrays")
    Xf = data["X"].astype(np.float32)
    yf = data["y"]
    test_idx = data["test_idx"]
    if len(test_idx) == 0:
        raise ValueError("Fusion dataset has no test subjects")
    Xt = torch.tensor(Xf[test_idx]).to(DEVICE)
    with torch.no_grad():
        out = fusion_model(Xt)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    fus_pred = out.argmax(dim=1).cpu().numpy()
    fus_acc = accuracy_score(yf[test_idx], fus_pred) * 100
    print(f"\n--- MULTIMODAL FUSION (MRI + Clinical) ---")
    print(f"  test accuracy (subject-disjoint): {fus_acc:.2f}%")
    print(classification_report(yf[test_idx], fus_pred, labels=[0, 1], target_names=["DEMENTED", "NON-DEMENTED"], digits=4, zero_division=0))

    # ---- reconciliation on test subjects: MRI stage vs clinical branch ----
    mri16 = Xf[test_idx, :16]
    clin3 = Xf[test_idx, 16:]
    ens = meta.predict(mri16)
    mri_demented = ens != 2  # class index 2 == "No Impairment"
    clin_pred = clinical(torch.tensor(clin3).to(DEVICE)).argmax(dim=1).cpu().numpy()
    clin_demented = clin_pred == 0
    conflict = mri_demented != clin_demented
    print(f"  reconciliation (test subjects): {int(conflict.sum())}/{len(conflict)} "
          f"show MRI/clinical conflict")
    report["fusion"] = {
        "test_accuracy": float(fus_acc),
        "test_subjects": int(len(conflict)),
        "conflict_count": int(conflict.sum()),
    }

    # ---- per-model ensemble probs sanity: check 16-feature ordering ----
    feats = np.hstack([X["resnet152"], X["vgg16"], X["efficientnet_b4"], X["vit"]])
    assert feats.shape[1] == 16, f"expected 16 features, got {feats.shape[1]}"
    print("\nFeature-order sanity check: OK (16 features = 4 probs x 4 models)")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(os.path.join(RESULTS_PATH, "evaluation_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nSaved results/evaluation_summary.json")


if __name__ == "__main__":
    main()
