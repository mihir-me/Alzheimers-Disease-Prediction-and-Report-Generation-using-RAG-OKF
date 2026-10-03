"""Retrain the clinical fusion branch (ClinicalFusionNet).

Inputs : mmse_norm, cdr_norm, nwbv_norm  (from brainscore.csv)
Output : DEMENTED (0) / NON-DEMENTED (1)
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split

from splits import clinical_data_path

BASE_PATH = Path(__file__).resolve().parent
MODELS_PATH = BASE_PATH / "models"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClinicalFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.network(x)


def create_diagnosis_label(group):
    group = str(group).strip()
    if group in ("Demented", "Converted"):
        return 0
    if group == "Nondemented":
        return 1
    return np.nan


def main():
    df = pd.read_csv(clinical_data_path())
    required_columns = {"Group", "MMSE", "CDR", "nWBV"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Clinical dataset is missing columns: {sorted(missing_columns)}")
    df["diagnosis"] = df["Group"].apply(create_diagnosis_label)
    df = df.dropna(subset=["diagnosis"]).copy()
    if df.empty:
        raise ValueError("Clinical dataset contains no recognized diagnosis labels")

    for col in ("MMSE", "CDR", "nWBV"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if df[col].notna().sum() == 0:
            raise ValueError(f"Clinical feature {col} contains no numeric values")
        df[col] = df[col].fillna(df[col].median())

    mmse = df["MMSE"].values / 30.0
    cdr = StandardScaler().fit_transform(df[["CDR"]].values).ravel()
    nwbv = StandardScaler().fit_transform(df[["nWBV"]].values).ravel()

    X = np.stack([mmse, cdr, nwbv], axis=1).astype(np.float32)
    y = df["diagnosis"].values.astype(np.int64)
    if np.unique(y).size != 2:
        raise ValueError("Clinical dataset must contain both diagnosis classes")

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_count = int(0.8 * len(dataset))
    if train_count == 0 or train_count == len(dataset):
        raise ValueError("Clinical dataset is too small to split")
    train_set, val_set = random_split(
        dataset,
        [train_count, len(dataset) - train_count],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = ClinicalFusionNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
    for epoch in range(80):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_truth = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                val_preds.extend(model(xb).argmax(dim=1).cpu().tolist())
                val_truth.extend(yb.tolist())
        val_acc = accuracy_score(val_truth, val_preds) * 100
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state")
    os.makedirs(MODELS_PATH, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_PATH, "clinical_branch_best.pth"))

    print("=" * 60)
    print("CLINICAL FUSION BRANCH")
    print("=" * 60)
    print(f"train_samples={len(train_set)} val_samples={len(val_set)}")
    print(f"best validation accuracy: {best_val:.2f}%")

    info = {
        "model": "ClinicalFusionNet",
        "features": ["MMSE", "CDR", "nWBV"],
        "input_features": 3,
        "classes": {"0": "Demented/Converted", "1": "Nondemented"},
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "best_validation_accuracy": float(best_val),
        "seed": SEED,
    }
    with open(os.path.join(MODELS_PATH, "clinical_branch_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("Saved clinical_branch_best.pth + clinical_branch_info.json")


if __name__ == "__main__":
    main()
