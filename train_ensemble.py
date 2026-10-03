"""Train the stacking ensemble meta-learner.

Meta-features = 16 softmax probabilities (4 classes x 4 models) computed on
the OASIS validation split.  The meta-learner is evaluated on the OASIS test
split (subject-disjoint from both training and validation).
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from splits import CLASSES, build_manifest
from train_base import MODEL_BUILDERS, ManifestDataset, make_transforms, DEVICE

MODELS_PATH = Path(__file__).resolve().parent / "models"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_KEYS = ["resnet152", "vgg16", "efficientnet_b4", "vit"]
DISPLAY_NAMES = ["ResNet-152", "VGG16", "EfficientNet-B4", "ViT"]


def load_base_models():
    models_dict = {}
    for key in MODEL_KEYS:
        size = MODEL_BUILDERS[key][1]
        model = MODEL_BUILDERS[key][0]().to(DEVICE)
        ckpt = os.path.join(MODELS_PATH, f"{key}_best.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models_dict[key] = (model, size)
    return models_dict


def collect(manifest, split, base_models, batch_size=32):
    """Return meta-features (N x 16) and labels for a split."""
    per_model = {}
    for key in MODEL_KEYS:
        model, size = base_models[key]
        ds = ManifestDataset(manifest, split, make_transforms(size, train=False))
        if len(ds) == 0:
            raise ValueError(f"No samples found for split: {split}")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE)
                with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(x)
                probs.append(torch.softmax(out, dim=1).cpu().numpy())
        per_model[key] = np.vstack(probs)
        print(f"  {key}: {per_model[key].shape}")

    X = np.hstack([per_model[k] for k in MODEL_KEYS])
    if X.ndim != 2 or X.shape[1] != 16 or not np.isfinite(X).all():
        raise ValueError(f"Invalid ensemble features for split: {split}")
    # labels are identical across models; reuse last loader's dataset
    ds = ManifestDataset(manifest, split, None)
    y = np.array(ds.labels)
    if len(y) != len(X):
        raise ValueError(f"Feature and label counts differ for split: {split}")
    return X, y


def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    manifest = build_manifest()
    base_models = load_base_models()
    print("Collecting validation meta-features...")
    val_X, val_y = collect(manifest, "val", base_models)
    print("Collecting test meta-features...")
    test_X, test_y = collect(manifest, "test", base_models)

    print("\nTraining meta-learner on OASIS validation features...")
    meta = LogisticRegression(max_iter=2000, random_state=SEED)
    meta.fit(val_X, val_y)

    val_pred = meta.predict(val_X)
    val_acc = accuracy_score(val_y, val_pred) * 100

    test_pred = meta.predict(test_X)
    test_acc = accuracy_score(test_y, test_pred) * 100

    print("=" * 70)
    print("ENSEMBLE RESULTS (OASIS validation/test)")
    print("=" * 70)
    print(f"Validation accuracy: {val_acc:.2f}%")
    print(f"Test accuracy      : {test_acc:.2f}%")
    print("\nClassification report (test):")
    print(classification_report(test_y, test_pred, target_names=CLASSES, digits=4, labels=list(range(4)), zero_division=0))
    print("\nConfusion matrix (test):")
    print(confusion_matrix(test_y, test_pred, labels=list(range(4))))

    # individual model test accuracies for the info file
    individual = {}
    for i, key in enumerate(MODEL_KEYS):
        acc = accuracy_score(test_y, test_X[:, i * 4:(i + 1) * 4].argmax(axis=1)) * 100
        individual[DISPLAY_NAMES[i]] = round(float(acc), 4)

    best_name = max(individual, key=individual.get)
    improvement = test_acc - individual[best_name]

    joblib.dump(meta, os.path.join(MODELS_PATH, "ensemble_meta_learner.pkl"))

    info = {
        "models": DISPLAY_NAMES,
        "input_sizes": {DISPLAY_NAMES[i]: MODEL_BUILDERS[k][1] for i, k in enumerate(MODEL_KEYS)},
        "num_classes": 4,
        "classes": CLASSES,
        "seed": SEED,
        "validation_samples": int(len(val_y)),
        "test_samples": int(len(test_y)),
        "validation_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "individual_test_accuracies": individual,
        "best_individual_model": best_name,
        "best_individual_accuracy": individual[best_name],
        "ensemble_improvement": float(improvement),
        "feature_count": int(test_X.shape[1]),
        "method": "Logistic Regression Stacking (OASIS validation/test)",
        "test_set": "OASIS (subject-disjoint)",
    }
    with open(os.path.join(MODELS_PATH, "ensemble_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("\nSaved ensemble_meta_learner.pkl + ensemble_info.json")


if __name__ == "__main__":
    main()
