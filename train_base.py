"""Fine-tune the four MRI base models on KAGGLE train + OASIS-train subjects.

Validation and test are the subject-disjoint OASIS held-out sets from
`splits.build_manifest`.  Each model starts from the existing KAGGLE-trained
checkpoint (offline, no downloads) and is fine-tuned with mixed precision.

Usage:
    python train_base.py --model resnet152 --epochs 8
    python train_base.py --model all --epochs 8
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from splits import CLASSES, build_manifest, resolve_path

BASE_PATH = Path(__file__).resolve().parent
MODELS_PATH = BASE_PATH / "models"
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4


def build_resnet152():
    model = models.resnet152(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, NUM_CLASSES))
    return model


def build_vgg16():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, NUM_CLASSES))
    return model


def build_efficientnet():
    return timm.create_model("efficientnet_b4", pretrained=False, num_classes=NUM_CLASSES)


def build_vit():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)


MODEL_BUILDERS = {
    "resnet152": (build_resnet152, 224),
    "vgg16": (build_vgg16, 224),
    "efficientnet_b4": (build_efficientnet, 380),
    "vit": (build_vit, 224),
}

MODEL_CONFIG = {
    "resnet152": {"batch_size": 16, "lr": 1e-4},
    "vgg16": {"batch_size": 8, "lr": 1e-4},
    "efficientnet_b4": {"batch_size": 8, "lr": 1e-4},
    "vit": {"batch_size": 8, "lr": 1e-4},
}


class ManifestDataset(Dataset):
    def __init__(self, manifest, split, transform):
        self.paths = []
        self.labels = []
        for idx, cls in enumerate(CLASSES):
            for rel in manifest[split][cls]:
                self.paths.append(resolve_path(rel))
                self.labels.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        im = Image.open(self.paths[i]).convert("RGB")
        return self.transform(im), self.labels[i]


def make_transforms(size, train=False):
    ops = [transforms.Resize((size, size)), transforms.Grayscale(num_output_channels=3)]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(12),
                transforms.RandomAffine(degrees=8, translate=(0.08, 0.08))]
    ops += [transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_labels.extend(y.numpy().tolist())
            correct += (pred == y.numpy()).sum()
            total += len(y)
    if total == 0:
        return 0.0, all_preds, all_labels
    return 100.0 * correct / total, all_preds, all_labels


def train_model(model_key, epochs, patience=3):
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    cfg = MODEL_CONFIG[model_key]
    size = MODEL_BUILDERS[model_key][1]

    print("=" * 70)
    print(f"FINE-TUNING {model_key} | size={size} | bs={cfg['batch_size']} | lr={cfg['lr']} | device={DEVICE}")
    print("=" * 70)

    manifest = build_manifest()

    train_ds = ManifestDataset(manifest, "train", make_transforms(size, train=True))
    val_ds = ManifestDataset(manifest, "val", make_transforms(size, train=False))
    test_ds = ManifestDataset(manifest, "test", make_transforms(size, train=False))
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise ValueError("Training, validation, and test datasets must not be empty")

    train_labels = np.asarray(train_ds.labels)
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    if len(counts) < NUM_CLASSES or np.any(counts[:NUM_CLASSES] == 0):
        raise ValueError("Training data must contain every class")
    inv_freq = 1.0 / counts.astype(np.float32)
    class_weights = torch.tensor(inv_freq / inv_freq.sum() * NUM_CLASSES, dtype=torch.float32).to(DEVICE)
    sample_weights = inv_freq[train_labels]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    print(f"train class counts={counts.tolist()}")

    model = MODEL_BUILDERS[model_key][0]().to(DEVICE)
    ckpt = os.path.join(MODELS_PATH, f"{model_key}_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded existing checkpoint: {ckpt}")
    else:
        print("No existing checkpoint -> starting from random init (not recommended)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val = -1.0
    best_state = None
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_correct = train_total = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)
            train_correct += (out.argmax(dim=1) == y).sum().item()
            train_total += len(y)
        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader)
        train_acc = 100.0 * train_correct / train_total
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": float(val_acc),
                        "train_loss": train_loss / train_total, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:02d}/{epochs} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}% | "
              f"loss={history[-1]['train_loss']:.4f} | lr={history[-1]['lr']:.2e} | {time.time()-t0:.0f}s")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
            print(f"   * new best val acc: {best_val:.2f}%")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("   early stop")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state")
    os.makedirs(MODELS_PATH, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_PATH, f"{model_key}_best.pth"))

    test_acc, test_preds, test_labels = evaluate(model, test_loader)
    val_acc, val_preds, val_labels = evaluate(model, val_loader)
    print(f"\n{model_key} -> val_acc={val_acc:.2f}% test_acc={test_acc:.2f}%")

    info = {
        "model_name": model_key,
        "num_classes": NUM_CLASSES,
        "img_size": size,
        "class_names": CLASSES,
        "seed": SEED,
        "epochs_trained": len(history),
        "best_validation_accuracy": float(best_val),
        "final_val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "history": history,
    }
    with open(os.path.join(MODELS_PATH, f"{model_key}_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Saved {model_key}_best.pth + {model_key}_info.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_BUILDERS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    keys = list(MODEL_BUILDERS) if args.model == "all" else [args.model]
    for key in keys:
        train_model(key, epochs=args.epochs)


if __name__ == "__main__":
    main()
