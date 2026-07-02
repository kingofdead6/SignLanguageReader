"""
Step 2 — Train a small MLP on the extracted keypoints.

Because the input is 63 normalized floats (not pixels), this trains on CPU
in under a minute and typically reaches 97-99% validation accuracy.

Includes:
  - Mirror augmentation (handles left AND right hands with one model)
  - Gaussian noise augmentation (robustness to landmark jitter)
  - Label smoothing + dropout (better calibrated confidences at inference)

Output:
  model/model.pt       trained weights
  model/classes.json   label mapping (copied from data/)

Usage:
  python train.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils import FEATURES, mirror_landmarks

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GestureMLP(nn.Module):
    """63 -> 256 -> 128 -> n_classes. ~50k params. Trains in seconds."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def augment(X: np.ndarray, y: np.ndarray):
    """Mirror flip + gaussian jitter. Triples the dataset."""
    X_mirror = np.stack([mirror_landmarks(x) for x in X])
    X_noise = X + np.random.normal(0, 0.015, X.shape).astype(np.float32)
    return (
        np.concatenate([X, X_mirror, X_noise]),
        np.concatenate([y, y, y]),
    )


def main():
    data = Path("data")
    X = np.load(data / "X.npy")
    y = np.load(data / "y.npy")
    with open(data / "classes.json") as f:
        classes = json.load(f)
    n_classes = len(classes)
    print(f"Loaded {len(X)} samples, {n_classes} classes. Device: {DEVICE}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )

    # Augment training set only — never the validation set
    X_train, y_train = augment(X_train, y_train)
    print(f"Train: {len(X_train)} (after augmentation) | Val: {len(X_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    model = GestureMLP(n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    out = Path("model")
    out.mkdir(exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1) == yb).sum().item()
        acc = correct / len(val_ds)

        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "model.pt")
            marker = "  <- saved"
        print(f"Epoch {epoch:02d} | loss {total_loss/len(train_ds):.4f} "
              f"| val acc {acc:.4f}{marker}")

    shutil.copy(data / "classes.json", out / "classes.json")
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("Next: python realtime.py   (or python server.py for the web UI)")


if __name__ == "__main__":
    main()
