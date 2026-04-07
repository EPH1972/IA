"""
utils.py – training helpers, metrics, and visualisation utilities
for the Heart Disease prediction project.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Train / Test loops
# ---------------------------------------------------------------------------

def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Evaluate on *dataloader*. Returns (mean_loss, accuracy_pct)."""
    model.eval()
    total_loss, correct = 0.0, 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    mean_loss = total_loss / len(dataloader)
    accuracy  = 100.0 * correct / size
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: list[float],
    test_losses: list[float],
    title: str = "Loss curves",
) -> None:
    """Plot training and validation loss curves side by side."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses,  label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_curve(
    accuracies: list[float],
    title: str = "Test accuracy",
) -> None:
    """Plot accuracy over epochs."""
    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, accuracies, color="green", label="Test accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str] | None = None,
) -> None:
    """Compute and display a confusion matrix."""
    if class_names is None:
        class_names = ["No disease", "Disease"]

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
