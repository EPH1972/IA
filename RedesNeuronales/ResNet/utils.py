"""
utils.py – Training helpers, metrics and visualisation for the ResNet project.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from classes import CIFAR10_CLASSES


# ---------------------------------------------------------------------------
# Train / Test loops
# ---------------------------------------------------------------------------

def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
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
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on *dataloader*. Returns (mean_loss, accuracy_pct)."""
    model.eval()
    total_loss, correct = 0.0, 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred        = model(X)
            total_loss += loss_fn(pred, y).item()
            correct    += (pred.argmax(1) == y).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / size


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: list[float],
    test_losses:  list[float],
    title: str = "Loss curves",
) -> None:
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
    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, accuracies, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> None:
    if class_names is None:
        class_names = CIFAR10_CLASSES

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.show()
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def show_sample_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n: int = 10,
) -> None:
    """Show *n* sample images with predicted vs true label."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])

    model.eval()
    X_batch, y_batch = next(iter(dataloader))
    with torch.no_grad():
        preds = model(X_batch.to(device)).argmax(1).cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    for i, ax in enumerate(axes.flat):
        img = X_batch[i].permute(1, 2, 0).numpy()
        img = img * std + mean                    # unnormalise
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        pred_name  = CIFAR10_CLASSES[preds[i]]
        true_name  = CIFAR10_CLASSES[y_batch[i]]
        color      = 'green' if preds[i] == y_batch[i] else 'red'
        ax.set_title(f"P:{pred_name}\nT:{true_name}", color=color, fontsize=8)
        ax.axis('off')

    plt.suptitle("Sample predictions (green=correct, red=wrong)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
