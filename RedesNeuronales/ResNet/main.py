import torch
from torch.utils.data import DataLoader

from classes import CIFAR10Dataset, ResNet18, CrossEntropyLoss, Adam
from utils import (
    train_loop, test_loop,
    plot_loss_curves, plot_accuracy_curve,
    plot_confusion_matrix, show_sample_predictions,
    count_parameters,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DATA_ROOT   = './data'
BATCH_SIZE  = 128
LR          = 1e-3
EPOCHS      = 30
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_dataset = CIFAR10Dataset(DATA_ROOT, train=True,  augment=True)
test_dataset  = CIFAR10Dataset(DATA_ROOT, train=False, augment=False)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = ResNet18(num_classes=10).to(device)
print(model)
print(f"Trainable parameters: {count_parameters(model):,}")

# ---------------------------------------------------------------------------
# Loss and optimiser (both custom)
# ---------------------------------------------------------------------------

loss_fn   = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train_losses, test_losses, accuracies = [], [], []

for epoch in range(1, EPOCHS + 1):
    tr_loss        = train_loop(train_loader, model, loss_fn, optimizer, device)
    te_loss, acc   = test_loop(test_loader,   model, loss_fn,            device)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    accuracies.append(acc)

    print(f"Epoch {epoch:>3d}/{EPOCHS}  "
          f"Train loss: {tr_loss:.4f}  "
          f"Test loss: {te_loss:.4f}  "
          f"Accuracy: {acc:.1f}%")

print(f"\nBest accuracy: {max(accuracies):.1f}%  (epoch {accuracies.index(max(accuracies))+1})")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

plot_loss_curves(train_losses, test_losses)
plot_accuracy_curve(accuracies)
plot_confusion_matrix(model, test_loader, device)
show_sample_predictions(model, test_loader, device)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

torch.save({'model_state': model.state_dict()}, 'resnet_model.pth')
print("Model saved to resnet_model.pth")
