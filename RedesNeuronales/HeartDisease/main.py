import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from classes import HeartDiseaseDataset, HeartDiseaseModel, CrossEntropyLoss, Adam
from utils import train_loop, test_loop, plot_loss_curves, plot_accuracy_curve, plot_confusion_matrix, count_parameters

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

CSV_PATH    = 'heart.csv'
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 60
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(df['target'].value_counts().to_string())
print()

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df['target'])

train_dataset = HeartDiseaseDataset(train_df)
test_dataset  = HeartDiseaseDataset(test_df, scaler=train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = HeartDiseaseModel(dropout=0.3)
print(model)
print(f"Trainable parameters: {count_parameters(model):,}")

# ---------------------------------------------------------------------------
# Optimizer and loss  (Adam instead of SGD)
# ---------------------------------------------------------------------------

loss_fn   = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train_losses, test_losses, accuracies = [], [], []

for epoch in range(1, EPOCHS + 1):
    tr_loss          = train_loop(train_loader, model, loss_fn, optimizer)
    te_loss, acc     = test_loop(test_loader,   model, loss_fn)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    accuracies.append(acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3d}/{EPOCHS}  "
              f"Train loss: {tr_loss:.4f}  "
              f"Test loss: {te_loss:.4f}  "
              f"Accuracy: {acc:.1f}%")

print("\nDone!")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

plot_loss_curves(train_losses, test_losses)
plot_accuracy_curve(accuracies)
plot_confusion_matrix(model, test_loader)

# ---------------------------------------------------------------------------
# Save model + scaler
# ---------------------------------------------------------------------------

torch.save({
    'model_state': model.state_dict(),
    'scaler':      train_dataset.scaler,
}, 'heart_model.pth')

print("Model saved to heart_model.pth")
