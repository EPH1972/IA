import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from classes import MNISTDataset, MNISTModel

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

batch_size    = 100
learning_rate = 0.003
epochs        = 10

# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

train_dataset = MNISTDataset('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
test_dataset  = MNISTDataset('t10k-images.idx3-ubyte',  't10k-labels.idx1-ubyte')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = MNISTModel()
print(model)

# ---------------------------------------------------------------------------
# Optimizer and loss
# ---------------------------------------------------------------------------

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# Train loop and test loop
# ---------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print(f"  loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred       = model(X)
            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct   /= size
    print(f"  Accuracy: {(100 * correct):>0.1f}%  Avg loss: {test_loss:>8f}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t + 1}\n{'-' * 30}")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader,  model, loss_fn)
    print("Done!")

    # Visualize 10 sample predictions from the test set
    model.eval()
    images, labels = next(iter(test_dataloader))
    with torch.no_grad():
        preds = model(images).argmax(1)

    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i, 0].numpy(), cmap='gray')
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}",
                  color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle("Sample Test Predictions (green=correct, red=wrong)")
    plt.tight_layout()
    plt.show()
