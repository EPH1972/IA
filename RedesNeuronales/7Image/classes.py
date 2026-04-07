import idx2numpy
import torch
from torch import nn
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """
    Custom Dataset that reads MNIST idx-ubyte files directly.
    Returns (image_tensor, label) where image is float32 in [0,1],
    shape (1, 28, 28) — channel-first, as expected by Conv2d.
    """

    def __init__(self, images_path: str, labels_path: str):
        images = idx2numpy.convert_from_file(images_path)  # (N, 28, 28)  uint8
        labels = idx2numpy.convert_from_file(labels_path)  # (N,)         uint8

        # Normalise to [0, 1] and add channel dim -> (N, 1, 28, 28)
        self.images = torch.tensor(images / 255.0, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MNISTModel(nn.Module):
    """
    Small CNN for MNIST (1x28x28 -> 10 classes).

    Architecture (following the notebook's CNN pattern):
        Conv2d(1,  32, 3)  -> (32, 26, 26)
        ReLU
        MaxPool2d(2)       -> (32, 13, 13)
        Conv2d(32, 64, 3)  -> (64, 11, 11)
        ReLU
        MaxPool2d(2)       -> (64,  5,  5)
        Flatten            -> 1600
        Linear(1600, 128)
        ReLU
        Linear(128, 10)
    """

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),    # (bs, 1, 28, 28) -> (bs, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # -> (bs, 32, 13, 13)
            nn.Conv2d(32, 64, kernel_size=3),   # -> (bs, 64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # -> (bs, 64,  5,  5)
            nn.Flatten(),                       # -> (bs, 1600)
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.seq(X)
