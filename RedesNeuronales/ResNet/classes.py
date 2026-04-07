"""
classes.py – Custom loss, custom Adam, and ResNet-18 (CIFAR-10 adapted)
built entirely from scratch using only PyTorch primitives (nn.Conv2d,
nn.BatchNorm2d, nn.Linear) but NO torchvision models, no built-in
optimizers, and no built-in loss functions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# ===========================================================================
# Custom loss
# ===========================================================================

class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss from scratch.

        log_softmax(z)_i = z_i - max(z) - log( Σ_j exp(z_j - max(z)) )
        loss = -mean_n( log_softmax(z_n)[y_n] )

    Subtracting the row maximum keeps exp() from overflowing (numerically
    stable log-sum-exp trick).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        shift     = logits.max(dim=1, keepdim=True).values
        log_probs = logits - shift - (logits - shift).exp().sum(dim=1, keepdim=True).log()
        nll       = -log_probs[torch.arange(len(targets)), targets]
        return nll.mean()


# ===========================================================================
# Custom Adam optimiser
# ===========================================================================

class Adam(torch.optim.Optimizer):
    """
    Adam (Kingma & Ba, 2015) implemented from scratch.

    Update rule:
        m_t = β1·m_{t-1} + (1-β1)·g_t        (biased 1st moment)
        v_t = β2·v_{t-1} + (1-β2)·g_t²        (biased 2nd moment)
        m̂   = m_t / (1 - β1^t)                (corrected 1st moment)
        v̂   = v_t / (1 - β2^t)                (corrected 2nd moment)
        θ_t = θ_{t-1} - α · m̂ / (√v̂ + ε)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group['lr']
            beta1, beta2 = group['betas']
            eps          = group['eps']
            wd           = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clone()
                if wd != 0.0:
                    grad.add_(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m']    = torch.zeros_like(p)
                    state['v']    = torch.zeros_like(p)

                state['step'] += 1
                t    = state['step']
                m, v = state['m'], state['v']

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


# ===========================================================================
# ResNet-18 (CIFAR-10 adapted) – from scratch
# ===========================================================================

class BasicBlock(nn.Module):
    """
    ResNet basic residual block (He et al., 2016).

        x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+skip) → ReLU

    When in_channels ≠ out_channels the skip connection uses a 1×1 conv
    (projection shortcut) to match dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut – only used when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual addition
        return self.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 inputs, 10 classes).

    CIFAR adaptations vs. original ImageNet ResNet-18:
      - First conv: 3×3 / stride=1  (instead of 7×7 / stride=2)
      - No MaxPool after the first conv
      - GlobalAvgPool at the end collapses the 4×4 feature map

    Architecture:
        stem  : Conv(3→64, 3×3) → BN → ReLU
        layer1: 2× BasicBlock(64→64,  stride=1)
        layer2: 2× BasicBlock(64→128, stride=2)
        layer3: 2× BasicBlock(128→256, stride=2)
        layer4: 2× BasicBlock(256→512, stride=2)
        pool  : AdaptiveAvgPool2d(1×1) → Flatten
        fc    : Linear(512 → num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Stem (CIFAR: 3×3 instead of 7×7, no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(512, num_classes)

        # Weight initialisation (He et al.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ===========================================================================
# CIFAR-10 dataset wrapper
# ===========================================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


class CIFAR10Dataset(Dataset):
    """
    Wraps torchvision's CIFAR-10 raw data with manual normalisation and
    optional random horizontal flip + random crop for training augmentation.
    No torchvision.transforms are used – all preprocessing is done with
    plain NumPy / PyTorch operations.
    """

    def __init__(self, root: str, train: bool = True, augment: bool = False):
        import torchvision.datasets as dsets
        raw = dsets.CIFAR10(root=root, train=train, download=True)

        # (N, 32, 32, 3) uint8  →  (N, 3, 32, 32) float32 in [0, 1]
        imgs   = np.array(raw.data, dtype=np.float32) / 255.0
        imgs   = imgs.transpose(0, 3, 1, 2)           # channel-first

        mean   = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std    = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
        imgs   = (imgs - mean) / std

        self.X       = torch.tensor(imgs,            dtype=torch.float32)
        self.y       = torch.tensor(raw.targets,     dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        img, label = self.X[idx].clone(), self.y[idx]

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(dims=[2])

            # Random crop: pad 4 pixels then crop back to 32×32
            pad   = 4
            _, h, w = img.shape
            padded  = torch.zeros(3, h + 2 * pad, w + 2 * pad)
            padded[:, pad:pad + h, pad:pad + w] = img
            top  = torch.randint(0, 2 * pad, (1,)).item()
            left = torch.randint(0, 2 * pad, (1,)).item()
            img  = padded[:, top:top + h, left:left + w]

        return img, label


# ===========================================================================
# Helper: load a single image for inference (used by app.py)
# ===========================================================================

def preprocess_image(path: str) -> torch.Tensor:
    """
    Load an image from *path*, resize to 32×32, normalise with CIFAR-10
    statistics, and return a (1, 3, 32, 32) tensor ready for the model.
    """
    img  = Image.open(path).convert('RGB').resize((32, 32), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0        # (32,32,3)
    arr  = arr.transpose(2, 0, 1)                          # (3,32,32)

    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(3, 1, 1)
    arr  = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,3,32,32)
