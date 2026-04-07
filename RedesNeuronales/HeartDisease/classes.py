import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Custom loss
# ---------------------------------------------------------------------------

class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss implemented from scratch.

    For a batch of logits z ∈ R^(N×C) and integer targets y ∈ {0,…,C-1}:

        log_softmax(z)_i = z_i - log( Σ_j exp(z_j) )
        loss = -mean_n( log_softmax(z_n)[y_n] )

    log-sum-exp is computed in a numerically stable way by subtracting
    the row maximum before exponentiating.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable log-softmax
        shift      = logits.max(dim=1, keepdim=True).values          # (N,1)
        log_probs  = logits - shift - (logits - shift).exp().sum(dim=1, keepdim=True).log()
        # Pick the log-prob of the correct class for each sample
        nll        = -log_probs[torch.arange(len(targets)), targets]  # (N,)
        return nll.mean()


# ---------------------------------------------------------------------------
# Custom Adam optimizer
# ---------------------------------------------------------------------------

class Adam(torch.optim.Optimizer):
    """
    Adam optimiser implemented from scratch.

    Update rule (Kingma & Ba, 2015):
        m_t = β1·m_{t-1} + (1-β1)·g_t          (1st moment)
        v_t = β2·v_{t-1} + (1-β2)·g_t²          (2nd moment)
        m̂_t = m_t / (1 - β1^t)                  (bias-corrected)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

    Parameters
    ----------
    params       : iterable of parameters to optimise
    lr           : learning rate (α), default 1e-3
    betas        : (β1, β2) momentum coefficients, default (0.9, 0.999)
    eps          : numerical stability constant ε, default 1e-8
    weight_decay : L2 penalty applied to gradient before moment update
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

                # Optional L2 regularisation: add wd·θ to the gradient
                if wd != 0.0:
                    grad.add_(p, alpha=wd)

                state = self.state[p]

                # Initialise state on first step
                if len(state) == 0:
                    state['step'] = 0
                    state['m']    = torch.zeros_like(p)   # 1st moment
                    state['v']    = torch.zeros_like(p)   # 2nd moment

                state['step'] += 1
                t    = state['step']
                m, v = state['m'], state['v']

                # Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-corrected estimates
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                # Parameter update
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
]
TARGET_COL = 'target'


class HeartDiseaseDataset(Dataset):
    """
    Dataset for the Heart Disease CSV.

    Parameters
    ----------
    df        : pandas DataFrame already split (train or test slice).
    scaler    : fitted StandardScaler.  Pass None to fit a new one (train set).
    """

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler | None = None):
        X = df[FEATURE_COLS].values.astype('float32')
        y = df[TARGET_COL].values.astype('int64')

        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HeartDiseaseModel(nn.Module):
    """
    Fully-connected classifier for the Heart Disease dataset.

    Architecture:
        Linear(13, 64)  -> BatchNorm -> ReLU -> Dropout(0.3)
        Linear(64, 32)  -> BatchNorm -> ReLU -> Dropout(0.3)
        Linear(32, 16)  -> ReLU
        Linear(16,  2)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(13, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 2),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.seq(X)
