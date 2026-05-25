"""
SEC 3 — Red MLP Actor-Critic para observaciones vectoriales (modo TorchCraft)
Complementa la CNN de network.py, que procesa frames de imagen apilados.
"""
import torch
import torch.nn as nn


class MLPFeatureExtractor(nn.Module):
    """Dos capas densas con activación Tanh (más estable que ReLU para vectores normalizados)."""

    def __init__(self, obs_size: int, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.Tanh(),
            nn.Linear(256, feature_dim),
            nn.Tanh(),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticMLP(nn.Module):
    """
    MLP compartido con dos cabezas:
      - actor:  distribución sobre N_ACTIONS acciones
      - critic: estimación del valor del estado

    Misma interfaz que ActorCritic (CNN) para que PPOAgent sea transparente.
    """

    def __init__(self, obs_size: int, n_actions: int, feature_dim: int = 256):
        super().__init__()
        self.features = MLPFeatureExtractor(obs_size, feature_dim)
        self.actor    = nn.Linear(feature_dim, n_actions)
        self.critic   = nn.Linear(feature_dim, 1)
        self._init_weights()

    def _init_weights(self):
        gain = nn.init.calculate_gain("tanh")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        return self.actor(feats), self.critic(feats).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
