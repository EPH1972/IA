"""
SEC 1 — Red Neuronal (Actor-Critic con CNN)
Extrae características espaciales de frames apilados y produce política + valor.
"""
import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """CNN estilo DeepMind para procesar frames de juego en escala de grises."""

    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, feature_dim),
            nn.ReLU(),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ActorCritic(nn.Module):
    """
    CNN compartida con dos cabezas:
      - actor:  distribución de política sobre N_ACTIONS acciones
      - critic: estimación del valor del estado
    """

    def __init__(self, in_channels: int, n_actions: int, feature_dim: int = 512):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, feature_dim)
        self.actor = nn.Linear(feature_dim, n_actions)
        self.critic = nn.Linear(feature_dim, 1)
        self._init_weights()

    def _init_weights(self):
        gain = nn.init.calculate_gain("relu")
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
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
