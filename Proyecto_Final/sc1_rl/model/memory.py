"""
SEC 1 — Buffer de Rollout (PPO on-policy)
Almacena transiciones de un episodio y calcula ventajas GAE.
"""
import numpy as np
import torch


class RolloutBuffer:
    """Buffer fijo para un rollout completo de tamaño `size`."""

    def __init__(self, size: int, obs_shape: tuple, device: torch.device):
        self.size = size
        self.device = device

        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.size

    def compute_returns(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE — Generalized Advantage Estimation."""
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            # dones[t] == 1 → el episodio terminó en t, next_val no aplica
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + self.values
        return advantages, returns

    def get_batches(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        batch_size: int,
    ):
        """Genera mini-batches aleatorios para la actualización PPO."""
        advantages, returns = self.compute_returns(last_value, gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                torch.tensor(self.obs[idx]).to(self.device),
                torch.tensor(self.actions[idx]).to(self.device),
                torch.tensor(self.log_probs[idx]).to(self.device),
                torch.tensor(advantages[idx]).to(self.device),
                torch.tensor(returns[idx]).to(self.device),
            )

    def reset(self):
        self._ptr = 0
