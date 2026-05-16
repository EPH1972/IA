"""
SEC 1 — Agente PPO (Proximal Policy Optimization)
Selecciona acciones, almacena transiciones y actualiza la política.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sc1_rl.environment.action_space import N_ACTIONS
from sc1_rl.model.memory import RolloutBuffer
from sc1_rl.model.network import ActorCritic


class PPOAgent:
    """
    Agente PPO con política Actor-Critic CNN.

    Flujo de uso:
        action, log_prob, value = agent.select_action(obs)
        agent.store(obs, action, log_prob, reward, value, done)
        ...   (rollout_steps veces)
        metrics = agent.update(last_obs)
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCritic(
            in_channels=obs_shape[0],
            n_actions=N_ACTIONS,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, obs_shape, self.device)

    # ── Inferencia ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Devuelve (action_id, log_prob, value) muestreados de la política."""
        obs_t = torch.tensor(obs[None], dtype=torch.float32).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs[None], dtype=torch.float32).to(self.device)
        _, value = self.network(obs_t)
        return value.item()

    # ── Almacenamiento ────────────────────────────────────────────────────────

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.buffer.add(obs, action, log_prob, reward, value, done)

    # ── Actualización PPO ─────────────────────────────────────────────────────

    def update(self, last_obs: np.ndarray) -> dict[str, float]:
        """Ejecuta la actualización PPO y devuelve métricas."""
        last_value = self.get_value(last_obs)

        totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        n_updates = 0

        for _ in range(self.epochs):
            for obs_b, act_b, logp_b, adv_b, ret_b in self.buffer.get_batches(
                last_value, self.gamma, self.gae_lambda, self.batch_size
            ):
                _, new_logp, entropy, new_value = self.network.get_action_and_value(
                    obs_b, act_b
                )

                ratio = torch.exp(new_logp - logp_b)
                pg1 = -adv_b * ratio
                pg2 = -adv_b * torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = nn.functional.mse_loss(new_value, ret_b)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.mean().item()
                totals["total_loss"] += loss.item()
                n_updates += 1

        self.buffer.reset()
        return {k: v / n_updates for k, v in totals.items()}

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
