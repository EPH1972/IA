"""
SEC 1 — Calculador de Recompensa
Estimación de recompensa basada en diferencia de frames (heurística visual).
Sin acceso directo al estado interno del juego.
"""
import numpy as np


class RewardCalculator:
    """
    Recompensa visual: mide la actividad en pantalla y penaliza inactividad.
    Puede extenderse con señales de juego más ricas (score, unidades, etc.).
    """

    def __init__(self, survival_bonus: float = 0.001, noop_penalty: float = 0.002,
                 activity_weight: float = 0.1):
        self.survival_bonus = survival_bonus
        self.noop_penalty = noop_penalty
        self.activity_weight = activity_weight

        self._prev_frame: np.ndarray | None = None
        self._step = 0
        self._cumulative = 0.0

    def reset(self):
        self._prev_frame = None
        self._step = 0
        self._cumulative = 0.0

    def compute(self, obs: np.ndarray, action_id: int, done: bool) -> float:
        reward = self.survival_bonus

        if action_id == 0:
            reward -= self.noop_penalty

        current_frame = obs[-1]
        if self._prev_frame is not None:
            diff = float(np.mean(np.abs(current_frame - self._prev_frame)))
            reward += diff * self.activity_weight

        self._prev_frame = current_frame.copy()
        self._step += 1
        self._cumulative += reward
        return reward

    @property
    def cumulative(self) -> float:
        return self._cumulative
