"""
SEC 1 — Entorno StarCraft 1 (Gymnasium)
Orquesta el flujo observación → acción → recompensa.
Delega toda comunicación con la VM al VMController (SEC 2).
"""
import time
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from sc1_rl.environment.action_space import N_ACTIONS, decode_action
from sc1_rl.environment.reward import RewardCalculator
from sc1_rl.environment.state_processor import StateProcessor
from sc1_rl.logger.action_logger import ActionLogger


class SC1Env(gym.Env):
    """
    Entorno Gymnasium para StarCraft 1.

    Parámetros
    ----------
    vm_controller : VMController
        Instancia de SEC 2 que gestiona la captura de pantalla y el input.
    action_logger : ActionLogger
        Logger de SEC 1 que registra cada acción enviada a la VM.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        vm_controller,
        action_logger: ActionLogger,
        obs_width: int = 128,
        obs_height: int = 128,
        frame_stack: int = 4,
        action_delay: float = 0.1,
        max_steps: int = 10_000,
    ):
        super().__init__()

        self.vm = vm_controller
        self.log = action_logger
        self.action_delay = action_delay
        self.max_steps = max_steps

        self.processor = StateProcessor(obs_width, obs_height, frame_stack)
        self.reward_calc = RewardCalculator()

        obs_shape = self.processor.observation_shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self._step_count = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.reward_calc.reset()
        self._step_count = 0
        self.log.log_episode_start()

        obs = self.processor.reset()
        screenshot = self.vm.capture_screen()
        if screenshot is not None:
            obs = self.processor.process(screenshot)

        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        decoded = decode_action(action)

        # ── LOG: todo lo que el modelo envía a la VM ─────────────────────────
        self.log.log_action(
            action_id=action,
            action_name=decoded.description,
            details={
                "type": decoded.action_type.name,
                "x": round(decoded.x, 4),
                "y": round(decoded.y, 4),
                "key": decoded.key,
                "group": decoded.group,
            },
        )

        # ── Ejecutar acción en la VM (SEC 2) ─────────────────────────────────
        self.vm.execute_action(decoded)
        time.sleep(self.action_delay)

        # ── Capturar nueva observación ────────────────────────────────────────
        screenshot = self.vm.capture_screen()
        if screenshot is not None:
            obs = self.processor.process(screenshot)
        else:
            obs = self.processor.get_stacked()

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = not self.vm.is_game_running()

        reward = self.reward_calc.compute(obs, action, terminated or truncated)
        self.log.log_reward(reward, self.reward_calc.cumulative)

        if terminated or truncated:
            self.log.log_episode_end(self.reward_calc.cumulative, self._step_count)

        return obs, reward, terminated, truncated, {"step": self._step_count}

    def close(self):
        self.log.close()
