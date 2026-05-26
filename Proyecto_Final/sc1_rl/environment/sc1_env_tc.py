"""
SEC 3 — Entorno StarCraft 1 basado en TorchCraft (Gymnasium)
Reemplaza la pipeline screenshot+OCR por estado estructurado BWAPI via ZMQ.

Diferencias respecto a SC1Env (modo pixel):
  · Observación: vector float32 de 189 features (no imagen)
  · Acciones:    196 macro-comandos BWAPI (no teclado/ratón)
  · Rewards:     calculados sobre estado real del juego (recursos exactos, HP)
"""
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from sc1_rl.torchcraft.action_space import N_ACTIONS_TC, decode_tc_action
from sc1_rl.torchcraft.command_executor import CommandExecutor
from sc1_rl.torchcraft.reward import TCRewardCalculator
from sc1_rl.torchcraft.state_encoder import OBS_SIZE_TC, StateEncoder
from sc1_rl.logger.action_logger import ActionLogger


class SC1EnvTC(gym.Env):
    """
    Entorno Gymnasium para StarCraft 1 usando TorchCraft.

    Parámetros
    ----------
    tc_client : TorchCraftClient
        Instancia de SEC 3 conectada al servidor BWAPI de la VM.
    action_logger : ActionLogger
        Logger de acciones (mismo que en SC1Env).
    max_steps : int
        Pasos máximos por episodio.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        tc_client,
        action_logger: ActionLogger,
        max_steps: int = 50_000,
    ):
        super().__init__()

        self.tc        = tc_client
        self.log       = action_logger
        self.max_steps = max_steps

        self.encoder  = StateEncoder()
        self.reward   = TCRewardCalculator()
        self.executor = CommandExecutor()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE_TC,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS_TC)

        self._step_count = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.reward.reset()
        self._step_count = 0
        self.log.log_episode_start()

        # Si la partida anterior terminó, BWEnv espera un nuevo HandshakeClient
        # (auto_restart=ON reinicia el juego pero resetea el protocolo ZMQ).
        if self.tc.state is not None and self.tc.state.game_ended:
            if not self.tc.reconnect():
                return np.zeros(OBS_SIZE_TC, dtype=np.float32), {}

        # Avanzar un frame con NOOP para obtener el estado inicial
        ok = self.tc.recv()
        if ok and self.tc.state is not None:
            self.encoder.update_map_size(self.tc.state)
            obs = self.encoder.encode(self.tc.state)
        else:
            obs = np.zeros(OBS_SIZE_TC, dtype=np.float32)

        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        decoded = decode_tc_action(action)

        # ── LOG ──────────────────────────────────────────────────────────────
        self.log.log_action(
            action_id=action,
            action_name=decoded.description,
            details={
                "type":     decoded.action_type.name,
                "grid_row": decoded.grid_row,
                "grid_col": decoded.grid_col,
            },
        )

        # ── Ejecutar en la VM y recibir siguiente estado ──────────────────────
        commands = self.executor.build_commands(decoded, self.tc.state)
        ok = self.tc.send(commands)
        state = self.tc.state

        if ok and state is not None:
            obs    = self.encoder.encode(state)
            reward = self.reward.compute(state, decoded)

            # Fallback: terminar si todos los aliados o todos los enemigos mueren,
            # por si BWEnv no envía is_terminal correctamente.
            n_army   = self.reward._prev_army_count
            n_enemy  = self.reward._prev_enemy_count
            combat_over = (
                n_army  is not None and n_army  == 0 or
                n_enemy is not None and n_enemy == 0
            )
            truncated = state.game_ended or combat_over

            # Log periódico de unidades para diagnóstico
            if self._step_count % 200 == 0:
                self.log.log_info(
                    "UNITS | step=%d  army=%s  enemies=%s",
                    self._step_count, n_army, n_enemy,
                )
        else:
            obs       = np.zeros(OBS_SIZE_TC, dtype=np.float32)
            reward    = 0.0
            truncated = True

        self._step_count += 1
        terminated = self._step_count >= self.max_steps

        self.log.log_reward(reward, self.reward.cumulative)

        info = {"step": self._step_count}
        if terminated or truncated:
            self.log.log_episode_end(self.reward.cumulative, self._step_count)
            info.update(self.reward.unit_stats)

        return obs, reward, terminated, truncated, info

    def close(self):
        self.tc.close()
        self.log.close()
