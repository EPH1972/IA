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
        expected_unit_count: int = 10,   # unidades totales esperadas por escenario (2 Dragoons + 3 Zealots) x2
        frame_skip: int = 8,             # repetir la misma accion N frames para dar tiempo a moverse
    ):
        super().__init__()

        self.tc        = tc_client
        self.log       = action_logger
        self.max_steps = max_steps
        self._expected_units = expected_unit_count
        self._frame_skip     = max(1, frame_skip)

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
        self.executor.reset()
        self._step_count = 0
        self.log.log_episode_start()

        # Guardar IDs de unidades de la partida anterior (stale units).
        # BWEnv auto_restart mantiene vivas las unidades supervivientes; hay que
        # eliminarlas del estado del nuevo juego para no contaminar la observación.
        stale_ids: set[int] = set()
        if self.tc.state is not None:
            stale_ids = {
                u.id
                for pid_units in self.tc.state.units.values()
                for u in pid_units.values()
            }

        # Si la partida anterior terminó, BWEnv espera un nuevo HandshakeClient.
        # También reconectamos si el socket ZMQ está en estado de error (EFSM).
        needs_reconnect = self.tc.state is not None and self.tc.state.game_ended
        if needs_reconnect:
            if not self.tc.reconnect():
                return np.zeros(OBS_SIZE_TC, dtype=np.float32), {}

        # Avanzar un frame con NOOP para obtener el estado inicial.
        ok = self.tc.recv()
        if not ok and not needs_reconnect:
            # Socket puede estar en estado EFSM; crear socket fresco y reconectar.
            if self.tc.reconnect():
                ok = self.tc.recv()

        # Si el NOOP devolvió EndGame (game terminó durante el reset), reconectar.
        if ok and self.tc.state is not None and self.tc.state.game_ended:
            stale_ids |= {
                u.id
                for pid_units in self.tc.state.units.values()
                for u in pid_units.values()
            }
            if self.tc.reconnect():
                ok = self.tc.recv()

        # Eliminar unidades stale del estado del nuevo juego.
        if ok and self.tc.state is not None:
            # Paso 1: eliminar por IDs conocidos de la partida anterior.
            if stale_ids:
                for pid_units in self.tc.state.units.values():
                    for uid in list(pid_units.keys()):
                        if uid in stale_ids:
                            del pid_units[uid]

            # Paso 2 (fallback): si aún hay más unidades de las esperadas, conservar
            # solo las N más recientes (IDs más altos), que corresponden al juego actual.
            all_units = [
                (u.id, pid, uid)
                for pid, pid_units in self.tc.state.units.items()
                for uid, u in pid_units.items()
            ]
            if len(all_units) > self._expected_units:
                all_units.sort(key=lambda t: t[0], reverse=True)
                keep_ids = {t[0] for t in all_units[:self._expected_units]}
                for pid_units in self.tc.state.units.values():
                    for uid in list(pid_units.keys()):
                        u = pid_units[uid]
                        if u.id not in keep_ids:
                            del pid_units[uid]

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

        # ── Ejecutar en la VM: repetir la acción frame_skip frames ───────────
        commands = self.executor.build_commands(decoded, self.tc.state)
        reward    = 0.0
        truncated = False
        ok        = False

        for _fs in range(self._frame_skip):
            ok = self.tc.send(commands)
            state = self.tc.state

            if not ok or state is None:
                truncated = True
                break

            reward += self.reward.compute(state, decoded)

            n_army    = self.reward._prev_army_count
            n_enemy   = self.reward._prev_enemy_count
            max_enemy = getattr(self.reward, "_max_enemy_seen", 0)
            combat_over = (
                n_army  is not None and n_army  == 0 or
                (n_enemy is not None and n_enemy == 0 and max_enemy > 0)
            )

            if state.game_ended or combat_over:
                # Drenar frames hasta recibir EndGame oficial
                if combat_over and not state.game_ended:
                    for _ in range(60):
                        if not self.tc.send([]):
                            break
                        state = self.tc.state
                        if state is None or state.game_ended:
                            break
                    if state is not None and not state.game_ended:
                        state.game_ended = True
                truncated = True
                break

        if ok and state is not None:
            obs = self.encoder.encode(state)

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
