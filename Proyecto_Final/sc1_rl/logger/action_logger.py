"""
SEC 1 — Action Logger
Registra cada acción que el agente envía a la VM con timestamp, episodio y detalles.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any


class ActionLogger:
    """
    Doble canal de log:
      - JSONL estructurado (action_log_file) → una línea por acción enviada
      - Python logging estándar             → consola + training.log
    """

    def __init__(self, log_dir: str, log_file: str = "action_log.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._action_path = self.log_dir / log_file
        self._action_file = open(self._action_path, "a", encoding="utf-8")

        self.logger = logging.getLogger("sc1_rl")
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

            fh = logging.FileHandler(self.log_dir / "training.log", encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

        self.logger.setLevel(logging.DEBUG)

        self._step = 0
        self._episode = 0

    # ── Acciones enviadas a la VM ─────────────────────────────────────────────

    def log_action(self, action_id: int, action_name: str, details: dict[str, Any]):
        """Registra una acción en el JSONL y en el logger estándar."""
        entry = {
            "ts": round(time.time(), 4),
            "episode": self._episode,
            "step": self._step,
            "action_id": action_id,
            "action": action_name,
            **details,
        }
        self._action_file.write(json.dumps(entry) + "\n")
        self._action_file.flush()
        self.logger.debug(
            "→ VM | ep=%d step=%d | [%d] %s %s",
            self._episode, self._step, action_id, action_name, details,
        )
        self._step += 1

    # ── Recompensas ───────────────────────────────────────────────────────────

    def log_reward(self, reward: float, cumulative: float):
        self.logger.debug("  reward=%.5f  cumulative=%.3f", reward, cumulative)

    # ── Control de episodios ──────────────────────────────────────────────────

    def log_episode_start(self):
        self._episode += 1
        self._step = 0
        self.logger.info("══ EPISODIO %d INICIO ══", self._episode)

    def log_episode_end(self, total_reward: float, steps: int):
        self.logger.info(
            "══ EPISODIO %d FIN | reward=%.3f | steps=%d ══",
            self._episode, total_reward, steps,
        )

    # ── Métricas de entrenamiento ─────────────────────────────────────────────

    def log_training_update(self, metrics: dict[str, float]):
        self.logger.info(
            "UPDATE | %s",
            "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()),
        )

    def log_info(self, msg: str, *args):
        self.logger.info(msg, *args)

    def close(self):
        self._action_file.close()
