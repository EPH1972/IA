"""
SEC 1 — Bucle de Entrenamiento
Coordina al agente y al entorno durante rollouts completos.
"""
import csv
import time
from pathlib import Path

from sc1_rl.logger.action_logger import ActionLogger
from sc1_rl.model.agent import PPOAgent

_METRICS_HEADER = [
    "global_step", "episode", "fps",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
]


class Trainer:
    """Ejecuta el bucle de entrenamiento PPO."""

    def __init__(
        self,
        agent: PPOAgent,
        env,
        action_logger: ActionLogger,
        total_timesteps: int = 1_000_000,
        save_interval: int = 10_000,
        log_interval: int = 1_000,
        checkpoint_dir: str = "checkpoints",
    ):
        self.agent = agent
        self.env = env
        self.log = action_logger
        self.total_timesteps = total_timesteps
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CSV de métricas para plot_logs.py
        self._metrics_path = Path("logs/metrics.csv")
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file   = open(self._metrics_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_METRICS_HEADER)
        if self._metrics_path.stat().st_size == 0:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        # CSV de episodios
        self._ep_path = Path("logs/episodes.csv")
        self._ep_file   = open(self._ep_path, "a", newline="", encoding="utf-8")
        self._ep_writer = csv.DictWriter(
            self._ep_file,
            fieldnames=["global_step", "episode", "reward", "steps", "fps"],
        )
        if self._ep_path.stat().st_size == 0:
            self._ep_writer.writeheader()
            self._ep_file.flush()

    def train(self):
        obs, _ = self.env.reset()
        global_step = 0
        episode = 0
        ep_reward = 0.0
        ep_steps  = 0
        start = time.time()

        self.log.log_info("Inicio de entrenamiento — %d pasos totales", self.total_timesteps)

        while global_step < self.total_timesteps:
            action, log_prob, value = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.agent.store(obs, action, log_prob, float(reward), value, done)
            obs = next_obs
            global_step += 1
            ep_reward  += float(reward)
            ep_steps   += 1

            if done:
                fps = global_step / (time.time() - start)
                self._ep_writer.writerow({
                    "global_step": global_step,
                    "episode":     episode + 1,
                    "reward":      round(ep_reward, 4),
                    "steps":       ep_steps,
                    "fps":         round(fps, 1),
                })
                self._ep_file.flush()
                ep_reward = 0.0
                ep_steps  = 0
                obs, _ = self.env.reset()
                episode += 1

            # Actualizar política cuando el buffer está lleno
            if global_step % self.agent.rollout_steps == 0:
                metrics = self.agent.update(obs)
                self.log.log_training_update(metrics)
                fps = global_step / (time.time() - start)
                self._csv_writer.writerow({
                    "global_step":   global_step,
                    "episode":       episode,
                    "fps":           round(fps, 1),
                    "policy_loss":   round(metrics.get("policy_loss",   0), 6),
                    "value_loss":    round(metrics.get("value_loss",    0), 6),
                    "entropy":       round(metrics.get("entropy",       0), 6),
                    "approx_kl":     round(metrics.get("approx_kl",     0), 6),
                    "clip_fraction": round(metrics.get("clip_fraction", 0), 6),
                })
                self._csv_file.flush()

            # Guardar checkpoint
            if global_step % self.save_interval == 0:
                path = self.checkpoint_dir / f"step_{global_step:08d}.pt"
                self.agent.save(str(path))
                self.log.log_info("Checkpoint guardado: %s", path)

            # Log periódico
            if global_step % self.log_interval == 0:
                fps = global_step / (time.time() - start)
                self.log.log_info(
                    "step=%d  fps=%.1f  episodios=%d",
                    global_step, fps, episode,
                )

    def close(self):
        self._csv_file.close()
        self._ep_file.close()
