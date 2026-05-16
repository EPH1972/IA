"""
SEC 1 — Bucle de Entrenamiento
Coordina al agente y al entorno durante rollouts completos.
"""
import time
from pathlib import Path

from sc1_rl.logger.action_logger import ActionLogger
from sc1_rl.model.agent import PPOAgent


class Trainer:
    """Ejecuta el bucle de entrenamiento PPO."""

    def __init__(
        self,
        agent: PPOAgent,
        env,                     # SC1Env
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

    def train(self):
        obs, _ = self.env.reset()
        global_step = 0
        episode = 0
        start = time.time()

        self.log.log_info("Inicio de entrenamiento — %d pasos totales", self.total_timesteps)

        while global_step < self.total_timesteps:
            action, log_prob, value = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.agent.store(obs, action, log_prob, float(reward), value, done)
            obs = next_obs
            global_step += 1

            if done:
                obs, _ = self.env.reset()
                episode += 1

            # Actualizar política cuando el buffer está lleno
            if global_step % self.agent.rollout_steps == 0:
                metrics = self.agent.update(obs)
                self.log.log_training_update(metrics)

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
