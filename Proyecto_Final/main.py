"""
Punto de entrada — StarCraft 1 RL Agent

Uso:
    python main.py                      # entrena desde cero
    python main.py --resume <ckpt.pt>   # retoma desde un checkpoint
    python main.py --config mi_cfg.yaml # usa configuración alternativa
"""
import argparse
from pathlib import Path

import yaml

# ── SEC 2: comunicación con VM ────────────────────────────────────────────────
from sc1_rl.vm.controller import VMController

# ── SEC 1: modelo y entorno ───────────────────────────────────────────────────
from sc1_rl.logger.action_logger import ActionLogger
from sc1_rl.environment.sc1_env import SC1Env
from sc1_rl.model.agent import PPOAgent
from sc1_rl.model.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Entrena un agente RL en StarCraft 1")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config YAML")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Checkpoint .pt desde el que reanudar")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vm_cfg = cfg["vm"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    # ── SEC 2: Conectar a la VM ───────────────────────────────────────────────
    vm = VMController(
        vm_name=vm_cfg["name"],
        vm_width=vm_cfg["resolution"][0],
        vm_height=vm_cfg["resolution"][1],
        screenshot_dir=vm_cfg.get("screenshot_dir"),
    )
    print(f"Conectando a la VM '{vm_cfg['name']}'...")
    if not vm.connect(start_if_stopped=True):
        print("ERROR: No se pudo conectar a la VM. Asegúrate de que está encendida.")
        return
    print("VM lista.\n")

    # ── SEC 1: Logger ─────────────────────────────────────────────────────────
    action_logger = ActionLogger(
        log_dir=log_cfg["dir"],
        log_file=log_cfg["action_log_file"],
    )

    # ── SEC 1: Entorno ────────────────────────────────────────────────────────
    obs_shape: tuple[int, int, int] = tuple(model_cfg["observation_shape"])  # type: ignore[assignment]
    env = SC1Env(
        vm_controller=vm,
        action_logger=action_logger,
        obs_width=obs_shape[2],
        obs_height=obs_shape[1],
        frame_stack=obs_shape[0],
        action_delay=vm_cfg.get("action_delay", 0.15),
        max_steps=train_cfg.get("max_episode_steps", 10_000),
    )

    # ── SEC 1: Agente PPO ─────────────────────────────────────────────────────
    agent = PPOAgent(
        obs_shape=obs_shape,
        device=model_cfg["device"],
        lr=model_cfg["learning_rate"],
        gamma=model_cfg["gamma"],
        gae_lambda=model_cfg["gae_lambda"],
        clip_epsilon=model_cfg["clip_epsilon"],
        epochs=model_cfg["epochs"],
        batch_size=model_cfg["batch_size"],
        rollout_steps=model_cfg["rollout_steps"],
        value_coef=model_cfg["value_coef"],
        entropy_coef=model_cfg["entropy_coef"],
        max_grad_norm=model_cfg["max_grad_norm"],
    )

    if args.resume:
        print(f"Reanudando desde: {args.resume}")
        agent.load(args.resume)

    # ── SEC 1: Entrenar ───────────────────────────────────────────────────────
    trainer = Trainer(
        agent=agent,
        env=env,
        action_logger=action_logger,
        total_timesteps=train_cfg["total_timesteps"],
        save_interval=train_cfg["save_interval"],
        log_interval=train_cfg["log_interval"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
    )

    print("Entrenamiento iniciado. Logs en:", Path(log_cfg["dir"]).resolve())
    print("Acciones enviadas a la VM →", Path(log_cfg["dir"], log_cfg["action_log_file"]).resolve())
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    finally:
        env.close()
        print("Cerrado correctamente.")


if __name__ == "__main__":
    main()
