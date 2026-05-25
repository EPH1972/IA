"""
Punto de entrada — StarCraft 1 RL Agent (modo TorchCraft)

Uso:
    python main.py                      # entrena desde cero
    python main.py --resume <ckpt.pt>   # retoma desde un checkpoint
    python main.py --fresh              # ignora checkpoints existentes
    python main.py --config alt.yaml    # configuración alternativa

Prerrequisitos:
    · VM Win7 corriendo con StarCraft + BWAPI + TorchCraft.dll activos
    · pip install torchcraft
    · Ver setup_bwapi.md para instrucciones completas
"""
import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("sc1_rl").setLevel(logging.DEBUG)

from sc1_rl.logger.action_logger import ActionLogger
from sc1_rl.torchcraft.client import TorchCraftClient
from sc1_rl.torchcraft.action_space import N_ACTIONS_TC
from sc1_rl.torchcraft.state_encoder import OBS_SIZE_TC
from sc1_rl.environment.sc1_env_tc import SC1EnvTC
from sc1_rl.model.agent import PPOAgent
from sc1_rl.model.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    checkpoints = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Entrena un agente RL en StarCraft 1 (TorchCraft)")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config YAML")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Checkpoint .pt explícito (por defecto: el más reciente)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignora checkpoints existentes y entrena desde cero")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    tc_cfg    = cfg.get("torchcraft", {})
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    log_cfg   = cfg["logging"]

    # ── Logger ────────────────────────────────────────────────────────────────
    action_logger = ActionLogger(
        log_dir=log_cfg["dir"],
        log_file=log_cfg["action_log_file"],
    )

    # ── Conectar a TorchCraft ─────────────────────────────────────────────────
    host = tc_cfg.get("host", "127.0.0.1")
    port = tc_cfg.get("port", 11111)
    client = TorchCraftClient(
        host=host,
        port=port,
        connect_timeout=tc_cfg.get("connect_timeout", 120.0),
    )
    print(f"Conectando a TorchCraft en {host}:{port}...")
    if not client.connect():
        print(
            "ERROR: No se pudo conectar al servidor TorchCraft.\n"
            "Asegúrate de que la VM está corriendo con BWAPI + TorchCraft.dll activos.\n"
            "Consulta setup_bwapi.md para instrucciones."
        )
        return
    print("TorchCraft conectado.\n")

    # ── Entorno ───────────────────────────────────────────────────────────────
    env = SC1EnvTC(
        tc_client=client,
        action_logger=action_logger,
        max_steps=train_cfg.get("max_episode_steps", 50_000),
    )

    # ── Agente PPO (MLP sobre vector estructurado) ────────────────────────────
    agent = PPOAgent(
        obs_shape=(OBS_SIZE_TC,),
        n_actions=N_ACTIONS_TC,
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

    checkpoint = None if args.fresh else (args.resume or find_latest_checkpoint(train_cfg["checkpoint_dir"]))
    if args.fresh:
        print("--fresh: entrenamiento desde cero (checkpoints ignorados).")
    elif checkpoint:
        print(f"Reanudando desde: {checkpoint}")
        agent.load(checkpoint)
    else:
        print("No se encontró checkpoint — entrenamiento desde cero.")

    # ── Entrenar ──────────────────────────────────────────────────────────────
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
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    finally:
        trainer.close()
        env.close()
        print("Cerrado correctamente.")


if __name__ == "__main__":
    main()
