"""
Visualizador de entrenamiento — StarCraft 1 RL (modo TorchCraft)

Lee los CSVs generados por trainer.py y produce un panel PNG con 6 gráficas.

Uso:
    python plot_logs.py              # genera una vez y sale
    python plot_logs.py --watch      # regenera cada 30 s mientras entrena
    python plot_logs.py --watch 60   # intervalo personalizado (segundos)
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

EP_FILE      = Path("logs/episodes.csv")
METRICS_FILE = Path("logs/metrics.csv")
OUT_FILE     = Path("logs/training_progress.png")

BG      = "#0d0d1a"
PANEL   = "#12122a"
GRID    = "#1e1e3a"
TEXT    = "#e8e8f0"
COLORS  = ["#00e5ff", "#ff6b35", "#39ff14", "#ff00ff",
           "#ffd700", "#00ff99", "#ff4466", "#aa88ff"]
C_BLUE  = "#00e5ff"
C_ORG   = "#ff6b35"
C_GRN   = "#39ff14"
C_PRP   = "#cc88ff"
C_YEL   = "#ffd700"
C_RED   = "#ff4466"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=6)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)


def _ma(arr, w):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError):
            pass
    return np.array(out, dtype=float)


# ── Generador principal ───────────────────────────────────────────────────────

def generate():
    ep_rows  = _read_csv(EP_FILE)
    met_rows = _read_csv(METRICS_FILE)

    if not ep_rows and not met_rows:
        print("No hay datos todavía. Espera a que el entrenamiento genere logs.")
        return

    # ── Datos de episodios ────────────────────────────────────────────────────
    ep_nums    = _floats(ep_rows, "episode")
    ep_rewards = _floats(ep_rows, "reward")
    ep_steps   = _floats(ep_rows, "steps")
    ep_fps     = _floats(ep_rows, "fps")
    ep_gstep   = _floats(ep_rows, "global_step")

    # ── Datos de updates PPO ──────────────────────────────────────────────────
    m_step    = _floats(met_rows, "global_step")
    m_ploss   = _floats(met_rows, "policy_loss")
    m_vloss   = _floats(met_rows, "value_loss")
    m_entropy = _floats(met_rows, "entropy")
    m_kl      = _floats(met_rows, "approx_kl")
    m_clip    = _floats(met_rows, "clip_fraction")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── 1. Reward por episodio (fila 0, columnas 0-1) ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1, "Reward total por episodio")

    if len(ep_rewards) > 0:
        ax1.scatter(ep_nums, ep_rewards, color=C_BLUE, s=12, alpha=0.5, zorder=3,
                    label="Episodio")
        w = max(1, len(ep_rewards) // 20)
        if len(ep_rewards) >= w:
            smooth = _ma(ep_rewards, w)
            xs = ep_nums[w - 1:]
            ax1.plot(xs, smooth, color=C_ORG, linewidth=2.0, zorder=4,
                     label=f"Media móvil ({w})")

        # Línea del mejor episodio
        best_idx = int(np.argmax(ep_rewards))
        ax1.axhline(ep_rewards[best_idx], color="#555577", linewidth=0.8,
                    linestyle="--", alpha=0.6)
        ax1.annotate(
            f"  Mejor: {ep_rewards[best_idx]:.3f}  (ep {int(ep_nums[best_idx])})",
            xy=(ep_nums[best_idx], ep_rewards[best_idx]),
            color=TEXT, fontsize=8, va="bottom",
        )

    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    # ── 2. Duración de episodios (fila 0, columna 2) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, "Pasos por episodio")

    if len(ep_steps) > 0:
        ax2.scatter(ep_nums, ep_steps, color=C_GRN, s=10, alpha=0.5, zorder=3)
        w = max(1, len(ep_steps) // 20)
        if len(ep_steps) >= w:
            ax2.plot(ep_nums[w - 1:], _ma(ep_steps, w), color=C_YEL,
                     linewidth=1.8, zorder=4)

    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")

    # ── 3. Policy loss (fila 1, columna 0) ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "Policy Loss")

    if len(m_ploss) > 0:
        ax3.plot(m_step, m_ploss, color=C_BLUE, linewidth=1.2, alpha=0.6)
        w = max(1, len(m_ploss) // 15)
        if len(m_ploss) >= w:
            ax3.plot(m_step[w - 1:], _ma(m_ploss, w), color=C_ORG,
                     linewidth=2.0, label="Media")
        ax3.axhline(0, color=GRID, linewidth=0.8, linestyle="--")

    ax3.set_xlabel("Step global")
    ax3.set_ylabel("Loss")

    # ── 4. Value loss (fila 1, columna 1) ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "Value Loss")

    if len(m_vloss) > 0:
        ax4.plot(m_step, m_vloss, color=C_PRP, linewidth=1.2, alpha=0.6)
        w = max(1, len(m_vloss) // 15)
        if len(m_vloss) >= w:
            ax4.plot(m_step[w - 1:], _ma(m_vloss, w), color=C_YEL,
                     linewidth=2.0)

    ax4.set_xlabel("Step global")
    ax4.set_ylabel("Loss")

    # ── 5. Entropía y KL (fila 1, columna 2) ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5, "Entropia  /  KL divergencia")

    if len(m_entropy) > 0:
        ax5.plot(m_step, m_entropy, color=C_GRN, linewidth=1.4,
                 label="Entropía", zorder=3)
    if len(m_kl) > 0:
        ax5r = ax5.twinx()
        ax5r.tick_params(colors=TEXT, labelsize=8)
        ax5r.plot(m_step, m_kl, color=C_RED, linewidth=1.2, alpha=0.8,
                  label="KL approx")
        ax5r.set_ylabel("KL", color=TEXT)
        ax5r.tick_params(axis="y", colors=TEXT)

    ax5.set_xlabel("Step global")
    ax5.set_ylabel("Entropía")
    ax5.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    # ── 6. FPS a lo largo del tiempo (fila 2, columna 0-1) ───────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    _style(ax6, "Frames por segundo (velocidad de entrenamiento)")

    if len(ep_fps) > 0:
        ax6.plot(ep_gstep, ep_fps, color=C_YEL, linewidth=1.0, alpha=0.7)
        w = max(1, len(ep_fps) // 30)
        if len(ep_fps) >= w:
            ax6.plot(ep_gstep[w - 1:], _ma(ep_fps, w), color=C_ORG,
                     linewidth=2.0, label="Media móvil")
        ax6.axhline(np.mean(ep_fps), color="#555577", linewidth=0.8,
                    linestyle="--",
                    label=f"Media global: {np.mean(ep_fps):.1f} fps")

    ax6.set_xlabel("Step global")
    ax6.set_ylabel("FPS")
    ax6.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    # ── 7. Distribución de reward por episodio (fila 2, columna 2) ───────────
    ax7 = fig.add_subplot(gs[2, 2])
    _style(ax7, "Distribución de reward por episodio")

    if len(ep_rewards) > 0:
        ax7.hist(ep_rewards, bins=min(30, max(5, len(ep_rewards) // 3)),
                 color=C_BLUE, alpha=0.7, density=False, zorder=3)
        ax7.axvline(np.mean(ep_rewards), color=C_ORG, linewidth=1.5,
                    linestyle="--", label=f"Media: {np.mean(ep_rewards):.3f}")
        if len(ep_rewards) >= 5:
            recent = ep_rewards[-max(5, len(ep_rewards) // 5):]
            ax7.axvline(np.mean(recent), color=C_GRN, linewidth=1.5,
                        linestyle="--",
                        label=f"Reciente: {np.mean(recent):.3f}")

    ax7.set_xlabel("Reward")
    ax7.set_ylabel("Episodios")
    ax7.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)

    # ── Título global ─────────────────────────────────────────────────────────
    n_ep  = len(ep_rewards)
    n_upd = len(m_ploss)
    g_step = int(ep_gstep[-1]) if len(ep_gstep) > 0 else 0
    mean_r = float(np.mean(ep_rewards)) if n_ep > 0 else 0.0

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    fig.suptitle(
        f"StarCraft 1 RL — Progreso del entrenamiento\n"
        f"Episodios: {n_ep}   |   Steps: {g_step:,}   |   "
        f"Updates PPO: {n_upd}   |   Reward medio: {mean_r:.3f}\n"
        f"Actualizado: {now}",
        color=TEXT, fontsize=11, y=1.00,
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[{now}] Panel guardado → {OUT_FILE.resolve()}")


# ── Entrada ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    watch    = "--watch" in sys.argv
    interval = 30
    for arg in sys.argv[1:]:
        if arg.isdigit():
            interval = int(arg)

    generate()

    if watch:
        print(f"Modo watch — actualizando cada {interval} s.  Ctrl+C para salir.")
        try:
            while True:
                time.sleep(interval)
                generate()
        except KeyboardInterrupt:
            print("Watch detenido.")
