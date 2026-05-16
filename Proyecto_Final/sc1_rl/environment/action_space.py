"""
SEC 1 — Espacio de acciones de StarCraft 1
Convierte un índice discreto en una acción estructurada (Action).
"""
from dataclasses import dataclass, field
from enum import IntEnum


class ActionType(IntEnum):
    NOOP = 0
    CAMERA_UP = 1
    CAMERA_DOWN = 2
    CAMERA_LEFT = 3
    CAMERA_RIGHT = 4
    LEFT_CLICK = 5
    RIGHT_CLICK = 6
    KEYBOARD = 7
    CONTROL_GROUP_SELECT = 8
    CONTROL_GROUP_SET = 9


@dataclass
class Action:
    action_type: ActionType
    x: float = 0.0          # coordenada X normalizada [0, 1]
    y: float = 0.0          # coordenada Y normalizada [0, 1]
    key: str = ""
    group: int = 0
    description: str = ""


# ── Configuración del grid de clics ──────────────────────────────────────────
GRID_SIZE = 16              # 16x16 = 256 posiciones por botón

# ── Hotkeys de StarCraft 1 ────────────────────────────────────────────────────
KEYBOARD_ACTIONS: list[str] = [
    "a",        # attack-move
    "b",        # build menu
    "s",        # stop
    "h",        # hold position
    "p",        # patrol
    "u",        # unload
    "r",        # repair / research
    "t",        # train / stim
    "m",        # move
    "g",        # gather / return cargo
    "escape",
    "enter",
    "f10",      # menú de juego
]

N_GRID = GRID_SIZE * GRID_SIZE          # 256
N_KEYBOARD = len(KEYBOARD_ACTIONS)      # 13
N_CONTROL_GROUPS = 9                    # grupos 1-9

N_ACTIONS = (
    1                    # NOOP
    + 4                  # cámara
    + N_GRID             # left clicks
    + N_GRID             # right clicks
    + N_KEYBOARD         # teclado
    + N_CONTROL_GROUPS   # seleccionar grupo
    + N_CONTROL_GROUPS   # asignar grupo (Ctrl+N)
)  # total: 548


def decode_action(action_id: int) -> Action:
    """Decodifica un índice entero en un objeto Action."""
    cursor = action_id

    if cursor == 0:
        return Action(ActionType.NOOP, description="noop")
    cursor -= 1

    if cursor < 4:
        dirs = ["up", "down", "left", "right"]
        types = [
            ActionType.CAMERA_UP, ActionType.CAMERA_DOWN,
            ActionType.CAMERA_LEFT, ActionType.CAMERA_RIGHT,
        ]
        return Action(types[cursor], description=f"camera_{dirs[cursor]}")
    cursor -= 4

    if cursor < N_GRID:
        row, col = divmod(cursor, GRID_SIZE)
        x = (col + 0.5) / GRID_SIZE
        y = (row + 0.5) / GRID_SIZE
        return Action(ActionType.LEFT_CLICK, x=x, y=y,
                      description=f"left_click({x:.2f},{y:.2f})")
    cursor -= N_GRID

    if cursor < N_GRID:
        row, col = divmod(cursor, GRID_SIZE)
        x = (col + 0.5) / GRID_SIZE
        y = (row + 0.5) / GRID_SIZE
        return Action(ActionType.RIGHT_CLICK, x=x, y=y,
                      description=f"right_click({x:.2f},{y:.2f})")
    cursor -= N_GRID

    if cursor < N_KEYBOARD:
        key = KEYBOARD_ACTIONS[cursor]
        return Action(ActionType.KEYBOARD, key=key, description=f"key_{key}")
    cursor -= N_KEYBOARD

    if cursor < N_CONTROL_GROUPS:
        g = cursor + 1
        return Action(ActionType.CONTROL_GROUP_SELECT, group=g,
                      description=f"select_group_{g}")
    cursor -= N_CONTROL_GROUPS

    g = cursor + 1
    return Action(ActionType.CONTROL_GROUP_SET, group=g,
                  description=f"set_group_{g}")
