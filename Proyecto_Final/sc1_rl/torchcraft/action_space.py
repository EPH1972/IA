"""
SEC 3 — Espacio de acciones TorchCraft (modo combate puro)

Para el escenario m5v5_c_far.scm (Marines vs Zerglings) solo tienen sentido
dos tipos de acción: NOOP y ATTACK_MOVE en la cuadrícula 8×8.
Reducir de 196 a 65 acciones hace que el 98% de acciones aleatorias sean
útiles, acelerando enormemente el aprendizaje inicial.
"""
from dataclasses import dataclass, field
from enum import IntEnum

GRID_SIZE = 8              # 8×8 = 64 posiciones espaciales
N_SPATIAL  = GRID_SIZE * GRID_SIZE


class TCActionType(IntEnum):
    NOOP         = 0
    ATTACK_MOVE  = 2


@dataclass
class TCAction:
    action_type: TCActionType
    grid_row:    int = 0
    grid_col:    int = 0
    description: str = field(default="")


# 64 posiciones ATTACK_MOVE + 1 NOOP = 65 acciones
N_ACTIONS_TC = N_SPATIAL + 1


def decode_tc_action(action_id: int) -> TCAction:
    """Decodifica un índice discreto en un TCAction."""
    if action_id == 0:
        return TCAction(TCActionType.NOOP, description="noop")
    idx = action_id - 1
    r, c = divmod(idx, GRID_SIZE)
    return TCAction(TCActionType.ATTACK_MOVE, r, c, f"attack_move({r},{c})")
