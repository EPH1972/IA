"""
SEC 3 — Espacio de acciones TorchCraft (macro-comandos BWAPI)

En lugar de simular teclado/ratón, el agente emite macro-acciones de alto nivel
que se traducen directamente a comandos BWAPI dentro del juego.

Jerarquía de acciones:
  NOOP
  GATHER_IDLE_WORKERS         → asigna SCVs ociosos al mineral más cercano
  ATTACK_MOVE   (8×8 grid)   → toda la armada ataca hacia una celda del mapa
  BUILD_SUPPLY  (8×8 grid)   → envía un SCV a construir Supply Depot
  BUILD_BARRACKS(8×8 grid)   → envía un SCV a construir Barracks
  TRAIN_SCV                  → entrena SCV desde el Command Center
  TRAIN_MARINE               → entrena Marine desde cualquier Barracks
"""
from dataclasses import dataclass, field
from enum import IntEnum

GRID_SIZE = 8              # 8×8 = 64 posiciones espaciales
N_SPATIAL  = GRID_SIZE * GRID_SIZE


class TCActionType(IntEnum):
    NOOP                 = 0
    GATHER_IDLE_WORKERS  = 1
    ATTACK_MOVE          = 2
    BUILD_SUPPLY_DEPOT   = 3
    BUILD_BARRACKS       = 4
    TRAIN_SCV            = 5
    TRAIN_MARINE         = 6


@dataclass
class TCAction:
    action_type: TCActionType
    grid_row:    int   = 0
    grid_col:    int   = 0
    description: str   = field(default="")


N_ACTIONS_TC = (
    1            # NOOP
    + 1          # GATHER_IDLE_WORKERS
    + N_SPATIAL  # ATTACK_MOVE
    + N_SPATIAL  # BUILD_SUPPLY_DEPOT
    + N_SPATIAL  # BUILD_BARRACKS
    + 1          # TRAIN_SCV
    + 1          # TRAIN_MARINE
)  # = 196


def decode_tc_action(action_id: int) -> TCAction:
    """Decodifica un índice discreto en un TCAction."""
    cursor = action_id

    if cursor == 0:
        return TCAction(TCActionType.NOOP, description="noop")
    cursor -= 1

    if cursor == 0:
        return TCAction(TCActionType.GATHER_IDLE_WORKERS, description="gather_idle")
    cursor -= 1

    if cursor < N_SPATIAL:
        r, c = divmod(cursor, GRID_SIZE)
        return TCAction(TCActionType.ATTACK_MOVE, r, c, f"attack_move({r},{c})")
    cursor -= N_SPATIAL

    if cursor < N_SPATIAL:
        r, c = divmod(cursor, GRID_SIZE)
        return TCAction(TCActionType.BUILD_SUPPLY_DEPOT, r, c, f"build_supply({r},{c})")
    cursor -= N_SPATIAL

    if cursor < N_SPATIAL:
        r, c = divmod(cursor, GRID_SIZE)
        return TCAction(TCActionType.BUILD_BARRACKS, r, c, f"build_barracks({r},{c})")
    cursor -= N_SPATIAL

    if cursor == 0:
        return TCAction(TCActionType.TRAIN_SCV, description="train_scv")
    return TCAction(TCActionType.TRAIN_MARINE, description="train_marine")
