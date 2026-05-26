"""
SEC 3 — Traductor TCAction → comandos BWAPI
Convierte una macro-acción del agente en la lista de comandos de bajo nivel
que TorchCraft envía al servidor BWAPI.

Sistema de coordenadas:
  · Posiciones para Move/Attack_Move: píxeles (unit.x, unit.y están en píxeles)
  · Posiciones para Build: BUILD TILES (1 tile = 32 px → pixel / 32)
  · state.map_size: [ancho, alto] en WALK TILES (1 tile = 8 px)
"""
import logging
import math

from sc1_rl.torchcraft.action_space import TCActionType, TCAction, GRID_SIZE

logger = logging.getLogger("sc1_rl")

# TorchCraft game-level command codes (NOT BWAPI UnitCommandType)
# code=0  → noop
# code=1  → QUIT  ← never send this!
# code=21 → COMMAND_UNIT_PROTECTED (only own units execute)
TC_CMD_UNIT_PROTECTED = 21

from sc1_rl.torchcraft.constants import (
    WORKER_TYPES, ARMY_TYPES, BUILDING_TYPES, RESOURCE_TYPES,
    MINERAL_TYPES, GAS_SOURCE_TYPES,
    CMD_ATTACK_MOVE, CMD_GATHER, CMD_BUILD, CMD_TRAIN,
    UTYPE_SCV, UTYPE_MARINE,
    UTYPE_CC, UTYPE_BARRACKS, UTYPE_SUPPLY,
    BUILDING_TRAINS,
)


class CommandExecutor:
    """Traduce TCAction al formato de lista de comandos de TorchCraft."""

    def __init__(self):
        self._own_unit_ids: frozenset | None = None

    def reset(self) -> None:
        """Llamar al inicio de cada episodio para reiniciar el split de equipos."""
        self._own_unit_ids = None

    def _init_teams(self, state) -> None:
        """Misma lógica de split posicional que TCRewardCalculator."""
        all_u = [
            u for units in state.units.values() for u in units.values()
            if u.type not in RESOURCE_TYPES and u.type not in BUILDING_TYPES
            and u.health > 0
        ]
        if len(all_u) < 2:
            return
        all_u.sort(key=lambda u: u.x + u.y)
        mid = len(all_u) // 2
        self._own_unit_ids = frozenset(u.id for u in all_u[:mid])

    def build_commands(self, action: TCAction, state) -> list:
        t = action.action_type
        if t == TCActionType.ATTACK_MOVE:
            return self._attack_move(action, state)
        return []

    # ── Helpers de coordenadas ────────────────────────────────────────────────

    def _grid_to_pixels(self, action: TCAction, state) -> tuple[int, int]:
        """Convierte (row, col) de la cuadrícula 8×8 a coordenadas en píxeles."""
        try:
            mw_walk, mh_walk = state.map_size
            map_w_px = int(mw_walk) * 8
            map_h_px = int(mh_walk) * 8
        except Exception:
            map_w_px = map_h_px = 1024
        x = int((action.grid_col + 0.5) / GRID_SIZE * map_w_px)
        y = int((action.grid_row + 0.5) / GRID_SIZE * map_h_px)
        return x, y

    def _grid_to_build_tiles(self, action: TCAction, state) -> tuple[int, int]:
        """Convierte (row, col) de la cuadrícula 8×8 a BUILD TILES (32 px)."""
        x_px, y_px = self._grid_to_pixels(action, state)
        return x_px // 32, y_px // 32

    # ── Recolección ───────────────────────────────────────────────────────────

    def _gather_idle(self, state) -> list:
        """Asigna todos los SCVs ociosos al recurso más cercano."""
        commands = []
        try:
            pid = state.player_id
            neutral = self._neutral_units(state)

            resources = [u for u in neutral.values() if u.type in MINERAL_TYPES | GAS_SOURCE_TYPES]
            if not resources:
                return []

            for uid, unit in state.units[pid].items():
                if unit.type not in WORKER_TYPES:
                    continue
                if not getattr(unit, "idle", True):
                    continue
                nearest = min(resources, key=lambda r: _dist(unit, r))
                commands.append([CMD_GATHER, uid, nearest.id if hasattr(nearest, "id") else uid, 0, 0, 0])
        except Exception:
            pass
        return commands

    # ── Ataque ────────────────────────────────────────────────────────────────

    def _attack_move(self, action: TCAction, state) -> list:
        """Envía las unidades del ejército propio en attack-move hacia la celda."""
        if self._own_unit_ids is None:
            self._init_teams(state)

        x, y = self._grid_to_pixels(action, state)
        commands = []
        try:
            for pid_units in state.units.values():
                for uid, unit in pid_units.items():
                    if self._own_unit_ids is not None and unit.id not in self._own_unit_ids:
                        continue
                    if unit.type in BUILDING_TYPES or unit.type in WORKER_TYPES or unit.type in RESOURCE_TYPES:
                        continue
                    if unit.health <= 0:
                        continue
                    # TorchCraft format: [TC_CMD_UNIT_PROTECTED, uid, bwapi_cmd, target_uid, x, y, extra]
                    commands.append([TC_CMD_UNIT_PROTECTED, uid, CMD_ATTACK_MOVE, -1, x, y, 0])
            if commands:
                logger.debug(
                    "ATTACK_MOVE → %d units → (%d,%d)  uids=%s",
                    len(commands), x, y, [c[1] for c in commands],
                )
        except Exception as exc:
            logger.warning("build_commands error: %s", exc)
        return commands

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build(self, action: TCAction, state, building_type: int) -> list:
        """Envía el primer SCV ocioso a construir un edificio en la celda indicada."""
        tx, ty = self._grid_to_build_tiles(action, state)
        try:
            for uid, unit in state.units[state.player_id].items():
                if unit.type in WORKER_TYPES and getattr(unit, "idle", False):
                    return [[CMD_BUILD, uid, -1, tx, ty, building_type]]
        except Exception:
            pass
        return []

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def _train(self, state, unit_type: int, building_type: int) -> list:
        """Entrena una unidad desde el primer edificio del tipo requerido que esté libre."""
        try:
            for uid, unit in state.units[state.player_id].items():
                if unit.type == building_type and not _is_training(unit):
                    return [[CMD_TRAIN, uid, -1, 0, 0, unit_type]]
        except Exception:
            pass
        return []

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _neutral_units(state) -> dict:
        if hasattr(state, "neutral_units"):
            return state.neutral_units or {}
        for pid in (11, 12, -1):
            if pid in state.units:
                return state.units[pid]
        return {}


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_training(unit) -> bool:
    return bool(
        getattr(unit, "training", False)
        or getattr(unit, "constructing", False)
    )
