"""
SEC 3 — Codificador de estado TorchCraft
Convierte un frame de TorchCraft a un vector numpy de tamaño fijo.

Estructura del vector (total OBS_SIZE_TC = 189 features):
  [0:4]    Recursos: minerals, gas, supply_used, supply_max
  [4:44]   Propios workers  (hasta 8 × 5 features)
  [44:104] Propia armada    (hasta 12 × 5 features)
  [104:144] Propios edificios (hasta 8 × 5 features)
  [144:184] Enemigos visibles (hasta 10 × 4 features)
  [184:189] Resumen: n_workers, n_army, n_buildings, n_enemies, frame_norm
"""
import numpy as np

from sc1_rl.torchcraft.constants import (
    WORKER_TYPES, ARMY_TYPES, BUILDING_TYPES, RESOURCE_TYPES,
)


# ── Dimensiones del vector de observación ────────────────────────────────────
MAX_WORKERS   = 8
MAX_ARMY      = 12
MAX_BUILDINGS = 8
MAX_ENEMIES   = 10

WORKER_FEATS   = 5   # x, y, hp_norm, is_idle, is_gathering
ARMY_FEATS     = 5   # x, y, hp_norm, type_norm, is_attacking
BUILDING_FEATS = 5   # x, y, hp_norm, type_norm, is_training
ENEMY_FEATS    = 4   # x, y, hp_norm, type_norm

RESOURCE_FEATS = 4
SUMMARY_FEATS  = 5

OBS_SIZE_TC = (
    RESOURCE_FEATS
    + MAX_WORKERS   * WORKER_FEATS
    + MAX_ARMY      * ARMY_FEATS
    + MAX_BUILDINGS * BUILDING_FEATS
    + MAX_ENEMIES   * ENEMY_FEATS
    + SUMMARY_FEATS
)  # = 189


def _hp_norm(unit) -> float:
    max_hp = getattr(unit, "max_health", 0)
    if max_hp <= 0:
        return 1.0
    return unit.health / max_hp


def _type_norm(unit) -> float:
    return (unit.type % 64) / 64.0


def _is_training(unit) -> float:
    return float(
        getattr(unit, "training", False)
        or getattr(unit, "constructing", False)
    )


def _neutral_units(state) -> dict:
    """Devuelve unidades neutrales (minerales, gas) independientemente de la versión de TC."""
    if hasattr(state, "neutral_units"):
        return state.neutral_units or {}
    for pid in (11, 12, -1):
        if pid in state.units:
            return state.units[pid]
    return {}


class StateEncoder:
    """
    Encode a TorchCraft game state frame into a fixed-size float32 array.

    map_w / map_h se actualizan en el primer frame del episodio a partir de
    state.map_size (walk tiles × 8 = píxeles).
    """

    def __init__(self, map_w: int = 1024, map_h: int = 1024):
        self.map_w = map_w
        self.map_h = map_h

    def update_map_size(self, state) -> None:
        """Extrae dimensiones del mapa del estado y actualiza el encoder."""
        try:
            mw, mh = state.map_size
            self.map_w = int(mw) * 8
            self.map_h = int(mh) * 8
        except Exception:
            pass

    def encode(self, state) -> np.ndarray:
        obs = np.zeros(OBS_SIZE_TC, dtype=np.float32)
        ptr = 0

        # ── Recursos ──────────────────────────────────────────────────────────
        try:
            res = state.resources[state.player_id]
            obs[ptr]   = min(res.ore,       2000) / 2000.0
            obs[ptr+1] = min(res.gas,       2000) / 2000.0
            obs[ptr+2] = min(res.used_psi,  200)  / 200.0
            obs[ptr+3] = min(res.total_psi, 200)  / 200.0
        except Exception:
            pass
        ptr += RESOURCE_FEATS

        # ── Clasificar unidades propias ────────────────────────────────────────
        workers: list   = []
        army: list      = []
        buildings: list = []

        try:
            for unit in state.units[state.player_id].values():
                utype = unit.type
                if utype in WORKER_TYPES:
                    workers.append(unit)
                elif utype in BUILDING_TYPES:
                    buildings.append(unit)
                else:
                    army.append(unit)
        except Exception:
            pass

        # ── Workers ───────────────────────────────────────────────────────────
        for i in range(MAX_WORKERS):
            if i < len(workers):
                u = workers[i]
                obs[ptr]   = u.x / self.map_w
                obs[ptr+1] = u.y / self.map_h
                obs[ptr+2] = _hp_norm(u)
                obs[ptr+3] = float(getattr(u, "idle", False))
                obs[ptr+4] = float(
                    getattr(u, "gathering_minerals", False)
                    or getattr(u, "gathering_gas", False)
                )
            ptr += WORKER_FEATS

        # ── Armada ────────────────────────────────────────────────────────────
        for i in range(MAX_ARMY):
            if i < len(army):
                u = army[i]
                obs[ptr]   = u.x / self.map_w
                obs[ptr+1] = u.y / self.map_h
                obs[ptr+2] = _hp_norm(u)
                obs[ptr+3] = _type_norm(u)
                obs[ptr+4] = float(getattr(u, "attacking", False))
            ptr += ARMY_FEATS

        # ── Edificios ─────────────────────────────────────────────────────────
        for i in range(MAX_BUILDINGS):
            if i < len(buildings):
                u = buildings[i]
                obs[ptr]   = u.x / self.map_w
                obs[ptr+1] = u.y / self.map_h
                obs[ptr+2] = _hp_norm(u)
                obs[ptr+3] = _type_norm(u)
                obs[ptr+4] = _is_training(u)
            ptr += BUILDING_FEATS

        # ── Enemigos visibles ─────────────────────────────────────────────────
        # Handles both the normal case (enemies under a different pid) and the
        # TorchCraft quirk where all units appear under player_id=0 (enemies
        # identified by their unit type not belonging to any Terran category).
        enemies: list = []
        try:
            pid_self = state.player_id
            _own_types = WORKER_TYPES | ARMY_TYPES | BUILDING_TYPES
            for pid, units in state.units.items():
                for u in units.values():
                    if u.type in RESOURCE_TYPES:
                        continue
                    if pid != pid_self:
                        enemies.append(u)
                    elif u.type not in _own_types:
                        # Non-Terran unit under own pid → treat as enemy
                        enemies.append(u)
        except Exception:
            pass

        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                u = enemies[i]
                obs[ptr]   = u.x / self.map_w
                obs[ptr+1] = u.y / self.map_h
                obs[ptr+2] = _hp_norm(u)
                obs[ptr+3] = _type_norm(u)
            ptr += ENEMY_FEATS

        # ── Resumen ───────────────────────────────────────────────────────────
        obs[ptr]   = len(workers)  / 20.0
        obs[ptr+1] = len(army)     / 50.0
        obs[ptr+2] = len(buildings)/ 20.0
        obs[ptr+3] = len(enemies)  / 30.0
        try:
            obs[ptr+4] = min(state.battle_frame_count, 100_000) / 100_000.0
        except Exception:
            pass

        return obs
