"""
SEC 3 — Calculador de recompensa TorchCraft
Usa estado estructurado de BWAPI en lugar de OCR/brightness.

Escenario objetivo: m5v5_c_far.scm — 5 Marines Terran vs 5 Zerglings
La recompensa se centra en combate: matar Zerglings y mantener vivos los Marines.
"""
import math
from sc1_rl.torchcraft.action_space import TCActionType
from sc1_rl.torchcraft.constants import (
    ARMY_TYPES, WORKER_TYPES, BUILDING_TYPES, RESOURCE_TYPES,
)

# Identifica unidades que no son propias ni recursos (= enemigas visibles)
_OWN_TYPES = ARMY_TYPES | WORKER_TYPES | BUILDING_TYPES


class TCRewardCalculator:
    """
    Recompensa orientada a micro-combate:

    · Daño a enemigo          +0.001 × HP_perdido
    · Enemigo eliminado       +0.50  por unidad
    · Marine muerto           −0.50  por unidad
    · Victoria (todos muertos) +10.0
    · Derrota (sin ejército)   −5.0
    · Attack-move enviado      +0.005
    · Supervivencia            +0.001 por paso
    · NOOP penalty             −0.003

    La identificación de "enemigos" funciona aunque TorchCraft ponga
    todas las unidades bajo el mismo player_id (filtra por tipo).
    """

    def __init__(
        self,
        survival_bonus:    float = 0.001,
        noop_penalty:      float = 0.003,
        attack_reward:     float = 0.005,
        enemy_damage_r:    float = 0.001,
        enemy_kill_r:      float = 0.50,
        allied_death_pen:  float = 0.50,
        win_reward:        float = 10.0,
        loss_penalty:      float = 5.0,
    ):
        self.survival_bonus   = survival_bonus
        self.noop_penalty     = noop_penalty
        self.attack_reward    = attack_reward
        self.enemy_damage_r   = enemy_damage_r
        self.enemy_kill_r     = enemy_kill_r
        self.allied_death_pen = allied_death_pen
        self.win_reward       = win_reward
        self.loss_penalty     = loss_penalty

        self._prev_enemy_hp    = None
        self._prev_enemy_count = None
        self._prev_army_count  = None
        self._prev_avg_dist    = None
        self._cumulative       = 0.0
        self._own_unit_ids     = None   # set on first classify call
        self._enemy_unit_ids   = None
        self._max_enemy_seen   = 0      # guard for combat_over

    def reset(self) -> None:
        self._prev_enemy_hp    = None
        self._prev_enemy_count = None
        self._prev_army_count  = None
        self._prev_avg_dist    = None
        self._cumulative       = 0.0
        self._own_unit_ids     = None
        self._enemy_unit_ids   = None
        self._max_enemy_seen   = 0

    def _init_teams_by_position(self, state) -> None:
        """Split units into own/enemy by initial position diagonal.

        BWEnv reports all unit player_ids as 0, so the only reliable separator
        is the initial position: one team starts near (0,0), the other near the
        far corner.  Lower x+y diagonal → own team (player 0 convention).
        """
        all_u = [
            u for units in state.units.values() for u in units.values()
            if u.type not in RESOURCE_TYPES and u.type not in BUILDING_TYPES
            and u.health > 0
        ]
        if len(all_u) < 2:
            return
        all_u.sort(key=lambda u: u.x + u.y)
        mid = len(all_u) // 2
        self._own_unit_ids   = frozenset(u.id for u in all_u[:mid])
        self._enemy_unit_ids = frozenset(u.id for u in all_u[mid:])
        import logging
        logging.getLogger("sc1_rl").info(
            "TEAMS own_ids=%s enemy_ids=%s",
            sorted(self._own_unit_ids), sorted(self._enemy_unit_ids),
        )

    def _classify(self, state):
        """Returns (army_units, enemy_units) as lists of UnitState."""
        if self._own_unit_ids is None:
            self._init_teams_by_position(state)

        army    = []
        enemies = []
        try:
            for units in state.units.values():
                for u in units.values():
                    if u.type in RESOURCE_TYPES:
                        continue
                    if self._own_unit_ids is None:
                        continue
                    if u.health <= 0:
                        continue
                    if u.id in self._own_unit_ids:
                        army.append(u)
                    elif u.id in self._enemy_unit_ids:
                        enemies.append(u)
        except Exception:
            pass
        return army, enemies

    def compute(self, state, action) -> float:
        reward = self.survival_bonus

        if action.action_type == TCActionType.NOOP:
            reward -= self.noop_penalty
        elif action.action_type == TCActionType.ATTACK_MOVE:
            reward += self.attack_reward

        army, enemies = self._classify(state)
        army_count  = len(army)
        enemy_count = len(enemies)
        enemy_hp    = sum(u.health for u in enemies)

        # ── Daño infligido ────────────────────────────────────────────────────
        if self._prev_enemy_hp is not None:
            hp_lost = self._prev_enemy_hp - enemy_hp
            if hp_lost > 0:
                reward += hp_lost * self.enemy_damage_r

        # ── Enemigos eliminados ───────────────────────────────────────────────
        if self._prev_enemy_count is not None:
            kills = self._prev_enemy_count - enemy_count
            if kills > 0:
                reward += kills * self.enemy_kill_r

        # ── Marines muertos ───────────────────────────────────────────────────
        if self._prev_army_count is not None:
            deaths = self._prev_army_count - army_count
            if deaths > 0:
                reward -= deaths * self.allied_death_pen

        # ── Condiciones terminales ────────────────────────────────────────────
        if self._prev_enemy_count is not None and self._prev_enemy_count > 0 and enemy_count == 0:
            reward += self.win_reward
        if self._prev_army_count is not None and self._prev_army_count > 0 and army_count == 0:
            reward -= self.loss_penalty

        # ── Distancia media al enemigo (incentiva acercarse) ──────────────────
        if len(army) > 0 and len(enemies) > 0:
            avg_dist = sum(
                min(math.hypot(m.x - e.x, m.y - e.y) for e in enemies)
                for m in army
            ) / len(army)
            if self._prev_avg_dist is not None:
                dist_delta = self._prev_avg_dist - avg_dist
                reward += dist_delta * 0.0005
            self._prev_avg_dist = avg_dist

        self._prev_enemy_hp    = enemy_hp
        self._prev_enemy_count = enemy_count
        self._prev_army_count  = army_count
        self._last_army        = army
        self._last_enemies     = enemies
        if enemy_count > self._max_enemy_seen:
            self._max_enemy_seen = enemy_count

        self._cumulative += reward
        return reward

    @property
    def cumulative(self) -> float:
        return self._cumulative

    @property
    def unit_stats(self) -> dict:
        """Estadísticas de unidades del último paso — para loggear al final del episodio."""
        army    = getattr(self, "_last_army",    []) or []
        enemies = getattr(self, "_last_enemies", []) or []

        def avg_hp(units):
            if not units:
                return 0.0
            return sum(u.health / max(u.max_health, 1) for u in units) / len(units)

        return {
            "army_count":    len(army),
            "enemy_count":   len(enemies),
            "army_hp_avg":   round(avg_hp(army),    3),
            "enemy_hp_avg":  round(avg_hp(enemies), 3),
        }
