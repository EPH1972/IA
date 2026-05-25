"""
SEC 3 — Calculador de recompensa TorchCraft
Usa estado estructurado de BWAPI en lugar de OCR/brightness.

Escenario objetivo: m5v5_c_far.scm — 5 Marines Terran vs 5 Zerglings
La recompensa se centra en combate: matar Zerglings y mantener vivos los Marines.
"""
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
        self._cumulative       = 0.0

    def reset(self) -> None:
        self._prev_enemy_hp    = None
        self._prev_enemy_count = None
        self._prev_army_count  = None
        self._cumulative       = 0.0

    def _classify(self, state):
        """Returns (army_units, enemy_units) as lists of UnitState.

        Handles both cases:
          - Correct: own units under state.player_id, enemies under a different pid.
          - TorchCraft quirk: all visible units under state.player_id; enemies
            identified by unit type not in Terran sets.
        """
        own_pid = state.player_id
        army   = []
        enemies = []

        try:
            for pid, units in state.units.items():
                for u in units.values():
                    if u.type in RESOURCE_TYPES:
                        continue
                    if pid == own_pid:
                        # Could be own unit OR enemy under wrong pid
                        if u.type in ARMY_TYPES or u.type in WORKER_TYPES or u.type in BUILDING_TYPES:
                            if u.type in ARMY_TYPES:
                                army.append(u)
                        else:
                            # Non-Terran unit under own pid → treat as enemy
                            enemies.append(u)
                    else:
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

        self._prev_enemy_hp    = enemy_hp
        self._prev_enemy_count = enemy_count
        self._prev_army_count  = army_count

        self._cumulative += reward
        return reward

    @property
    def cumulative(self) -> float:
        return self._cumulative
