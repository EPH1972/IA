"""
SEC 1 — Calculador de Recompensa
Rewards escalados según la cadena causal de victoria en StarCraft BW:
  explorar → seleccionar → recolectar → construir → entrenar → atacar → ganar

Secuencia de construcción rastreada internamente:
  LEFT_CLICK (unit_selected) → key_b → key_edificio → LEFT_CLICK  →  building_placed_reward
"""
import numpy as np

from sc1_rl.environment.action_space import ActionType, decode_action

# Hotkeys válidos para seleccionar tipo de edificio tras abrir el menú build
_BUILDING_KEYS = {"s", "b", "r", "a", "t", "e", "f", "u"}


class RewardCalculator:
    """
    Jerarquía de rewards (de menor a mayor proximidad a la victoria):

    · Supervivencia              (+0.001)
    · Exploración cámara         (+0.0002)
    · Penalización NOOP          (−0.005)
    · Click fallido              (−0.001 neto)
    · Unidad/estructura selecc.  (+0.003)
    · Right-click con unidad     (+0.006)   → mover / recolectar
    · Mineral recolectado        (+0.010×Δ)
    · Gas recolectado            (+0.015×Δ)
    · Gather (g)                 (+0.010)   con unidad
    · Move (m)                   (+0.004)   con unidad
    · Build menu (b)             (+0.012)   con unidad  — paso 1 de construcción
    · Edificio seleccionado      (+0.015)   tras build menu — paso 2
    · Edificio colocado          (+0.040)   click tras seleccionar edificio — paso 3
    · Train (t)                  (+0.015)   con unidad
    · Attack-move (a)            (+0.020)   con unidad
    """

    # Estados de la máquina de construcción
    _BUILD_IDLE     = 0
    _BUILD_MENU     = 1   # 'b' pulsado con unidad
    _BUILD_SELECTED = 2   # hotkey de edificio pulsado

    def __init__(
        self,
        survival_bonus:          float = 0.001,
        noop_penalty:            float = 0.005,
        camera_reward:           float = 0.0002,
        unit_selected_reward:    float = 0.003,
        click_miss_penalty:      float = 0.002,
        right_click_unit_reward: float = 0.006,
        mineral_reward:          float = 0.010,
        gas_reward:              float = 0.015,
        gather_reward:           float = 0.010,
        move_reward:             float = 0.004,
        build_reward:            float = 0.012,
        building_select_reward:  float = 0.015,
        building_place_reward:   float = 0.040,
        train_reward:            float = 0.015,
        attack_reward:           float = 0.020,
    ):
        self.survival_bonus          = survival_bonus
        self.noop_penalty            = noop_penalty
        self.camera_reward           = camera_reward
        self.unit_selected_reward    = unit_selected_reward
        self.click_miss_penalty      = click_miss_penalty
        self.right_click_unit_reward = right_click_unit_reward
        self.mineral_reward          = mineral_reward
        self.gas_reward              = gas_reward
        self.gather_reward           = gather_reward
        self.move_reward             = move_reward
        self.build_reward            = build_reward
        self.building_select_reward  = building_select_reward
        self.building_place_reward   = building_place_reward
        self.train_reward            = train_reward
        self.attack_reward           = attack_reward

        self._step        = 0
        self._cumulative  = 0.0
        self._build_state = self._BUILD_IDLE
        self._build_steps = 0   # pasos desde que se abrió el menú

    def reset(self):
        self._step        = 0
        self._cumulative  = 0.0
        self._build_state = self._BUILD_IDLE
        self._build_steps = 0

    def compute(
        self,
        obs:            np.ndarray,
        action_id:      int,
        done:           bool,
        resource_delta: tuple[int, int] = (0, 0),
        unit_selected:  bool = False,
    ) -> float:
        reward  = self.survival_bonus
        decoded = decode_action(action_id)

        # ── Penalización NOOP ─────────────────────────────────────────────────
        if decoded.action_type == ActionType.NOOP:
            reward -= self.noop_penalty
            self._build_state = self._BUILD_IDLE

        # ── Exploración del mapa ──────────────────────────────────────────────
        if decoded.action_type in (
            ActionType.CAMERA_UP, ActionType.CAMERA_DOWN,
            ActionType.CAMERA_LEFT, ActionType.CAMERA_RIGHT,
        ):
            reward += self.camera_reward

        # ── Clicks ───────────────────────────────────────────────────────────
        if decoded.action_type == ActionType.LEFT_CLICK:
            if unit_selected:
                reward += self.unit_selected_reward
                # Resetea la secuencia de construcción si se selecciona algo nuevo
                # (solo si no estamos esperando colocar el edificio)
                if self._build_state != self._BUILD_SELECTED:
                    self._build_state = self._BUILD_IDLE
            else:
                # ── Paso 3: colocar edificio ──────────────────────────────────
                if self._build_state == self._BUILD_SELECTED:
                    reward += self.building_place_reward
                    self._build_state = self._BUILD_IDLE
                else:
                    reward -= self.click_miss_penalty

        if decoded.action_type == ActionType.RIGHT_CLICK and unit_selected:
            reward += self.right_click_unit_reward
            self._build_state = self._BUILD_IDLE

        # ── Recursos recolectados ─────────────────────────────────────────────
        mineral_delta, gas_delta = resource_delta
        reward += mineral_delta * self.mineral_reward
        reward += gas_delta     * self.gas_reward

        # ── Hotkeys estratégicos ──────────────────────────────────────────────
        if decoded.action_type == ActionType.KEYBOARD:
            key = decoded.key

            if key == "escape":
                self._build_state = self._BUILD_IDLE

            elif unit_selected:
                if key == "a":
                    reward += self.attack_reward
                    self._build_state = self._BUILD_IDLE
                elif key == "t":
                    reward += self.train_reward
                    self._build_state = self._BUILD_IDLE
                elif key == "g":
                    reward += self.gather_reward
                    self._build_state = self._BUILD_IDLE
                elif key == "m":
                    reward += self.move_reward
                    self._build_state = self._BUILD_IDLE
                elif key == "b":
                    # ── Paso 1: abrir menú de construcción ────────────────────
                    reward += self.build_reward
                    self._build_state = self._BUILD_MENU
                    self._build_steps = 0

            elif self._build_state == self._BUILD_MENU and key in _BUILDING_KEYS:
                # ── Paso 2: seleccionar tipo de edificio ──────────────────────
                reward += self.building_select_reward
                self._build_state = self._BUILD_SELECTED

        # Caducar la secuencia si pasan demasiados pasos sin completarla
        if self._build_state != self._BUILD_IDLE:
            self._build_steps += 1
            if self._build_steps > 20:
                self._build_state = self._BUILD_IDLE

        self._step      += 1
        self._cumulative += reward
        return reward

    @property
    def cumulative(self) -> float:
        return self._cumulative
