"""
SEC 3 — Constantes BWAPI derivadas de TorchCraft
Carga los tipos de unidad y comando desde el paquete torchcraft cuando está disponible.
Si no está instalado, usa los enteros directos de BWAPI (verificados contra la spec).
"""

try:
    import torchcraft as tc
    _TC_AVAILABLE = True

    # ── Tipos de unidad ──────────────────────────────────────────────────────
    UTYPE_SCV              = tc.BW.UnitType.Terran_SCV
    UTYPE_MARINE           = tc.BW.UnitType.Terran_Marine
    UTYPE_FIREBAT          = tc.BW.UnitType.Terran_Firebat
    UTYPE_GHOST            = tc.BW.UnitType.Terran_Ghost
    UTYPE_VULTURE          = tc.BW.UnitType.Terran_Vulture
    UTYPE_SIEGE_TANK       = tc.BW.UnitType.Terran_Siege_Tank_Tank_Mode
    UTYPE_CC               = tc.BW.UnitType.Terran_Command_Center
    UTYPE_SUPPLY           = tc.BW.UnitType.Terran_Supply_Depot
    UTYPE_BARRACKS         = tc.BW.UnitType.Terran_Barracks
    UTYPE_FACTORY          = tc.BW.UnitType.Terran_Factory
    UTYPE_STARPORT         = tc.BW.UnitType.Terran_Starport
    UTYPE_ENG_BAY          = tc.BW.UnitType.Terran_Engineering_Bay
    UTYPE_MINERAL_FIELD    = tc.BW.UnitType.Resource_Mineral_Field
    UTYPE_MINERAL_FIELD_2  = tc.BW.UnitType.Resource_Mineral_Field_Type_2
    UTYPE_MINERAL_FIELD_3  = tc.BW.UnitType.Resource_Mineral_Field_Type_3
    UTYPE_VESPENE          = tc.BW.UnitType.Resource_Vespene_Geyser
    UTYPE_REFINERY         = tc.BW.UnitType.Terran_Refinery

    # ── Tipos de comando BWAPI ───────────────────────────────────────────────
    CMD_MOVE               = tc.BW.UnitCommandType.Move
    CMD_ATTACK_MOVE        = tc.BW.UnitCommandType.Attack_Move
    CMD_GATHER             = tc.BW.UnitCommandType.Gather
    CMD_BUILD              = tc.BW.UnitCommandType.Build
    CMD_TRAIN              = tc.BW.UnitCommandType.Train
    CMD_RIGHT_CLICK_POS    = tc.BW.UnitCommandType.Right_Click_Position
    CMD_RIGHT_CLICK_UNIT   = tc.BW.UnitCommandType.Right_Click_Unit

except ImportError:
    _TC_AVAILABLE = False

    # Enteros BWAPI (fuente: bwapi.github.io/bwapi/class_b_w_a_p_i_1_1_unit_type.html)
    UTYPE_MARINE           = 0
    UTYPE_GHOST            = 1
    UTYPE_VULTURE          = 2
    UTYPE_GOLIATH          = 3
    UTYPE_SIEGE_TANK       = 5
    UTYPE_SCV              = 7
    UTYPE_WRAITH           = 8
    UTYPE_FIREBAT          = 32
    UTYPE_CC               = 106
    UTYPE_SUPPLY           = 109
    UTYPE_BARRACKS         = 111
    UTYPE_FACTORY          = 113
    UTYPE_STARPORT         = 116
    UTYPE_ENG_BAY          = 117
    UTYPE_REFINERY         = 120
    UTYPE_MINERAL_FIELD    = 176
    UTYPE_MINERAL_FIELD_2  = 177
    UTYPE_MINERAL_FIELD_3  = 178
    UTYPE_VESPENE          = 188

    CMD_MOVE               = 10   # BWAPI UnitCommandType::Move
    CMD_ATTACK_MOVE        = 1    # BWAPI UnitCommandType::Attack_Move
    CMD_GATHER             = 15   # BWAPI UnitCommandType::Gather
    CMD_BUILD              = 2    # BWAPI UnitCommandType::Build
    CMD_TRAIN              = 4    # BWAPI UnitCommandType::Train
    CMD_RIGHT_CLICK_POS    = 30   # BWAPI UnitCommandType::Right_Click_Position
    CMD_RIGHT_CLICK_UNIT   = 31   # BWAPI UnitCommandType::Right_Click_Unit


# ── Conjuntos de clasificación ─────────────────────────────────────────────────
WORKER_TYPES = frozenset({UTYPE_SCV})

ARMY_TYPES = frozenset({
    UTYPE_MARINE, UTYPE_FIREBAT, UTYPE_GHOST,
    UTYPE_VULTURE, UTYPE_SIEGE_TANK,
})

BUILDING_TYPES = frozenset({
    UTYPE_CC, UTYPE_SUPPLY, UTYPE_BARRACKS,
    UTYPE_FACTORY, UTYPE_STARPORT, UTYPE_ENG_BAY, UTYPE_REFINERY,
})

MINERAL_TYPES = frozenset({
    UTYPE_MINERAL_FIELD, UTYPE_MINERAL_FIELD_2, UTYPE_MINERAL_FIELD_3,
})

GAS_SOURCE_TYPES = frozenset({UTYPE_VESPENE, UTYPE_REFINERY})

RESOURCE_TYPES = MINERAL_TYPES | GAS_SOURCE_TYPES

# Mapa edificio→tipo de unidad que puede entrenar
BUILDING_TRAINS = {
    UTYPE_CC:       UTYPE_SCV,
    UTYPE_BARRACKS: UTYPE_MARINE,
}
