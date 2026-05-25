"""
SEC 3 — Módulo TorchCraft
Reemplaza la capa VBoxManage+screenshot con estado estructurado BWAPI via ZMQ.

Requiere:
    pip install torchcraft
    BWAPI 4.4 + TorchCraft.dll instalados en la VM (ver setup_bwapi.md)
"""
from sc1_rl.torchcraft.client import TorchCraftClient
from sc1_rl.torchcraft.action_space import N_ACTIONS_TC, decode_tc_action, TCAction, TCActionType
from sc1_rl.torchcraft.state_encoder import StateEncoder, OBS_SIZE_TC

__all__ = [
    "TorchCraftClient",
    "N_ACTIONS_TC",
    "decode_tc_action",
    "TCAction",
    "TCActionType",
    "StateEncoder",
    "OBS_SIZE_TC",
]
