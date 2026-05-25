"""
Pure-Python TorchCraft FlatBuffer protocol.

Implements encode/decode for the TorchCraft wire format without the torchcraft
C extension.  Uses only pyzmq + flatbuffers (both already installed).

Wire format:
  ZMQ REQ socket, strict send→recv alternation
  Each ZMQ message = one FlatBuffers Message{uid, msg_type, msg} table
  No size prefix (standard FlatBuffers framing, ZMQ handles boundaries)

Handshake:
  client → HandshakeClient{protocol=30}   wrapped in Message{msg_type=1}
  server → HandshakeServer{…}             wrapped in Message{msg_type=3}

Game loop (per BWAPI frame):
  client → Commands{[Command,…]}          wrapped in Message{msg_type=2}
  server → StateUpdate{Frame|FrameDiff}   wrapped in Message{msg_type=4}
"""
import random
import string
import struct

import flatbuffers
import flatbuffers.table as _T
import flatbuffers.number_types as _N

# ── Any enum ─────────────────────────────────────────────────────────────────
class Any:
    NONE            = 0
    HandshakeClient = 1
    Commands        = 2
    HandshakeServer = 3
    StateUpdate     = 4
    PlayerLeft      = 5
    EndGame         = 6
    Error           = 7

# ── FrameOrFrameDiff enum ────────────────────────────────────────────────────
class FrameType:
    NONE      = 0
    Frame     = 1
    FrameDiff = 2

PROTOCOL_VERSION = 30


def make_uid(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def _int_vector(b: flatbuffers.Builder, values: list) -> int:
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependInt32(v)
    return b.EndVector()


def _command(b: flatbuffers.Builder, code: int, args: list) -> int:
    """Build Command{code, args, str} table. VT: code=4, args=6, str=8."""
    s_off   = b.CreateString("")
    arg_off = _int_vector(b, args)
    b.StartObject(3)
    b.PrependInt32Slot(0, code, 0)
    b.PrependUOffsetTRelativeSlot(1, arg_off, 0)
    b.PrependUOffsetTRelativeSlot(2, s_off, 0)
    return b.EndObject()


def _message(b: flatbuffers.Builder, uid_off: int, msg_type: int, inner_off: int) -> int:
    """Wrap inner table in Message{msg:Any, uid:string}.
    Schema order: msg (union → VT4=msg_type ubyte, VT6=msg offset), uid (VT8)."""
    b.StartObject(3)
    b.PrependUint8Slot(0, msg_type, 0)              # slot 0 → VT4: msg_type
    b.PrependUOffsetTRelativeSlot(1, inner_off, 0)  # slot 1 → VT6: msg
    b.PrependUOffsetTRelativeSlot(2, uid_off, 0)    # slot 2 → VT8: uid
    return b.EndObject()


def encode_handshake(uid: str) -> bytes:
    """
    Build Message{HandshakeClient{protocol=30}}.
    HandshakeClient VT: protocol=4, map=6, window_size=8, window_pos=10, micro_mode=12
    """
    b = flatbuffers.Builder(256)
    uid_off = b.CreateString(uid)
    map_off = b.CreateString("")

    b.StartObject(5)
    b.PrependInt32Slot(0, PROTOCOL_VERSION, 0)   # protocol
    b.PrependUOffsetTRelativeSlot(1, map_off, 0) # map
    # window_size (field 2) and window_pos (field 3): omitted → defaults
    b.PrependBoolSlot(4, False, False)            # micro_mode
    hc_off = b.EndObject()

    msg_off = _message(b, uid_off, Any.HandshakeClient, hc_off)
    b.Finish(msg_off)
    return bytes(b.Output())


def encode_commands(uid: str, commands: list = None) -> bytes:
    """
    Build Message{Commands{[Command]}}.
    commands: list of [code, arg1, arg2, ...] ints
    Commands VT: commands=4
    """
    if commands is None:
        commands = []
    b = flatbuffers.Builder(256 + 64 * max(len(commands), 1))
    uid_off = b.CreateString(uid)

    cmd_offs = []
    for cmd in commands:
        code = int(cmd[0])
        args = [int(a) for a in cmd[1:] if isinstance(a, (int, float))]
        cmd_offs.append(_command(b, code, args))

    b.StartVector(4, len(cmd_offs), 4)
    for off in reversed(cmd_offs):
        b.PrependUOffsetTRelative(off)
    vec_off = b.EndVector()

    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, vec_off, 0)   # commands
    cmds_off = b.EndObject()

    msg_off = _message(b, uid_off, Any.Commands, cmds_off)
    b.Finish(msg_off)
    return bytes(b.Output())


# ─────────────────────────────────────────────────────────────────────────────
# DECODING helpers
# ─────────────────────────────────────────────────────────────────────────────

def _root(raw: bytes) -> _T.Table:
    buf = bytearray(raw)
    pos = struct.unpack_from("<I", buf, 0)[0]
    return _T.Table(buf, pos)


def _str(tab: _T.Table, vt: int) -> str:
    o = tab.Offset(vt)
    return tab.String(tab.Pos + o).decode("utf-8", errors="replace") if o else ""


def _u8(tab, vt, d=0):
    return tab.GetSlot(vt, d, _N.Uint8Flags)

def _i32(tab, vt, d=0):
    return tab.GetSlot(vt, d, _N.Int32Flags)

def _u32(tab, vt, d=0):
    return tab.GetSlot(vt, d, _N.Uint32Flags)

def _i64(tab, vt, d=0):
    return tab.GetSlot(vt, d, _N.Int64Flags)

def _bool(tab, vt, d=False):
    return bool(tab.GetSlot(vt, d, _N.BoolFlags))

def _inner_table(tab: _T.Table, vt: int) -> "_T.Table | None":
    o = tab.Offset(vt)
    if not o:
        return None
    inner = _T.Table(tab.Bytes, 0)
    tab.Union(inner, o)
    return inner

def _ref_table(tab: _T.Table, vt: int) -> "_T.Table | None":
    """Read a direct table-reference field (not a union, just an offset to a table)."""
    o = tab.Offset(vt)
    if not o:
        return None
    return _T.Table(tab.Bytes, tab.Indirect(tab.Pos + o))

def _vec_len(tab: _T.Table, vt: int) -> int:
    o = tab.Offset(vt)
    return tab.VectorLen(o) if o else 0

def _vec_table(tab: _T.Table, vt: int, idx: int) -> "_T.Table | None":
    """Return Table for the idx-th element of a vector-of-tables field."""
    o = tab.Offset(vt)
    if not o:
        return None
    vec_start = tab.Vector(o)
    ref_pos   = vec_start + idx * 4
    inner_pos = tab.Indirect(ref_pos)
    return _T.Table(tab.Bytes, inner_pos)

def _vec_i32(tab: _T.Table, vt: int) -> list:
    """Return list of int32 from a vector-of-scalars field."""
    o = tab.Offset(vt)
    if not o:
        return []
    n         = tab.VectorLen(o)
    vec_start = tab.Vector(o)
    return [struct.unpack_from("<i", tab.Bytes, vec_start + i * 4)[0] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# DECODE Message
# ─────────────────────────────────────────────────────────────────────────────

def decode_message(raw: bytes):
    """
    Returns (msg_type: int, inner_tab: Table | None)
    msg_type → Any.* constants
    inner_tab → the HandshakeServer or StateUpdate table inside the same buffer
    Schema: Message{msg:Any (VT4=msg_type, VT6=msg), uid:string (VT8)}
    """
    msg = _root(raw)
    msg_type  = _u8(msg, 4)          # VT4: msg_type (union discriminant)
    inner_tab = _inner_table(msg, 6) # VT6: msg (union table)
    return msg_type, inner_tab


# ─────────────────────────────────────────────────────────────────────────────
# DECODE HandshakeServer
# ─────────────────────────────────────────────────────────────────────────────

class ServerHandshake:
    """Result of parsing a HandshakeServer message."""
    __slots__ = ("lag_frames", "map_w", "map_h", "map_name",
                 "player_id", "neutral_id", "battle_frame_count")

    def __init__(self):
        self.lag_frames         = 0
        self.map_w              = 512
        self.map_h              = 512
        self.map_name           = ""
        self.player_id          = 0
        self.neutral_id         = 11
        self.battle_frame_count = 0


def decode_handshake_server(tab: _T.Table) -> ServerHandshake:
    """
    HandshakeServer VT:
      lag_frames=4, map_size=6(Vec2 struct), ground_height_data=8, walkable_data=10,
      map_name=12, is_replay=14, player_id=16, neutral_id=18, battle_frame_count=20,
      buildable_data=22, start_locations=24, players=26
    """
    hs = ServerHandshake()
    if tab is None:
        return hs

    hs.lag_frames         = _i32(tab, 4)
    hs.map_name           = _str(tab, 12)
    hs.player_id          = _i32(tab, 16)
    hs.neutral_id         = _i32(tab, 18)
    hs.battle_frame_count = _i32(tab, 20)

    # Vec2 struct for map_size — inline struct, 8 bytes (x=int32, y=int32)
    o = tab.Offset(6)
    if o:
        base = tab.Pos + o
        hs.map_w = struct.unpack_from("<i", tab.Bytes, base)[0]
        hs.map_h = struct.unpack_from("<i", tab.Bytes, base + 4)[0]

    return hs


# ─────────────────────────────────────────────────────────────────────────────
# DECODE StateUpdate → GameState
# ─────────────────────────────────────────────────────────────────────────────

class UnitState:
    """Minimal unit representation compatible with state_encoder.py and command_executor.py."""
    __slots__ = (
        "id", "type", "x", "y", "health", "max_health",
        "shield", "max_shield", "energy",
        "idle", "gathering_minerals", "gathering_gas",
        "attacking", "training", "constructing",
        "resources",
    )

    def __init__(self):
        self.id               = 0
        self.type             = 0
        self.x                = 0
        self.y                = 0
        self.health           = 1
        self.max_health       = 1
        self.shield           = 0
        self.max_shield       = 0
        self.energy           = 0
        self.idle             = True
        self.gathering_minerals = False
        self.gathering_gas    = False
        self.attacking        = False
        self.training         = False
        self.constructing     = False
        self.resources        = 0


class ResourceState:
    __slots__ = ("ore", "gas", "used_psi", "total_psi")

    def __init__(self):
        self.ore       = 0
        self.gas       = 0
        self.used_psi  = 0
        self.total_psi = 8


class GameState:
    """
    State object returned by the client each frame.
    Compatible with state_encoder.py and command_executor.py.

    units[player_id] → {unit_id: UnitState}
    resources[player_id] → ResourceState
    neutral_units → {unit_id: UnitState}  (minerals/geysers)
    """
    __slots__ = (
        "player_id", "neutral_id",
        "map_size",          # (walk_w, walk_h)
        "units",             # {pid: {uid: UnitState}}
        "resources",         # {pid: ResourceState}
        "neutral_units",     # {uid: UnitState}
        "game_ended",
        "battle_frame_count",
        "reward",
    )

    def __init__(self):
        self.player_id          = 0
        self.neutral_id         = 11
        self.map_size           = (64, 64)
        self.units              = {}
        self.resources          = {}
        self.neutral_units      = {}
        self.game_ended         = False
        self.battle_frame_count = 0
        self.reward             = 0


# Unit FlatBuffer VTable offsets (37 fields starting at VT 4):
# id=4, x=6, y=8, health=10, max_health=12, shield=14, max_shield=16, energy=18,
# maxCD=20, groundCD=22, airCD=24, flags=26(i64), visible=28, type=30, armor=32,
# shieldArmor=34, size=36, pixel_x=38, pixel_y=40, pixel_size_x=42, pixel_size_y=44,
# groundATK=46, airATK=48, groundDmgType=50, airDmgType=52, groundRange=54, airRange=56,
# orders=58, command=60(struct), velocityX=62(f64), velocityY=64(f64),
# playerId=66, resources=68, buildTechUpgradeType=70, remainingBuildTrainTime=72,
# remainingUpgradeResearchTime=74, spellCD=76, associatedUnit=78, associatedCount=80

# BWAPI unit flags bit positions (from BWAPI source):
_FLAG_IDLE             = 1 << 0   # bit 0 in BroodwarImpl / UnitImpl isIdle
_FLAG_GATHERING_MIN    = 1 << 9   # isGatheringMinerals
_FLAG_GATHERING_GAS    = 1 << 10  # isGatheringGas
_FLAG_ATTACKING        = 1 << 5   # isAttacking
_FLAG_TRAINING         = 1 << 18  # isTraining
_FLAG_CONSTRUCTING     = 1 << 7   # isConstructing


def _decode_unit(u_tab: _T.Table) -> UnitState:
    u = UnitState()
    u.id         = _i32(u_tab, 4)
    u.x          = _i32(u_tab, 6)
    u.y          = _i32(u_tab, 8)
    u.health     = _i32(u_tab, 10)
    u.max_health = _i32(u_tab, 12)
    u.shield     = _i32(u_tab, 14)
    u.max_shield = _i32(u_tab, 16)
    u.energy     = _i32(u_tab, 18)
    u.type       = _i32(u_tab, 30)
    u.resources  = _i32(u_tab, 68)  # resource amount (minerals on patch)

    flags = _i64(u_tab, 26)
    u.idle              = bool(flags & _FLAG_IDLE)
    u.gathering_minerals = bool(flags & _FLAG_GATHERING_MIN)
    u.gathering_gas     = bool(flags & _FLAG_GATHERING_GAS)
    u.attacking         = bool(flags & _FLAG_ATTACKING)
    u.training          = bool(flags & _FLAG_TRAINING)
    u.constructing      = bool(flags & _FLAG_CONSTRUCTING)

    # remainingBuildTrainTime > 0 is a reliable training indicator
    if _i32(u_tab, 72) > 0:
        u.training = True

    return u


def _decode_frame(frame_tab: _T.Table, state: GameState, player_id: int, neutral_id: int):
    """
    Frame VT: units=4, actions=6, resources=8, bullets=10, creep_map=12,
              width=14(u32), height=16(u32), reward=18, is_terminal=20
    """
    state.reward     = _i32(frame_tab, 18)
    state.game_ended = _bool(frame_tab, 20)

    # ── Resources ──────────────────────────────────────────────────────────
    n_res = _vec_len(frame_tab, 8)
    for i in range(n_res):
        rp = _vec_table(frame_tab, 8, i)
        if rp is None:
            continue
        # ResourcesOfPlayer VT: playerId=4, resources=6
        pid = _i32(rp, 4)
        res_tab = _ref_table(rp, 6)
        rs = ResourceState()
        if res_tab is not None:
            # Resources VT: ore=4, gas=6, used_psi=8, total_psi=10
            rs.ore       = _i32(res_tab, 4)
            rs.gas       = _i32(res_tab, 6)
            rs.used_psi  = _i32(res_tab, 8)
            rs.total_psi = _i32(res_tab, 10) or 8
        state.resources[pid] = rs

    # ── Units ───────────────────────────────────────────────────────────────
    n_uop = _vec_len(frame_tab, 4)
    for i in range(n_uop):
        uop = _vec_table(frame_tab, 4, i)
        if uop is None:
            continue
        # UnitsOfPlayer VT: playerId=4, units=6
        pid    = _i32(uop, 4)
        n_u    = _vec_len(uop, 6)
        unit_d = {}
        for j in range(n_u):
            u_tab = _vec_table(uop, 6, j)
            if u_tab is None:
                continue
            u = _decode_unit(u_tab)
            unit_d[u.id] = u
        state.units[pid] = unit_d
        if pid == neutral_id or pid >= 11:
            state.neutral_units = unit_d


def _decode_framediff(diff_tab: _T.Table, state: GameState, player_id: int, neutral_id: int):
    """
    FrameDiff VT:
      pids=4([int]), unitDiffContainers=6([UnitDiffContainer]), actions=8,
      resources=10([ResourcesOfPlayer]), bullets=12,
      reward=14(int), is_terminal=16(bool)
    """
    state.reward     = _i32(diff_tab, 14)
    state.game_ended = _bool(diff_tab, 16)

    # Resources — same format as Frame
    n_res = _vec_len(diff_tab, 10)
    for i in range(n_res):
        rp = _vec_table(diff_tab, 10, i)
        if rp is None:
            continue
        pid     = _i32(rp, 4)
        res_tab = _ref_table(rp, 6)
        rs = ResourceState()
        if res_tab is not None:
            rs.ore       = _i32(res_tab, 4)
            rs.gas       = _i32(res_tab, 6)
            rs.used_psi  = _i32(res_tab, 8)
            rs.total_psi = _i32(res_tab, 10) or 8
        state.resources[pid] = rs

    # UnitDiffContainers — apply position/health deltas
    # UnitDiffContainer VT: pid=4, unitDiffs=6([UnitDiff])
    # UnitDiff VT: id=4, var_ids=6([int]), var_diffs=8([int])
    # var_ids: 0-based index into Unit fields (id=0,x=1,y=2,health=3,…,flags=11,type=13)
    _UNIT_FIELD_X      = 1
    _UNIT_FIELD_Y      = 2
    _UNIT_FIELD_HEALTH = 3
    _UNIT_FIELD_FLAGS  = 11

    n_containers = _vec_len(diff_tab, 6)
    for ci in range(n_containers):
        container = _vec_table(diff_tab, 6, ci)
        if container is None:
            continue
        pid     = _i32(container, 4)
        pid_map = state.units.get(pid, {})
        n_diffs = _vec_len(container, 6)
        for di in range(n_diffs):
            udiff = _vec_table(container, 6, di)
            if udiff is None:
                continue
            uid     = _i32(udiff, 4)
            var_ids  = _vec_i32(udiff, 6)
            var_vals = _vec_i32(udiff, 8)
            unit = pid_map.get(uid)
            if unit is None:
                continue
            for fi, fv in zip(var_ids, var_vals):
                if fi == _UNIT_FIELD_X:
                    unit.x = fv
                elif fi == _UNIT_FIELD_Y:
                    unit.y = fv
                elif fi == _UNIT_FIELD_HEALTH:
                    unit.health = fv
                elif fi == _UNIT_FIELD_FLAGS:
                    unit.idle       = bool(fv & _FLAG_IDLE)
                    unit.attacking  = bool(fv & _FLAG_ATTACKING)
                    unit.training   = bool(fv & _FLAG_TRAINING)
        state.units[pid] = pid_map


def decode_state_update(tab: _T.Table, prev_state: "GameState | None" = None,
                        player_id: int = 0, neutral_id: int = 11,
                        battle_frame_count: int = 0) -> GameState:
    """
    StateUpdate VT:
      data_type=4(ubyte), data=6(union→Frame/FrameDiff),
      deaths=8([int]), frame_from_bwapi=10, battle_frame_count=12, ...
    """
    state = GameState()
    if prev_state is not None:
        state.player_id     = prev_state.player_id
        state.neutral_id    = prev_state.neutral_id
        state.map_size      = prev_state.map_size
        state.units         = {pid: dict(u) for pid, u in prev_state.units.items()}
        state.resources     = dict(prev_state.resources)
        state.neutral_units = prev_state.neutral_units

    if tab is None:
        state.game_ended = True
        return state

    state.player_id          = player_id
    state.neutral_id         = neutral_id
    state.battle_frame_count = _i32(tab, 12) or battle_frame_count

    # Eliminar unidades muertas de TODOS los dicts de jugadores
    dead_ids = _vec_i32(tab, 8)
    if dead_ids:
        for pid_units in state.units.values():
            for uid in dead_ids:
                pid_units.pop(uid, None)

    data_type = _u8(tab, 4)
    data_tab  = _inner_table(tab, 6)

    if data_type == FrameType.Frame and data_tab is not None:
        _decode_frame(data_tab, state, player_id, neutral_id)
    elif data_type == FrameType.FrameDiff and data_tab is not None:
        _decode_framediff(data_tab, state, player_id, neutral_id)

    return state
