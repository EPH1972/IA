"""
TorchCraft ZMQ client — pure Python, no C extension required.

Uses pyzmq + flatbuffers to speak the TorchCraft wire protocol directly
with BWEnv.dll running inside the Win7 VM.

ZMQ protocol:  REQ socket, strict send→recv alternation
FlatBuffers:   Message{uid, msg_type, inner_table}  (no size prefix)
Handshake:     HandshakeClient{protocol=30} → HandshakeServer
Game loop:     Commands{[…]} → StateUpdate{Frame|FrameDiff}
"""
import logging
import time

import zmq

from sc1_rl.torchcraft.proto import (
    Any as MsgType,
    GameState,
    ServerHandshake,
    decode_handshake_server,
    decode_message,
    decode_state_update,
    encode_commands,
    encode_handshake,
    make_uid,
)

logger = logging.getLogger("sc1_rl")


class TorchCraftClient:
    """
    Wrapper around the TorchCraft ZMQ protocol.

    Usage:
        client = TorchCraftClient()
        if client.connect():
            client.send([])          # NOOP to advance one frame
            client.recv()
            state = client.state     # GameState
            client.close()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        connect_timeout: float = 120.0,
    ):
        self.host            = host
        self.port            = port
        self.connect_timeout = connect_timeout
        self._uid: str       = make_uid()
        self._ctx: zmq.Context | None  = None
        self._sock: zmq.Socket | None  = None
        self._hs:  ServerHandshake | None = None
        self.state: GameState | None   = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Connect to the TorchCraft server and complete the handshake.

        Sends HandshakeClient once and waits up to connect_timeout seconds
        for BWEnv.dll to reply.  A single persistent socket avoids breaking
        the ZMQ REQ/REP state machine while the game is still loading.
        """
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.SNDTIMEO, 10_000)
        self._sock.setsockopt(zmq.RCVTIMEO, int(self.connect_timeout * 1000))
        self._sock.setsockopt(zmq.LINGER,   0)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

        try:
            self._sock.send(encode_handshake(self._uid))
            logger.info(
                "HandshakeClient enviado — esperando respuesta del servidor "
                "(timeout=%.0f s)...", self.connect_timeout
            )
            raw = self._sock.recv()

            msg_type, inner = decode_message(raw)
            if msg_type == MsgType.HandshakeServer:
                self._hs = decode_handshake_server(inner)
                logger.info(
                    "TorchCraft conectado — map=%s player=%d neutral=%d lag=%d",
                    self._hs.map_name, self._hs.player_id,
                    self._hs.neutral_id, self._hs.lag_frames,
                )
                self.state            = GameState()
                self.state.player_id  = self._hs.player_id
                self.state.neutral_id = self._hs.neutral_id
                self.state.map_size   = (self._hs.map_w, self._hs.map_h)
                return True

            logger.warning("TC connect: msg_type inesperado=%d", msg_type)

        except zmq.ZMQError as exc:
            logger.error(
                "No se pudo conectar a TorchCraft en %s:%d — %s",
                self.host, self.port, exc,
            )

        if self._sock is not None:
            self._sock.close()
            self._sock = None
        return False

    def reconnect(self) -> bool:
        """
        Re-handshake sobre el socket existente después de EndGame.

        Con auto_restart=ON en bwapi.ini, BWEnv reinicia la partida
        y espera un nuevo HandshakeClient en el mismo socket ZMQ REP.
        Llamar a send([]) en ese estado causa un deadlock; hay que
        volver a hacer el handshake completo.
        """
        if self._sock is None:
            return False
        try:
            self._sock.send(encode_handshake(self._uid))
            logger.info("Reconnect: HandshakeClient enviado (timeout=%.0f s)...",
                        self.connect_timeout)
            raw = self._sock.recv()
            msg_type, inner = decode_message(raw)
            if msg_type == MsgType.HandshakeServer:
                self._hs              = decode_handshake_server(inner)
                self.state            = GameState()
                self.state.player_id  = self._hs.player_id
                self.state.neutral_id = self._hs.neutral_id
                self.state.map_size   = (self._hs.map_w, self._hs.map_h)
                logger.info(
                    "Reconnect OK — map=%s player=%d",
                    self._hs.map_name, self._hs.player_id,
                )
                return True
            logger.warning("Reconnect: msg_type inesperado=%d", msg_type)
        except zmq.ZMQError as exc:
            logger.error("Reconnect fallido: %s", exc)
        return False

    # ── Game-loop I/O ─────────────────────────────────────────────────────────

    def send(self, commands: list) -> bool:
        """
        Send a list of BWAPI commands and wait for the next frame state.

        commands: list of [cmd_code, arg1, arg2, …]   (all ints)
                  Pass [] for a no-op heartbeat.

        Returns True if the frame was received without errors.
        After a True return, self.state holds the updated GameState.
        """
        if self._sock is None:
            return False
        try:
            self._sock.send(encode_commands(self._uid, commands))
            raw = self._sock.recv()
            return self._process_state(raw)
        except zmq.ZMQError as exc:
            logger.warning("TorchCraft send/recv error: %s", exc)
            return False

    def recv(self) -> bool:
        """Alias kept for API compatibility — see send()."""
        return self.send([])

    def _process_state(self, raw: bytes) -> bool:
        msg_type, inner = decode_message(raw)

        if msg_type == MsgType.StateUpdate:
            hs = self._hs
            self.state = decode_state_update(
                inner,
                prev_state        = self.state,
                player_id         = hs.player_id if hs else 0,
                neutral_id        = hs.neutral_id if hs else 11,
                battle_frame_count = (self.state.battle_frame_count + 1)
                                     if self.state else 0,
            )
            return True

        if msg_type in (MsgType.EndGame, MsgType.PlayerLeft):
            logger.info("TorchCraft: game ended (msg_type=%d)", msg_type)
            if self.state is not None:
                self.state.game_ended = True
            return True

        if msg_type == MsgType.Error:
            logger.warning("TorchCraft server error message received")
            return False

        logger.debug("Unexpected msg_type=%d from server", msg_type)
        return False

    # ── State helpers ─────────────────────────────────────────────────────────

    def is_game_running(self) -> bool:
        return self.state is not None and not self.state.game_ended

    @property
    def player_id(self) -> int:
        if self._hs:
            return self._hs.player_id
        return self.state.player_id if self.state else 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
