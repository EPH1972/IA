"""
SEC 2 — Controlador de VM (alto nivel)
Interfaz única para SC1Env: captura pantalla + ejecuta acciones.
Traduce objetos Action (SEC 1) a comandos VBoxManage concretos.
"""
import time

from PIL import Image

from sc1_rl.environment.action_space import Action, ActionType
from sc1_rl.vm.connector import VMConnector
from sc1_rl.vm.input import InputHandler
from sc1_rl.vm.screen import ScreenCapture

_CAMERA_KEYS = {
    ActionType.CAMERA_UP:    "up",
    ActionType.CAMERA_DOWN:  "down",
    ActionType.CAMERA_LEFT:  "left",
    ActionType.CAMERA_RIGHT: "right",
}


class VMController:
    """
    Punto de entrada para toda comunicación con la VM.
    SC1Env sólo llama a este objeto; no conoce VBoxManage ni scan codes.
    """

    def __init__(
        self,
        vm_name: str,
        vm_width: int = 640,
        vm_height: int = 480,
        screenshot_dir: str | None = None,
    ):
        self.vm_name = vm_name
        self.vm_width = vm_width
        self.vm_height = vm_height

        self.connector = VMConnector(vm_name)
        self.screen = ScreenCapture(vm_name, screenshot_dir)
        self.input = InputHandler(vm_name)

    # ── Conexión ──────────────────────────────────────────────────────────────

    def connect(self, start_if_stopped: bool = True) -> bool:
        """Asegura que la VM esté corriendo. Devuelve True si está lista."""
        if self.connector.is_running():
            return True
        if not start_if_stopped:
            return False
        ok = self.connector.start()
        if not ok:
            return False
        return self.connector.wait_until_running()

    # ── Observación ───────────────────────────────────────────────────────────

    def capture_screen(self) -> Image.Image | None:
        return self.screen.capture()

    def is_game_running(self) -> bool:
        return self.connector.is_running()

    # ── Acciones ──────────────────────────────────────────────────────────────

    def execute_action(self, action: Action):
        """
        Recibe un Action decodificado y lo traduce a comandos de VM.
        Llamado exclusivamente desde SC1Env.step().
        """
        t = action.action_type

        if t == ActionType.NOOP:
            return

        if t in _CAMERA_KEYS:
            self.input.press_key(_CAMERA_KEYS[t])

        elif t == ActionType.LEFT_CLICK:
            px = int(action.x * self.vm_width)
            py = int(action.y * self.vm_height)
            self.input.mouse_click(px, py, "left")

        elif t == ActionType.RIGHT_CLICK:
            px = int(action.x * self.vm_width)
            py = int(action.y * self.vm_height)
            self.input.mouse_click(px, py, "right")

        elif t == ActionType.KEYBOARD:
            self.input.press_key(action.key)

        elif t == ActionType.CONTROL_GROUP_SELECT:
            self.input.press_key(str(action.group))

        elif t == ActionType.CONTROL_GROUP_SET:
            self.input.press_ctrl_key(str(action.group))
