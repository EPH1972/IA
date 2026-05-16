"""
SEC 2 — Manejador de Input
Envía teclado y ratón a la VM mediante VBoxManage controlvm.

Teclado: scan codes AT (hex) vía keyboardputscancode
Ratón:   coordenadas absolutas vía 'mouse abs x y dz dw buttons'
         buttons: 0=ninguno  1=izquierdo  2=derecho  4=medio
"""
import subprocess
import time


# ── Scan codes AT (press + release) para teclas de StarCraft ─────────────────
SCAN_CODES: dict[str, list[str]] = {
    # Letras
    "a": ["1e", "9e"], "b": ["30", "b0"], "c": ["2e", "ae"], "d": ["20", "a0"],
    "e": ["12", "92"], "f": ["21", "a1"], "g": ["22", "a2"], "h": ["23", "a3"],
    "i": ["17", "97"], "j": ["24", "a4"], "k": ["25", "a5"], "l": ["26", "a6"],
    "m": ["32", "b2"], "n": ["31", "b1"], "o": ["18", "98"], "p": ["19", "99"],
    "r": ["13", "93"], "s": ["1f", "9f"], "t": ["14", "94"], "u": ["16", "96"],
    "v": ["2f", "af"], "w": ["11", "91"], "x": ["2d", "ad"], "y": ["15", "95"],
    "z": ["2c", "ac"],
    # Números
    "1": ["02", "82"], "2": ["03", "83"], "3": ["04", "84"],
    "4": ["05", "85"], "5": ["06", "86"], "6": ["07", "87"],
    "7": ["08", "88"], "8": ["09", "89"], "9": ["0a", "8a"], "0": ["0b", "8b"],
    # Especiales
    "escape": ["01", "81"],
    "enter":  ["1c", "9c"],
    "space":  ["39", "b9"],
    "tab":    ["0f", "8f"],
    # Teclas extendidas (prefijo E0)
    "up":        ["e0", "48", "e0", "c8"],
    "down":      ["e0", "50", "e0", "d0"],
    "left":      ["e0", "4b", "e0", "cb"],
    "right":     ["e0", "4d", "e0", "cd"],
    "delete":    ["e0", "53", "e0", "d3"],
    "home":      ["e0", "47", "e0", "c7"],
    "end":       ["e0", "4f", "e0", "cf"],
    "page_up":   ["e0", "49", "e0", "c9"],
    "page_down": ["e0", "51", "e0", "d1"],
    # Función
    "f1": ["3b", "bb"], "f2":  ["3c", "bc"], "f3":  ["3d", "bd"],
    "f4": ["3e", "be"], "f5":  ["3f", "bf"], "f6":  ["40", "c0"],
    "f7": ["41", "c1"], "f8":  ["42", "c2"], "f9":  ["43", "c3"],
    "f10": ["44", "c4"], "f11": ["57", "d7"], "f12": ["58", "d8"],
    # Modificadores (solo para uso interno en combinaciones)
    "ctrl":  ["1d", "9d"],
    "shift": ["2a", "aa"],
    "alt":   ["38", "b8"],
}

MOUSE_BUTTONS = {"none": 0, "left": 1, "right": 2, "middle": 4}


class InputHandler:
    """Envía eventos de teclado y ratón a la VM VirtualBox."""

    VBOXMANAGE = "VBoxManage"

    def __init__(self, vm_name: str, timeout: float = 3.0):
        self.vm_name = vm_name
        self.timeout = timeout

    # ── Interno ───────────────────────────────────────────────────────────────

    def _controlvm(self, *args: str) -> bool:
        try:
            r = subprocess.run(
                [self.VBOXMANAGE, "controlvm", self.vm_name, *args],
                capture_output=True,
                timeout=self.timeout,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False

    # ── Teclado ───────────────────────────────────────────────────────────────

    def press_key(self, key: str) -> bool:
        """Pulsa y suelta una tecla."""
        codes = SCAN_CODES.get(key.lower())
        if not codes:
            return False
        return self._controlvm("keyboardputscancode", *codes)

    def press_ctrl_key(self, key: str) -> bool:
        """Pulsa Ctrl + tecla."""
        ctrl_press = SCAN_CODES["ctrl"][:1]
        ctrl_release = SCAN_CODES["ctrl"][1:]
        key_codes = SCAN_CODES.get(key.lower(), [])
        if not key_codes:
            return False
        mid = len(key_codes) // 2
        return self._controlvm(
            "keyboardputscancode",
            *ctrl_press, *key_codes[:mid], *key_codes[mid:], *ctrl_release,
        )

    # ── Ratón ─────────────────────────────────────────────────────────────────

    def mouse_move(self, x: int, y: int) -> bool:
        """Mueve el cursor a coordenadas absolutas de la VM."""
        return self._controlvm("mouse", "abs", str(x), str(y), "0", "0", "0")

    def mouse_click(self, x: int, y: int, button: str = "left") -> bool:
        """Clic en coordenadas absolutas de la VM."""
        btn = str(MOUSE_BUTTONS.get(button, 0))
        ok = self._controlvm("mouse", "abs", str(x), str(y), "0", "0", "0")
        if not ok:
            return False
        self._controlvm("mouse", "abs", str(x), str(y), "0", "0", btn)
        return self._controlvm("mouse", "abs", str(x), str(y), "0", "0", "0")

    def mouse_double_click(self, x: int, y: int, button: str = "left") -> bool:
        self.mouse_click(x, y, button)
        time.sleep(0.05)
        return self.mouse_click(x, y, button)
