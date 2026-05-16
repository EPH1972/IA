"""
SEC 2 — Captura de Pantalla
Captura el framebuffer de la VM mediante VBoxManage screenshotpng.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


class ScreenCapture:
    """Captura frames de la VM a través de VBoxManage."""

    VBOXMANAGE = "VBoxManage"

    def __init__(self, vm_name: str, screenshot_dir: str | None = None):
        self.vm_name = vm_name
        self._dir = Path(screenshot_dir) if screenshot_dir else Path(tempfile.gettempdir())
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "sc1_frame.png"

    def capture(self, timeout: float = 5.0) -> Image.Image | None:
        """
        Toma un screenshot de la VM.
        Devuelve una imagen PIL (copia en memoria) o None si falla.
        """
        try:
            result = subprocess.run(
                [
                    self.VBOXMANAGE, "controlvm", self.vm_name,
                    "screenshotpng", str(self._path),
                ],
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode == 0 and self._path.exists():
                # .copy() cierra el archivo antes de que VBoxManage lo sobreescriba
                return Image.open(self._path).copy()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass
        return None
