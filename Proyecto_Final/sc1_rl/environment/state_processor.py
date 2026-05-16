"""
SEC 1 — Procesador de Estado
Convierte capturas de pantalla PIL en observaciones numpy apiladas (frame stack).
"""
import numpy as np
from PIL import Image


class StateProcessor:
    """
    Redimensiona y apila frames en escala de grises.
    Salida: array float32 de forma (stack_size, height, width), valores [0, 1].
    """

    def __init__(self, width: int = 128, height: int = 128, stack_size: int = 4):
        self.width = width
        self.height = height
        self.stack_size = stack_size
        self._frames: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        blank = np.zeros((self.height, self.width), dtype=np.float32)
        self._frames = [blank.copy() for _ in range(self.stack_size)]
        return self._stack()

    def process(self, screenshot: Image.Image) -> np.ndarray:
        gray = (
            screenshot
            .convert("L")
            .resize((self.width, self.height), Image.BILINEAR)
        )
        frame = np.array(gray, dtype=np.float32) / 255.0
        self._frames.pop(0)
        self._frames.append(frame)
        return self._stack()

    def get_stacked(self) -> np.ndarray:
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)   # (stack_size, H, W)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.stack_size, self.height, self.width)
