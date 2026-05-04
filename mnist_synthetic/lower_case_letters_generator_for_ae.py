import math
import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar
from numpy.typing import NDArray

from mnist_synthetic.config import GeneratorConfig
from mnist_synthetic.utils.drawing import Drawing
from mnist_synthetic.location import IntLocation


class LowerCaseLettersGeneratorForAE(ABC):
    ALLOWS_SYMBOLS: ClassVar[tuple[str, ...]] = ('а', 'б', 'в')  # Дополняйте по мере необходимости

    def __init__(self, seed: int | None = None, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.seed_rng = np.random.default_rng(seed=seed)
        self.draw: Drawing = Drawing(self.seed_rng, self.config.draw_color, self.config.draw_thickness)
        self.last_params = np.zeros(3, dtype=np.float32)  # [наклон, ширина, высота]

    def _init_image(self):
        img = np.zeros((self.config.height, self.config.width), dtype=np.uint8)
        return img

    def _normalize(self, val, min_v, max_v):
        return float(np.clip((val - min_v) / (max_v - min_v), 0, 1))

    def generate_with_params(self) -> tuple[NDArray[np.uint8], str, NDArray[np.float32]]:
        label = self.seed_rng.choice(self.ALLOWS_SYMBOLS)
        img = getattr(self, f'generate_{label}')()
        return img, label, self.last_params

    def generate_а(self) -> NDArray[np.uint8]:
        img = self._init_image()
        # [наклон, ширина, высота]
        # Рисуем "а" как: овал из линий + палка справа
        c, r = 14, 14
        rw, rh = 6, 8

        # Рисуем контур овала (вместо заливки)
        # В OpenCV/Numpy проще всего это сделать через cv2.ellipse с thickness=2
        cv2.ellipse(img, (c, r), (rw, rh), 0, 0, 360, 255, 1)
        # Палка справа
        cv2.line(img, (c + rw, r - rh), (c + rw, r + rh), 255, 1)

        return img