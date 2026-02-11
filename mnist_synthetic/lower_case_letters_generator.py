import math
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from mnist_synthetic.config import GeneratorConfig
from mnist_synthetic.utils.drawing import Drawing
from mnist_synthetic.location import IntLocation, RangeLocation, RangeOnPrevLocation, RelatedLocation, \
    RangeRelatedLocation, RangeOnPrevRelatedLocation


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, include: list[str] | None = None) -> tuple[NDArray[np.uint8], str]:
        raise NotImplementedError()


class LowerCaseLettersGenerator(BaseGenerator):
    ALLOWS_SYMBOLS: ClassVar[tuple[str, ...]] = ('а', 'б', 'в', 'г', 'д')  # TODO: Дополнить

    def __init__(self, seed: int | None = None, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()

        self.seed_rng = np.random.default_rng(seed=seed)
        self.draw: Drawing = Drawing(self.seed_rng, self.config.draw_color, self.config.draw_thickness)

    def _init_image(self):
        img = np.zeros((self.config.height, self.config.height), dtype=np.uint8)
        if self.config.channels > 1:
            img = img[None,]
            img = np.repeat(img, self.config.channels, axis=0)

        return img

    def generate(self, include: list[str] | None = None) -> tuple[NDArray[np.uint8], str]:
        if include is None:
            include = self.ALLOWS_SYMBOLS

        label: str = self.seed_rng.choice(include, size=1)[0]
        img = getattr(self, f'generate_{label}')()

        return img, label

    def generate_а(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Построение "Овала" для буквы
        center_x = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.15, 0.19) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.20, 0.24) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(25, 40))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )
        # 2. Построение наклонной палки справа
        ellipse_top_y = center_y - height // 2
        ellipse_bottom_y = center_y + height // 2
        ellipse_right_x = center_x + width // 2

        # Верхняя часть
        stick_top_y = ellipse_top_y - int(self.draw.seed_rng.uniform(3, 5))
        stick_top_x = ellipse_right_x + int(self.draw.seed_rng.uniform(3, 5))

        # Нижняя
        stick_bottom_y = ellipse_bottom_y + int(self.draw.seed_rng.uniform(1, 3))
        stick_bottom_x = ellipse_right_x + int(self.draw.seed_rng.uniform(2, 5))

        self.draw.draw_line(
            img,
            left_location=IntLocation(stick_top_x),
            top_location=IntLocation(stick_top_y),
            right_location=IntLocation(stick_bottom_x),
            bottom_location=IntLocation(stick_bottom_y),
        )

        # 3. Построение хвостика
        start_x = stick_bottom_x
        start_y = stick_bottom_y - 2

        width = int(self.draw.seed_rng.uniform(4, 6))  # вправо
        height = int(self.draw.seed_rng.uniform(3, 4))  # вниз

        # Центр таков, что точка 90° = (start_x, start_y)
        tail_center_x = start_x
        tail_center_y = start_y + height - 3

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(tail_center_x),
            center_y_range=IntLocation(tail_center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(90),
            end_angle_location=IntLocation(0),
        )

        return img

    def generate_б(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Построение большой наклонной палки
        stick_top_x, stick_top_y, stick_bottom_x, stick_bottom_y = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.55, 0.65, max_size=self.config.width),
            top_location=RangeRelatedLocation(0.15, 0.25, max_size=self.config.height),
            right_location=RangeRelatedLocation(0.45, 0.55, max_size=self.config.width),
            bottom_location=RangeRelatedLocation(0.60, 0.75, max_size=self.config.height),
        )

        # 2. Построение маленькой дуги справа, сшитой с палкой
        radius = int(self.draw.seed_rng.integers(2, 5))

        center_x = stick_top_x + radius
        center_y = stick_top_y

        # Рисуем дугу, выпуклую вправо
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(radius),
            height_location=IntLocation(radius),
            angle_location=IntLocation(90),
            start_angle_location=IntLocation(90),
            end_angle_location=IntLocation(270),
        )

        # 3. Нижняя петля - аккуратно сшита с палкой, чуть шире и левее
        dx = stick_bottom_x - stick_top_x
        dy = stick_bottom_y - stick_top_y
        stick_length = math.hypot(dx, dy)

        if stick_length < 5:
            return img

        ux = dx / stick_length
        uy = dy / stick_length

        # Высота петли: 40–45% от палки, минус 1–2 пикселя
        ellipse_full_height = max(2, int(self.draw.seed_rng.uniform(0.40, 0.45) * stick_length) - 1)
        ellipse_radius_y = ellipse_full_height // 2
        ellipse_radius_x = int(self.draw.seed_rng.uniform(8, 11))

        # Смещение влево: только 1–2 пикселя перпендикулярно
        offset_left = int(self.draw.seed_rng.uniform(2, 4))

        # Центр эллипса:
        # сначала идём вверх от нижней точки на radius_y вдоль палки
        base_center_x = stick_bottom_x - ux * ellipse_radius_y
        base_center_y = stick_bottom_y - uy * ellipse_radius_y

        # затем смещаемся влево (перпендикулярно)
        center_x = int(base_center_x - uy * offset_left)
        center_y = int(base_center_y + ux * offset_left)

        angle_deg = math.degrees(math.atan2(dy, dx))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(ellipse_radius_x - 2),
            height_location=IntLocation(ellipse_radius_y + 2),
            angle_location=IntLocation(int(angle_deg)),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        return img


    def generate_в(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Построение нижнего "овала" для буквы
        center_x = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.7, 0.80) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.15, 0.19) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.20, 0.24) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(15, 30))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        # Построение верхнего, более длинного овала
        center_x = int(self.draw.seed_rng.uniform(0.4, 0.45) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.32, 0.37) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.1, 0.15) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.32, 0.40) * self.config.height)
        angle = angle + 3 #int(self.draw.seed_rng.uniform(15, 40))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        return img


    def generate_г(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Построение большой наклонной палки
        stick_top_x, stick_top_y, stick_bottom_x, stick_bottom_y = self.draw.draw_line(
            img,
            left_location=RangeRelatedLocation(0.65, 0.65, max_size=self.config.width),
            top_location=RangeRelatedLocation(0.25, 0.35, max_size=self.config.height),
            right_location=RangeRelatedLocation(0.5, 0.55, max_size=self.config.width),
            bottom_location=RangeRelatedLocation(0.70, 0.85, max_size=self.config.height),
        )

        # 2. Построение маленькой дуги слева сверху, сшитой с палкой
        radius = int(self.draw.seed_rng.integers(3, 6))

        center_x = stick_top_x - radius
        center_y = stick_top_y

        # Рисуем дугу, выпуклую влево
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(radius),
            height_location=IntLocation(radius),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(-90),
        )

        # 3. Построение маленькой дуги справа снизу, сшитой с палкой
        radius = int(self.draw.seed_rng.integers(4, 7))

        center_x = stick_bottom_x + 1
        center_y = stick_bottom_y - radius + 2

        # Рисуем дугу, выпуклую вправо
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(radius),
            height_location=IntLocation(radius),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(-15),
            end_angle_location=IntLocation(90),
        )



        return img


    # TODO: доделать
    def generate_д(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Построение "Овала" для буквы
        center_x = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.11, 0.15) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.16, 0.20) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(25, 40))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )
        # 2. Построение наклонной палки справа
        ellipse_top_y = center_y - height // 2
        ellipse_bottom_y = center_y + height // 2
        ellipse_right_x = center_x + width // 2

        # Верхняя часть
        stick_top_y = ellipse_top_y - int(self.draw.seed_rng.uniform(3, 5))
        stick_top_x = ellipse_right_x + int(self.draw.seed_rng.uniform(3, 5))

        # Нижняя
        stick_bottom_y = ellipse_bottom_y + int(self.draw.seed_rng.uniform(4, 7))
        stick_bottom_x = ellipse_right_x + int(self.draw.seed_rng.uniform(1, 3))

        self.draw.draw_line(
            img,
            left_location=IntLocation(stick_top_x),
            top_location=IntLocation(stick_top_y),
            right_location=IntLocation(stick_bottom_x),
            bottom_location=IntLocation(stick_bottom_y),
        )




        return img
