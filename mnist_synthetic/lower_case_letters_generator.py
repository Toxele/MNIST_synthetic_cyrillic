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

    ALLOWS_SYMBOLS: ClassVar[tuple[str, ...]] = ('а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з',
                                                 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
                                                 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ',
                                                 'ъ', 'ы', 'ь', 'э', 'ю', 'я')

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
        stick_bottom_x = ellipse_right_x - int(self.draw.seed_rng.uniform(1, 2))

        self.draw.draw_line(
            img,
            left_location=IntLocation(stick_top_x),
            top_location=IntLocation(stick_top_y),
            right_location=IntLocation(stick_bottom_x),
            bottom_location=IntLocation(stick_bottom_y),
        )


        # 3. Закругление

        ellipse_center_x_location = stick_bottom_x  - int(self.draw.seed_rng.uniform(3, 5))
        ellipse_center_y_location = stick_bottom_y + int(self.draw.seed_rng.uniform(3, 5))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(ellipse_center_x_location),
            center_y_range=IntLocation(ellipse_center_y_location),
            width_location=IntLocation(int(width * 0.9)),
            height_location=IntLocation(int(height * 0.9)),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        return img

    def generate_е(self) -> NDArray[np.uint8]:
        img = self._init_image()

        center_x = int(self.draw.seed_rng.uniform(0.42, 0.48) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.48, 0.55) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.15, 0.19) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.24, 0.29) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(-3, 3))  # Slight natural tilt

        start_angle = int(self.draw.seed_rng.uniform(45, 55))
        end_angle = int(self.draw.seed_rng.uniform(305, 315))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(start_angle),
            end_angle_location=IntLocation(end_angle + 30),
        )


        bar_y = center_y + int(self.draw.seed_rng.uniform(-3, 1))

        bar_left_x = center_x - int(self.draw.seed_rng.uniform(4, 6))
        bar_right_x = bar_left_x + int(self.draw.seed_rng.uniform(7, 10))

        self.draw.draw_line(
            img,
            left_location=IntLocation(bar_left_x),
            top_location=IntLocation(bar_y),
            right_location=IntLocation(bar_right_x),
            bottom_location=IntLocation(bar_y),
        )

        return img

    def generate_ё(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Основная часть буквы "е" (полуовал с горизонтальной чертой)
        center_x = int(self.draw.seed_rng.uniform(0.42, 0.48) * self.config.width)
        center_y = int(self.draw.seed_rng.uniform(0.52, 0.59) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.15, 0.19) * self.config.width)
        height = int(self.draw.seed_rng.uniform(0.24, 0.29) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(-3, 3))

        start_angle = int(self.draw.seed_rng.uniform(45, 55))
        end_angle = int(self.draw.seed_rng.uniform(305, 315))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(width),
            height_location=IntLocation(height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(start_angle),
            end_angle_location=IntLocation(end_angle + 30),
        )

        bar_y = center_y + int(self.draw.seed_rng.uniform(-3, 1))
        bar_left_x = center_x - int(self.draw.seed_rng.uniform(4, 6))
        bar_right_x = bar_left_x + int(self.draw.seed_rng.uniform(7, 10))

        self.draw.draw_line(
            img,
            left_location=IntLocation(bar_left_x),
            top_location=IntLocation(bar_y),
            right_location=IntLocation(bar_right_x),
            bottom_location=IntLocation(bar_y),
        )

        # 2. Две точки над буквой
        dot_offset_y = int(self.draw.seed_rng.uniform(8, 14))  # Расстояние от верха основной части до точек
        dot_radius = int(self.draw.seed_rng.uniform(1, 1))  # Радиус точек
        dot_spacing = int(self.draw.seed_rng.uniform(8, 14))  # Расстояние между точками

        # Верхняя часть основного овала для позиционирования точек
        top_of_main_shape = center_y - height // 2

        # Позиция точек по Y
        dot_y = top_of_main_shape - dot_offset_y

        # Позиции точек по X (симметрично относительно центра буквы)
        dot1_x = center_x - dot_spacing // 2
        dot2_x = center_x + dot_spacing // 2

        # Рисуем первую точку как маленький замкнутый эллипс
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(dot1_x),
            center_y_range=IntLocation(dot_y),
            width_location=IntLocation(dot_radius),
            height_location=IntLocation(dot_radius),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(int(self.draw.seed_rng.uniform(0, 90))),
        )

        # Рисуем вторую точку
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(dot2_x),
            center_y_range=IntLocation(dot_y),
            width_location=IntLocation(dot_radius),
            height_location=IntLocation(dot_radius),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(int(self.draw.seed_rng.uniform(0, 90))),
        )

        return img

    def generate_ж(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Общая высота буквы
        total_height = int(self.draw.seed_rng.uniform(0.65, 0.75) * self.config.height)
        center_y = int(self.draw.seed_rng.uniform(0.48, 0.55) * self.config.height)

        # 1. Левая дуга (как у "е", открытая вправо)
        left_center_x = int(self.draw.seed_rng.uniform(0.32, 0.36) * self.config.width)
        left_width = int(self.draw.seed_rng.uniform(0.13, 0.17) * self.config.width)
        left_height = int(self.draw.seed_rng.uniform(0.20, 0.24) * self.config.height)
        angle = int(self.draw.seed_rng.uniform(-5, 5))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(left_center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(left_width),
            height_location=IntLocation(left_height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(-135),
            end_angle_location=IntLocation(135),
        )

        # 2. Правая дуга (зеркальная, открытая влево)
        right_center_x = int(self.draw.seed_rng.uniform(0.75, 0.78) * self.config.width)
        right_width = left_width  # Симметричная ширина
        right_height = left_height  # Симметричная высота

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(right_center_x),
            center_y_range=IntLocation(center_y),
            width_location=IntLocation(right_width),
            height_location=IntLocation(right_height),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(45),
            end_angle_location=IntLocation(315),
        )


        stick_x = (right_center_x + left_center_x) // 2

        self.draw.draw_line(
            img,
            left_location=IntLocation(stick_x),
            top_location=IntLocation(8),
            right_location=IntLocation(stick_x),
            bottom_location=IntLocation(18),
        )


        return img

    def generate_з(self) -> NDArray[np.uint8]:
        img = self._init_image()

        tilt = int(self.draw.seed_rng.uniform(5, 12))

        # 1. Верхняя дуга (строго верхняя половина цифры 3)
        rx1 = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry1 = int(self.draw.seed_rng.uniform(0.14, 0.18) * self.config.height)

        cx1 = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.width)
        cy1 = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx1), center_y_range=IntLocation(cy1),
            width_location=IntLocation(rx1), height_location=IntLocation(ry1),
            angle_location=IntLocation(tilt), start_angle_location=IntLocation(-150),
            end_angle_location=IntLocation(110)
        )

        # 2. Нижняя петля (короткая, компактная)
        rx2 = int(self.draw.seed_rng.uniform(0.14, 0.18) * self.config.width)
        ry2 = int(self.draw.seed_rng.uniform(0.20, 0.25) * self.config.height)  # Сделали высоту меньше

        # Центр нижней петли смещен влево и вниз от центра верхней
        cx2 = cx1 - int(rx1 * 0.5)
        cy2 = cy1 + ry1 + int(self.draw.seed_rng.uniform(2, 5))

        # 3. Рисуем нижнюю "тройку", переходящую в петлю "у"
        # Стартует от центра (270 - верх), идет вправо (0), вниз (90) и закручивается влево-вверх (210)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx2), center_y_range=IntLocation(cy2),
            width_location=IntLocation(rx2), height_location=IntLocation(ry2),
            angle_location=IntLocation(tilt), start_angle_location=IntLocation(-90), end_angle_location=IntLocation(210)
        )

        # 4. Диагональный перекрестный хвостик
        # Идет от левого края нижней петли наверх вправо
        cross_start_x = cx2 - rx2
        cross_start_y = cy2 + int(ry2 * 0.5)

        cross_end_x = cx2 + int(rx2 * 0.5)
        cross_end_y = cy2 - int(ry2 * 0.3)

        self.draw.draw_line(
            img, left_location=IntLocation(cross_start_x), top_location=IntLocation(cross_start_y),
            right_location=IntLocation(cross_end_x), bottom_location=IntLocation(cross_end_y)
        )

        return img

    def generate_и(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Радиусы (ширина и высота "петелек" буквы)
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.28, 0.34) * self.config.height)

        # Высота расположения (центр эллипсов)
        cy = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)

        # Центр левой петли
        cx1 = int(self.draw.seed_rng.uniform(0.25, 0.32) * self.config.width)

        # Центр правой петли.
        # Сдвигаем так, чтобы правая стенка первого эллипса и левая стенка второго слились в один центральный столбик
        offset = int(rx * 1.75) + int(self.draw.seed_rng.uniform(0, 1))
        cx2 = cx1 + offset

        # Общий наклон буквы (курсив)
        angle = int(self.draw.seed_rng.uniform(5, 12))

        # 1. Левая половина "и" (первая ямка)
        # start_angle: около -20 (правая стенка обрывается высоко, чтобы соединиться со второй половиной)
        # end_angle: около 220 (левая стенка - полноценный высокий столбик)
        start_angle_1 = int(self.draw.seed_rng.uniform(-30, -10))
        end_angle_1 = int(self.draw.seed_rng.uniform(200, 220))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx1),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(start_angle_1),
            end_angle_location=IntLocation(end_angle_1),
        )

        # 2. Правая половина "и" (с коротким хвостиком)
        # Задаем tail_angle = 15-35 градусов, чтобы линия оборвалась в самом низу
        # (чуть правее нижней точки), образуя аккуратный маленький хвостик.
        # end_angle: около 200 (левая стенка второго эллипса сливается с правой стенкой первого)
        tail_angle = int(self.draw.seed_rng.uniform(15, 35))
        end_angle_2 = int(self.draw.seed_rng.uniform(190, 210))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx2),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(tail_angle),
            end_angle_location=IntLocation(end_angle_2),
        )

        return img

    def generate_й(self) -> NDArray[np.uint8]:
        # Основа буквы "й" - это идеальная "и"
        img = self.generate_и()

        rx = int(self.draw.seed_rng.uniform(0.17, 0.21) * self.config.width)

        cy_hat = int(self.draw.seed_rng.uniform(0.001, 0.003) * self.config.height)
        cx_hat = int(self.draw.seed_rng.uniform(0.32, 0.40) * self.config.width)

        # Рисуем дугу над буквой
        hat_rx = int(rx * 1.5)
        hat_ry = int(self.draw.seed_rng.uniform(3, 5))
        angle = int(self.draw.seed_rng.uniform(-5, 5))

        # Углы для "улыбки", открытой вверх
        start_angle = int(self.draw.seed_rng.uniform(15, 30))
        end_angle = int(self.draw.seed_rng.uniform(150, 165))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx_hat),
            center_y_range=IntLocation(cy_hat),
            width_location=IntLocation(hat_rx),
            height_location=IntLocation(hat_ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(start_angle),
            end_angle_location=IntLocation(end_angle),
        )

        return img

    def generate_к(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # 1. Левая вертикальная (или слегка наклонная) палочка
        x1_top = int(self.draw.seed_rng.uniform(0.35, 0.40) * self.config.width)
        y1_top = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)

        x1_bot = x1_top - int(self.draw.seed_rng.uniform(2, 5))  # Курсивный сдвиг влево
        y1_bot = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)

        self.draw.draw_line(
            img,
            left_location=IntLocation(x1_top),
            top_location=IntLocation(y1_top),
            right_location=IntLocation(x1_bot),
            bottom_location=IntLocation(y1_bot),
        )

        # 2. Верхняя правая ветка
        # Отходит примерно от середины (чуть ниже) левой палочки
        mid_y = (y1_top + y1_bot) // 2 + int(self.draw.seed_rng.uniform(0, 3))
        mid_x = (x1_top + x1_bot) // 2

        x2_top = x1_top + int(self.draw.seed_rng.uniform(0.20, 0.25) * self.config.width)
        y2_top = y1_top + int(self.draw.seed_rng.uniform(-2, 3))

        # Легкая дуга для верхней ветки
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(mid_x),
            center_y_range=IntLocation(y2_top),
            width_location=IntLocation(x2_top - mid_x),
            height_location=IntLocation(mid_y - y2_top),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(90),
        )

        # 3. Нижняя правая ветка
        x2_bot = x1_bot + int(self.draw.seed_rng.uniform(0.22, 0.28) * self.config.width)
        y2_bot = y1_bot

        self.draw.draw_line(
            img,
            left_location=IntLocation(mid_x),
            top_location=IntLocation(mid_y),
            right_location=IntLocation(x2_bot),
            bottom_location=IntLocation(y2_bot),
        )

        return img

    def generate_л(self) -> NDArray[np.uint8]:
        img = self._init_image()

        bot_y = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)
        peak_y = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)

        left_x = int(self.draw.seed_rng.uniform(0.20, 0.30) * self.config.width)
        peak_x = left_x + int(self.draw.seed_rng.uniform(0.18, 0.25) * self.config.width)
        right_x = peak_x + int(self.draw.seed_rng.uniform(0.18, 0.25) * self.config.width)

        # 1. Стартовый левый крючок
        hook_r = int(self.draw.seed_rng.uniform(3, 6))
        hook_cx = left_x + hook_r
        hook_cy = bot_y - hook_r

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(hook_cx),
            center_y_range=IntLocation(hook_cy),
            width_location=IntLocation(hook_r),
            height_location=IntLocation(hook_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(90),  # От нижней точки
            end_angle_location=IntLocation(180),  # Только до левой точки (чтобы не слипалось)
        )

        # 2. Подъем к пику
        self.draw.draw_line(
            img,
            left_location=IntLocation(hook_cx),
            top_location=IntLocation(bot_y),  # От низа крючка
            right_location=IntLocation(peak_x),
            bottom_location=IntLocation(peak_y),
        )

        # 3. Спуск (правая ножка)
        self.draw.draw_line(
            img,
            left_location=IntLocation(peak_x),
            top_location=IntLocation(peak_y),
            right_location=IntLocation(right_x),
            bottom_location=IntLocation(bot_y),
        )

        # 4. Правый хвостик
        tail_r = int(self.draw.seed_rng.uniform(3, 6))
        tail_cx = right_x
        tail_cy = bot_y - tail_r

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(tail_cx),
            center_y_range=IntLocation(tail_cy),
            width_location=IntLocation(tail_r),
            height_location=IntLocation(tail_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),  # Справа
            end_angle_location=IntLocation(90),  # До нижней точки
        )

        return img

    def generate_м(self) -> NDArray[np.uint8]:
        img = self._init_image()

        bot_y = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)
        peak_y = int(self.draw.seed_rng.uniform(0.35, 0.45) * self.config.height)
        valley_y = bot_y - int(self.draw.seed_rng.uniform(0, 5))

        step = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)

        left_x = int(self.draw.seed_rng.uniform(0.15, 0.22) * self.config.width)
        peak1_x = left_x + step
        valley_x = peak1_x + step
        peak2_x = valley_x + step
        right_x = peak2_x + step

        # 1. Стартовый левый крючок
        hook_r = int(self.draw.seed_rng.uniform(3, 6))
        hook_cx = left_x + hook_r
        hook_cy = bot_y - hook_r

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(hook_cx),
            center_y_range=IntLocation(hook_cy),
            width_location=IntLocation(hook_r),
            height_location=IntLocation(hook_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(90),  # Начинается снизу
            end_angle_location=IntLocation(180),  # Идет только влево, не закручиваясь внутрь
        )

        # 2. Подъем на первый пик (стартует от нижней точки левого крючка)
        self.draw.draw_line(
            img,
            left_location=IntLocation(hook_cx),
            top_location=IntLocation(bot_y),
            right_location=IntLocation(peak1_x),
            bottom_location=IntLocation(peak_y),
        )

        # 3. Спуск в "долину" (середина буквы)
        self.draw.draw_line(
            img,
            left_location=IntLocation(peak1_x),
            top_location=IntLocation(peak_y),
            right_location=IntLocation(valley_x),
            bottom_location=IntLocation(valley_y),
        )

        # 4. Подъем на второй пик
        self.draw.draw_line(
            img,
            left_location=IntLocation(valley_x),
            top_location=IntLocation(valley_y),
            right_location=IntLocation(peak2_x),
            bottom_location=IntLocation(peak_y),
        )

        # 5. Спуск на правую ножку
        self.draw.draw_line(
            img,
            left_location=IntLocation(peak2_x),
            top_location=IntLocation(peak_y),
            right_location=IntLocation(right_x),
            bottom_location=IntLocation(bot_y),
        )

        # 6. Правый хвостик
        tail_r = int(self.draw.seed_rng.uniform(3, 6))
        tail_cx = right_x
        tail_cy = bot_y - tail_r

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(tail_cx),
            center_y_range=IntLocation(tail_cy),
            width_location=IntLocation(tail_r),
            height_location=IntLocation(tail_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),  # От правого края
            end_angle_location=IntLocation(90),  # Вниз до соединения с линией
        )

        return img

    def generate_н(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Две вертикальные наклонные линии и мостик между ними
        height = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)
        width = int(self.draw.seed_rng.uniform(0.20, 0.28) * self.config.width)

        x1_top = int(self.draw.seed_rng.uniform(0.30, 0.38) * self.config.width)
        y_top = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)

        dx1 = int(self.draw.seed_rng.uniform(2, 5))
        dx2 = int(self.draw.seed_rng.uniform(2, 5))

        x2_top = x1_top + width

        # Левая ножка
        self.draw.draw_line(
            img,
            left_location=IntLocation(x1_top),
            top_location=IntLocation(y_top),
            right_location=IntLocation(x1_top - dx1),
            bottom_location=IntLocation(y_top + height),
        )

        # Правая ножка
        self.draw.draw_line(
            img,
            left_location=IntLocation(x2_top),
            top_location=IntLocation(y_top),
            right_location=IntLocation(x2_top - dx2),
            bottom_location=IntLocation(y_top + height),
        )

        # Горизонтальная перемычка (немного провисающая или наклонная)
        bridge_y1 = y_top + height // 2 + int(self.draw.seed_rng.uniform(-2, 2))
        bridge_y2 = y_top + height // 2 + int(self.draw.seed_rng.uniform(-2, 2))

        bridge_x1 = x1_top - dx1 // 2
        bridge_x2 = x2_top - dx2 // 2

        self.draw.draw_line(
            img,
            left_location=IntLocation(bridge_x1),
            top_location=IntLocation(bridge_y1),
            right_location=IntLocation(bridge_x2),
            bottom_location=IntLocation(bridge_y2),
        )

        return img

    def generate_о(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Чистый, слегка наклонный эллипс без хвостиков
        cx = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.width)
        cy = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.height)

        rx = int(self.draw.seed_rng.uniform(0.15, 0.19) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.20, 0.25) * self.config.height)

        angle = int(self.draw.seed_rng.uniform(10, 25))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        return img

    # TODO: доделать
    def generate_п(self) -> NDArray[np.uint8]:
        img = self._init_image()


        bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)
        top_y = int(self.draw.seed_rng.uniform(0.30, 0.40) * self.config.height)

        width = int(self.draw.seed_rng.uniform(0.24, 0.32) * self.config.width)
        left_x = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.width)
        right_x = left_x + width

        arch_rx = width // 2
        arch_ry = int(self.draw.seed_rng.uniform(5, 8))

        arch_cy = top_y + arch_ry

        self.draw.draw_line(
            img,
            left_location=IntLocation(left_x),
            top_location=IntLocation(top_y - 2),
            right_location=IntLocation(left_x),
            bottom_location=IntLocation(bot_y),
        )

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(left_x + arch_rx),
            center_y_range=IntLocation(arch_cy),
            width_location=IntLocation(arch_rx),
            height_location=IntLocation(arch_ry),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(180),
            end_angle_location=IntLocation(360),
        )

        self.draw.draw_line(
            img,
            left_location=IntLocation(right_x),
            top_location=IntLocation(arch_cy),
            right_location=IntLocation(right_x),
            bottom_location=IntLocation(bot_y),
        )

        tail_r = int(self.draw.seed_rng.uniform(2, 4))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(right_x + tail_r - 1),
            center_y_range=IntLocation(bot_y - tail_r),
            width_location=IntLocation(tail_r),
            height_location=IntLocation(tail_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(90),
            end_angle_location=IntLocation(180),
        )

        return img

    def generate_р(self) -> NDArray[np.uint8]:
        img = self._init_image()

        top_y = int(self.draw.seed_rng.uniform(0.20, 0.30) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.85, 0.95) * self.config.height)
        left_x = int(self.draw.seed_rng.uniform(0.35, 0.45) * self.config.width)
        tilt = int(self.draw.seed_rng.uniform(3, 6))

        self.draw.draw_line(
            img, left_location=IntLocation(left_x), top_location=IntLocation(top_y),
            right_location=IntLocation(left_x - tilt), bottom_location=IntLocation(bot_y)
        )

        arch_start_y = top_y + int(self.draw.seed_rng.uniform(2, 6))

        width = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.width)
        arch_ry = int(self.draw.seed_rng.uniform(5, 8))

        cx = left_x + width // 2
        cy = arch_start_y + arch_ry

        arch_angle = int(self.draw.seed_rng.uniform(-25, -15))

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx), center_y_range=IntLocation(cy),
            width_location=IntLocation(width // 2), height_location=IntLocation(arch_ry),
            angle_location=IntLocation(arch_angle), start_angle_location=IntLocation(180),
            end_angle_location=IntLocation(360)
        )

        base_y = int(self.draw.seed_rng.uniform(0.65, 0.75) * self.config.height)

        right_x = cx + int((width // 2) * math.cos(math.radians(abs(arch_angle))))

        self.draw.draw_line(
            img, left_location=IntLocation(right_x), top_location=IntLocation(cy),
            right_location=IntLocation(right_x - int(tilt / 2)), bottom_location=IntLocation(base_y)
        )

        tail_len = int(self.draw.seed_rng.uniform(3, 6))
        self.draw.draw_line(
            img, left_location=IntLocation(right_x - int(tilt / 2)), top_location=IntLocation(base_y),
            right_location=IntLocation(right_x - int(tilt / 2) + tail_len), bottom_location=IntLocation(base_y - 2)
        )

        return img

    def generate_т(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Идеальная перевернутая "ш" (3 свода)
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.28, 0.35) * self.config.height)

        # Центр арок поднят выше, так как своды смотрят вверх
        cy = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.height)

        cx1 = int(self.draw.seed_rng.uniform(0.25, 0.32) * self.config.width)
        offset = int(rx * 1.85) + int(self.draw.seed_rng.uniform(-1, 1))
        cx2 = cx1 + offset
        cx3 = cx2 + offset

        angle = int(self.draw.seed_rng.uniform(5, 15))

        # 1. Первый свод (левый)
        # Рисуем верхнюю половину эллипса: от 150 (лево-низ) через верх (270) до 30 (право-низ)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx1), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(150),
            end_angle_location=IntLocation(360 + 30)
        )

        # 2. Второй свод (центральный)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx2), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(150),
            end_angle_location=IntLocation(360 + 30)
        )

        return img

    def generate_с(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Буква "с" - это просто левая половина наклоненного овала
        cx = int(self.draw.seed_rng.uniform(0.50, 0.58) * self.config.width)
        cy = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.height)

        rx = int(self.draw.seed_rng.uniform(0.14, 0.18) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.20, 0.26) * self.config.height)

        angle = int(self.draw.seed_rng.uniform(5, 20))  # Курсивный наклон

        # В OpenCV 0 это право, 90 низ, 180 лево, 270 верх.
        # Чтобы нарисовать левый полумесяц, идем от 45 (низ-право) через лево до 315 (верх-право).
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(45),
            end_angle_location=IntLocation(315),
        )

        return img



    def generate_у(self) -> NDArray[np.uint8]:
        img = self._init_image()

        cy1 = int(self.draw.seed_rng.uniform(0.25, 0.32) * self.config.height)
        cx1 = int(self.draw.seed_rng.uniform(0.35, 0.42) * self.config.width)

        rx1 = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry1 = int(self.draw.seed_rng.uniform(0.15, 0.20) * self.config.height)

        # 1. Верхняя чаша
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx1),
            center_y_range=IntLocation(cy1),
            width_location=IntLocation(rx1),
            height_location=IntLocation(ry1),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(180),
        )

        # Правый край чаши
        loop_top_x = cx1 + rx1
        loop_top_y = cy1

        # Ограничили длину хвоста, чтобы не уходил за пределы массива
        loop_bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)

        # 2. Прямая линия вниз
        self.draw.draw_line(
            img,
            left_location=IntLocation(loop_top_x),
            top_location=IntLocation(loop_top_y),
            right_location=IntLocation(loop_top_x),
            bottom_location=IntLocation(loop_bot_y),
        )

        # 3. Закругление внизу петли
        loop_r = int(self.draw.seed_rng.uniform(4, 7))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(loop_top_x - loop_r),
            center_y_range=IntLocation(loop_bot_y),
            width_location=IntLocation(loop_r),
            height_location=IntLocation(loop_r),
            angle_location=IntLocation(0),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(180),
        )

        # 4. Диагональ (перекрестие)
        cross_end_x = loop_top_x + int(self.draw.seed_rng.uniform(4, 8))
        cross_end_y = loop_top_y + int(self.draw.seed_rng.uniform(2, 6))

        self.draw.draw_line(
            img,
            left_location=IntLocation(loop_top_x - loop_r * 2),
            top_location=IntLocation(loop_bot_y),
            right_location=IntLocation(cross_end_x),
            bottom_location=IntLocation(cross_end_y),
        )

        return img

    def generate_ф(self) -> NDArray[np.uint8]:
        img = self._init_image()

        x_mid = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.width)
        top_y = int(self.draw.seed_rng.uniform(0.15, 0.25) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)

        dx = int(self.draw.seed_rng.uniform(1, 4))  # Наклон мачты

        # 1. Овал
        # Центр овала находится примерно посередине мачты
        cx = x_mid - dx // 2
        cy = (top_y + bot_y) // 2

        rx = int(self.draw.seed_rng.uniform(0.20, 0.28) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.18, 0.22) * self.config.height)

        angle = int(self.draw.seed_rng.uniform(-5, 5))

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(360),
        )

        # 2. Вертикальная мачта (рисуется поверх овала, разрезая его)
        self.draw.draw_line(
            img,
            left_location=IntLocation(x_mid),
            top_location=IntLocation(top_y),
            right_location=IntLocation(x_mid - dx),
            bottom_location=IntLocation(bot_y),
        )

        return img

    def generate_х(self) -> NDArray[np.uint8]:
        img = self._init_image()

        cx = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.width)
        cy = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)

        # Делаем радиусы шире, чтобы буква была более округлой
        rx = int(self.draw.seed_rng.uniform(0.14, 0.20) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.20, 0.28) * self.config.height)

        tilt = int(self.draw.seed_rng.uniform(5, 15))

        # Сдвигаем центры ровно на радиус + небольшой зазор,
        # чтобы они аккуратно касались/чуть-чуть пересекались в самом центре
        offset = rx + int(self.draw.seed_rng.uniform(0, 2))

        # 1. Левая половинка (дуга, открытая влево)
        # Углы 70 и 290 заставляют дугу сильнее закругляться к центру
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx + offset),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(tilt),
            start_angle_location=IntLocation(70),
            end_angle_location=IntLocation(290),
        )

        # 2. Правая половинка (дуга, открытая вправо)
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx - offset),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(tilt),
            start_angle_location=IntLocation(-110),
            end_angle_location=IntLocation(110),
        )

        return img

    def generate_ц(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # База "ц" - буква "и"
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.28, 0.34) * self.config.height)
        cy = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)

        cx1 = int(self.draw.seed_rng.uniform(0.25, 0.32) * self.config.width)
        offset = int(rx * 1.75) + int(self.draw.seed_rng.uniform(0, 1))
        cx2 = cx1 + offset
        angle = int(self.draw.seed_rng.uniform(5, 12))

        # 1. Левая половина "и"
        start_angle_1 = int(self.draw.seed_rng.uniform(-30, -10))
        end_angle_1 = int(self.draw.seed_rng.uniform(200, 220))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx1),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(start_angle_1),
            end_angle_location=IntLocation(end_angle_1),
        )

        # 2. Правая половина
        # start_angle = 0, чтобы правая линия пошла строго до самого низа без хвостика
        end_angle_2 = int(self.draw.seed_rng.uniform(190, 210))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(cx2),
            center_y_range=IntLocation(cy),
            width_location=IntLocation(rx),
            height_location=IntLocation(ry),
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(0),
            end_angle_location=IntLocation(end_angle_2),
        )

        # 3. Очень маленький хвостик (как просили)
        loop_r = int(self.draw.seed_rng.uniform(2, 4))
        loop_cx = cx2 + rx
        loop_cy = cy + ry + loop_r - 1

        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(loop_cx),
            center_y_range=IntLocation(loop_cy),
            width_location=IntLocation(loop_r),
            height_location=IntLocation(int(loop_r * 1.5)),  # Чуть вытянута вниз
            angle_location=IntLocation(angle),
            start_angle_location=IntLocation(-90),  # Приклеена к ножке
            end_angle_location=IntLocation(180),  # Закругляется влево
        )

        return img

    def generate_ч(self) -> NDArray[np.uint8]:
        img = self._init_image()

        cup_rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        cup_ry = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.height)

        cup_cx = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.width)
        cup_cy = int(self.draw.seed_rng.uniform(0.40, 0.48) * self.config.height)

        # 1. Чашечка (U-форма)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cup_cx), center_y_range=IntLocation(cup_cy),
            width_location=IntLocation(cup_rx), height_location=IntLocation(cup_ry),
            angle_location=IntLocation(0), start_angle_location=IntLocation(0), end_angle_location=IntLocation(200),
        )

        # 2. Правая мачта
        stick_x = cup_cx + cup_rx
        top_y = cup_cy
        bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)

        self.draw.draw_line(
            img, left_location=IntLocation(stick_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick_x), bottom_location=IntLocation(bot_y),
        )

        # 3. Финишный хвостик (увеличен, поднят и сдвинут вправо)
        tail_r = int(self.draw.seed_rng.uniform(5, 8))  # Был 4-7
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(stick_x + tail_r), center_y_range=IntLocation(bot_y - tail_r + 1),
            width_location=IntLocation(tail_r), height_location=IntLocation(tail_r),
            angle_location=IntLocation(0), start_angle_location=IntLocation(90), end_angle_location=IntLocation(180),
        )

        return img


    def generate_ш(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # три раза повторяем геометрию "и"
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.28, 0.35) * self.config.height)
        cy = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.height)

        cx1 = int(self.draw.seed_rng.uniform(0.18, 0.25) * self.config.width)
        offset = int(rx * 1.75) + int(self.draw.seed_rng.uniform(0, 1))
        cx2 = cx1 + offset
        cx3 = cx2 + offset  # Третья секция

        angle = int(self.draw.seed_rng.uniform(5, 15))

        # 1. Первая ямка
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx1), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(-30),
            end_angle_location=IntLocation(220),
        )

        # 2. Вторая ямка
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx2), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(-30),
            end_angle_location=IntLocation(210),
        )

        # 3. Третья ямка (с хвостиком)
        tail_angle = int(self.draw.seed_rng.uniform(15, 35))
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx3), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(tail_angle),
            end_angle_location=IntLocation(210),
        )

        return img

    def generate_щ(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Точная копия "ш" (3 ямки)
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.25, 0.32) * self.config.height)
        cy = int(self.draw.seed_rng.uniform(0.35, 0.45) * self.config.height)

        cx1 = int(self.draw.seed_rng.uniform(0.18, 0.25) * self.config.width)
        offset = int(rx * 1.75) + int(self.draw.seed_rng.uniform(0, 1))
        cx2 = cx1 + offset
        cx3 = cx2 + offset

        angle = int(self.draw.seed_rng.uniform(5, 15))

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx1), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(-30),
            end_angle_location=IntLocation(220),
        )

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx2), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(-30),
            end_angle_location=IntLocation(210),
        )

        # Третья ямка уходит строго вниз (start_angle=0) для присоединения петельки
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx3), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(0), end_angle_location=IntLocation(210),
        )

        # Добавляем маленькую петельку (копия из "ц")
        loop_r = int(self.draw.seed_rng.uniform(2, 4))
        loop_cx = cx3 + rx
        loop_cy = cy + ry + loop_r - 1

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(loop_cx), center_y_range=IntLocation(loop_cy),
            width_location=IntLocation(loop_r), height_location=IntLocation(int(loop_r * 1.5)),
            angle_location=IntLocation(angle), start_angle_location=IntLocation(-90),
            end_angle_location=IntLocation(180),
        )

        return img

    def generate_ъ(self) -> NDArray[np.uint8]:
        img = self._init_image()

        top_y = int(self.draw.seed_rng.uniform(0.30, 0.40) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)
        stick_x = int(self.draw.seed_rng.uniform(0.40, 0.50) * self.config.width)

        # 1. Центральная мачта
        self.draw.draw_line(
            img,
            left_location=IntLocation(stick_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick_x), bottom_location=IntLocation(bot_y)
        )

        # 2. Нижняя петля (справа от мачты)
        loop_r = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(stick_x + loop_r), center_y_range=IntLocation(bot_y - loop_r),
            width_location=IntLocation(loop_r), height_location=IntLocation(int(loop_r * 1.2)),
            angle_location=IntLocation(0), start_angle_location=IntLocation(0), end_angle_location=IntLocation(360)
        )

        # 3. Горизонтальный козырек (влево от верха мачты)
        visor_len = int(self.draw.seed_rng.uniform(0.15, 0.20) * self.config.width)
        left_visor_x = stick_x - visor_len
        self.draw.draw_line(
            img,
            left_location=IntLocation(left_visor_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick_x), bottom_location=IntLocation(top_y)
        )

        # 4. Хвостик-засечка на козырьке (от -180 (лево) до -90 (верх))
        hook_r = int(self.draw.seed_rng.uniform(2, 4))
        self.draw.draw_ellipse(
            img,
            center_x_location=IntLocation(left_visor_x + hook_r), center_y_range=IntLocation(top_y + hook_r),
            width_location=IntLocation(hook_r), height_location=IntLocation(hook_r),
            angle_location=IntLocation(0), start_angle_location=IntLocation(180), end_angle_location=IntLocation(270)
        )

        return img

    def generate_ы(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Удлинили палки (top_y стал меньше)
        top_y = int(self.draw.seed_rng.uniform(0.20, 0.30) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)

        # 1. Левая мачта
        stick1_x = int(self.draw.seed_rng.uniform(0.20, 0.25) * self.config.width)
        self.draw.draw_line(
            img, left_location=IntLocation(stick1_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick1_x), bottom_location=IntLocation(bot_y)
        )

        # 2. Левая петля (увеличен минимальный размер)
        loop_r = int(self.draw.seed_rng.uniform(0.14, 0.18) * self.config.width)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(stick1_x + loop_r), center_y_range=IntLocation(bot_y - loop_r),
            width_location=IntLocation(loop_r), height_location=IntLocation(int(loop_r * 1.2)),
            angle_location=IntLocation(0), start_angle_location=IntLocation(0), end_angle_location=IntLocation(360)
        )

        # 3. Правая мачта (увеличен отступ)
        gap = int(self.draw.seed_rng.uniform(0.10, 0.18) * self.config.width)  # Был 0.05-0.10
        stick2_x = stick1_x + loop_r * 2 + gap
        self.draw.draw_line(
            img, left_location=IntLocation(stick2_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick2_x), bottom_location=IntLocation(bot_y)
        )

        # 4. Правый хвостик
        tail_r = int(self.draw.seed_rng.uniform(3, 5))
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(stick2_x + tail_r - 1), center_y_range=IntLocation(bot_y - tail_r),
            width_location=IntLocation(tail_r), height_location=IntLocation(tail_r),
            angle_location=IntLocation(0), start_angle_location=IntLocation(90), end_angle_location=IntLocation(180)
        )

        return img

    def generate_ь(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Удлинили палку
        top_y = int(self.draw.seed_rng.uniform(0.20, 0.30) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)
        stick_x = int(self.draw.seed_rng.uniform(0.35, 0.45) * self.config.width)

        # 1. Мачта
        self.draw.draw_line(
            img, left_location=IntLocation(stick_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick_x), bottom_location=IntLocation(bot_y)
        )

        # 2. Петля
        loop_r = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(stick_x + loop_r), center_y_range=IntLocation(bot_y - loop_r),
            width_location=IntLocation(loop_r), height_location=IntLocation(int(loop_r * 1.2)),
            angle_location=IntLocation(0), start_angle_location=IntLocation(0), end_angle_location=IntLocation(360)
        )

        return img

    def generate_э(self) -> NDArray[np.uint8]:
        img = self._init_image()

        cx = int(self.draw.seed_rng.uniform(0.50, 0.60) * self.config.width)
        cy = int(self.draw.seed_rng.uniform(0.45, 0.55) * self.config.height)

        # Делаем эллипс более вытянутым по вертикали (как "С")
        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.24, 0.30) * self.config.height)

        # 1. Полукруг (усилено закругление: от -120 до 120)
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(0), start_angle_location=IntLocation(-120), end_angle_location=IntLocation(120)
        )

        # 2. Горизонтальная перекладина (удлинена и сдвинута вправо, чтобы проткнуть дугу)
        bar_len = int(self.draw.seed_rng.uniform(0.18, 0.24) * self.config.width)
        # Начинается левее, а заканчивается немного ПРАВЕЕ центра дуги (cx)
        self.draw.draw_line(
            img, left_location=IntLocation(cx - bar_len), top_location=IntLocation(cy),
            right_location=IntLocation(cx + int(rx * 0.5)), bottom_location=IntLocation(cy)
        )

        return img

    def generate_ю(self) -> NDArray[np.uint8]:
        img = self._init_image()

        # Наклон менее агрессивный
        tilt = int(self.draw.seed_rng.uniform(1, 3))

        # 1. Левая мачта
        x1_top = int(self.draw.seed_rng.uniform(0.20, 0.28) * self.config.width)
        y1_top = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)
        x1_bot = x1_top - tilt
        y1_bot = int(self.draw.seed_rng.uniform(0.70, 0.80) * self.config.height)

        self.draw.draw_line(
            img, left_location=IntLocation(x1_bot), top_location=IntLocation(y1_bot),
            right_location=IntLocation(x1_top), bottom_location=IntLocation(y1_top)
        )

        # 2. Горизонтальная перемычка
        y_mid = (y1_top + y1_bot) // 2
        x_mid = (x1_top + x1_bot) // 2

        rx = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        ry = int(self.draw.seed_rng.uniform(0.18, 0.24) * self.config.height)

        bridge_len = int(self.draw.seed_rng.uniform(0.18, 0.24) * self.config.width)
        x_bridge_end = x_mid + bridge_len

        self.draw.draw_line(
            img, left_location=IntLocation(x_mid), top_location=IntLocation(y_mid),
            right_location=IntLocation(x_bridge_end), bottom_location=IntLocation(y_mid)
        )

        # 3. Овал
        cx = x_bridge_end + rx
        cy = y_mid

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(cx), center_y_range=IntLocation(cy),
            width_location=IntLocation(rx), height_location=IntLocation(ry),
            angle_location=IntLocation(tilt), start_angle_location=IntLocation(0), end_angle_location=IntLocation(360)
        )

        return img

    def generate_я(self) -> NDArray[np.uint8]:
        img = self._init_image()

        top_y = int(self.draw.seed_rng.uniform(0.25, 0.35) * self.config.height)
        bot_y = int(self.draw.seed_rng.uniform(0.75, 0.85) * self.config.height)

        # 1. Правая мачта (опускается до самого низа)
        stick_x = int(self.draw.seed_rng.uniform(0.65, 0.75) * self.config.width)
        self.draw.draw_line(
            img, left_location=IntLocation(stick_x), top_location=IntLocation(top_y),
            right_location=IntLocation(stick_x), bottom_location=IntLocation(bot_y)
        )

        # 2. Верхняя левая петля
        loop_r = int(self.draw.seed_rng.uniform(0.12, 0.16) * self.config.width)
        loop_cy = top_y + loop_r
        loop_cx = stick_x - loop_r

        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(loop_cx), center_y_range=IntLocation(loop_cy),
            width_location=IntLocation(loop_r), height_location=IntLocation(loop_r),
            angle_location=IntLocation(0), start_angle_location=IntLocation(0), end_angle_location=IntLocation(360)
        )

        # 3. Диагональная ножка (ограничиваем минимальный угол, чтобы была круче)
        leg_start_x = stick_x
        leg_start_y = loop_cy + loop_r

        # Ножка не может уходить слишком далеко влево (крутой наклон)
        leg_end_x = int(self.draw.seed_rng.uniform(0.35, 0.45) * self.config.width)

        self.draw.draw_line(
            img, left_location=IntLocation(leg_end_x), top_location=IntLocation(bot_y),
            right_location=IntLocation(leg_start_x), bottom_location=IntLocation(leg_start_y)
        )

        # 4. Финишный хвостик от правой мачты
        tail_r = int(self.draw.seed_rng.uniform(3, 5))
        self.draw.draw_ellipse(
            img, center_x_location=IntLocation(stick_x + tail_r - 1), center_y_range=IntLocation(bot_y - tail_r),
            width_location=IntLocation(tail_r), height_location=IntLocation(tail_r),
            angle_location=IntLocation(0), start_angle_location=IntLocation(90), end_angle_location=IntLocation(180)
        )

        return img