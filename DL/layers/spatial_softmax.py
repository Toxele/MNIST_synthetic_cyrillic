import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """
    Интегральная регрессия координат (Spatial Softmax / Differentiable Center of Mass).
    Вычисляет математическое ожидание координат (X, Y) для каждого канала.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        # Создаем сетку координат от -1 до 1
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing='ij'
        )
        # ИСПРАВЛЕНИЕ: Используем reshape(-1) вместо view(-1) для несмежных тензоров
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape

        # Превращаем карту HxW в плоский вектор и берем Softmax
        x_flat = x.view(batch_size, channels, -1)
        softmax_weights = F.softmax(x_flat, dim=-1)

        # Интегральное ожидание координаты X и Y
        expected_x = torch.sum(softmax_weights * self.pos_x, dim=-1)
        expected_y = torch.sum(softmax_weights * self.pos_y, dim=-1)

        # Конкатенируем координаты: [cx1, cy1, cx2, cy2, ...]
        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        return keypoints