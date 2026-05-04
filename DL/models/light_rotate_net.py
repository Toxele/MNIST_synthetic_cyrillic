import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightRotateNet(nn.Module):
    """
    Легковесная сеть для оценки угла поворота (размер ~4 КБ).
    Использует Global Average Pooling для радикального сокращения параметров
    и формирования компактного геометрического эмбеддинга.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Сжатая сверточная часть (всего 12 фильтров суммарно)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(4, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 7x7
        )

        # GAP превращает 8x7x7 в вектор из 8 значений
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Геометрический паспорт (Эмбеддинг)
        self.embedding_layer = nn.Sequential(
            nn.Linear(8, self.embedding_dim),
            nn.ReLU(inplace=False)
        )

        # Регрессор угла
        self.regressor = nn.Linear(self.embedding_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.gap(x)
        x = self.flatten(x)

        # Формируем компактное описание формы
        embedding = self.embedding_layer(x)

        # Предсказываем sin/cos
        sin_cos = self.regressor(embedding)
        sin_cos = F.normalize(sin_cos, p=2, dim=1)

        return sin_cos, embedding