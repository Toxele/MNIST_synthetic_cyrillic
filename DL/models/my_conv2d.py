import torch
import torch.nn as nn

class MyConv2d(nn.Module):
    """
    Базовый сверточный блок: Свертка -> Пакетная нормализация -> ReLU.
    Отключен inplace для корректной работы алгоритмов интерпретируемости (Grad-CAM).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)