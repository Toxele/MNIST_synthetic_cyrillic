import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from DL.models.my_conv2d import MyConv2d
from DL.layers.spatial_softmax import SpatialSoftmax


class RotateNet(nn.Module):
    """
    Универсальная сеть для оценки угла поворота, строящаяся из JSON конфигурации.
    Поддерживает обычный Flatten или SpatialSoftmax в качестве финального слоя.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.in_channels = config.get('in_channels', 1)
        self.image_size = config.get('image_size', 28)
        self.head_type = config.get('head_type', 'flatten')

        layers_list = []
        for layer in config['layers']:
            if layer['type'] == 'conv':
                layers_list.append(
                    MyConv2d(layer['in'], layer['out'], kernel_size=layer['kernel_size'], padding=layer['padding']))
            elif layer['type'] == 'pool':
                layers_list.append(
                    nn.MaxPool2d(kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding']))
            else:
                raise NotImplementedError(f"Layer type {layer['type']} is not implemented.")

        self.features = nn.Sequential(*layers_list)

        # Динамически вычисляем размерности выхода сверток с помощью фиктивного тензора
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
            dummy_out = self.features(dummy_input)
            _, C, H, W = dummy_out.shape

        if self.head_type == 'spatial_softmax':
            self.head = SpatialSoftmax(height=H, width=W)
            self.out_features = C * 2  # По 2 координаты (X, Y) на каждый канал
        elif self.head_type == 'flatten':
            self.head = nn.Flatten()
            self.out_features = C * H * W
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

        self.regressor = nn.Linear(self.out_features, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)
        latent_features = self.head(feat_map)

        sin_cos = self.regressor(latent_features)
        sin_cos = F.normalize(sin_cos, p=2, dim=1)

        return sin_cos, latent_features