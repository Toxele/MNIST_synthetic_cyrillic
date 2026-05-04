import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Union
from DL.models.my_conv2d import MyConv2d
from DL.layers.arcface_layer import ArcFaceLayer


class CustomClassifierFromConfig(nn.Module):
    """
    Классификатор, генерируемый на основе конфигурационного словаря.
    Поддерживает добавление счетчика пикселей и метрическое обучение (ArcFace).
    """

    def __init__(self, config: Dict[str, Any], use_pixel_count: bool = False, use_arcface: bool = False):
        super().__init__()

        if 'layers' not in config:
            raise ValueError("Словарь конфигурации должен содержать ключ 'layers'")

        self.image_size = config.get('image_size', 28)
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 33)
        self.use_pixel_count = use_pixel_count
        self.use_arcface = use_arcface

        layers_list = []
        for layer in config['layers']:
            if layer['type'] == 'conv':
                layers_list.append(
                    MyConv2d(layer['in'], layer['out'], kernel_size=layer['kernel_size'], padding=layer['padding']))
            elif layer['type'] == 'pool':
                layers_list.append(
                    nn.MaxPool2d(kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding']))
            else:
                raise NotImplementedError(f"Тип слоя {layer['type']} не реализован")

        self.feature_extractor = nn.Sequential(*layers_list)
        self.fc_flat = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        if self.use_pixel_count:
            flattened_size += 1

        if self.use_arcface:
            self.cls_ = ArcFaceLayer(flattened_size, self.num_classes)
        else:
            self.cls_ = nn.Linear(flattened_size, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_extra: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        blocks = []
        features = x

        for layer in self.feature_extractor:
            x = layer(x)
            if isinstance(layer, MyConv2d) and len(blocks) < 2:
                blocks.append(x)

        flatten = self.fc_flat(x)

        if self.use_pixel_count:
            pixel_density = features.view(features.size(0), -1).mean(dim=1, keepdim=True)
            flatten = torch.cat([flatten, pixel_density], dim=1)

        if self.use_arcface:
            classes = self.cls_(flatten, labels)
        else:
            classes = self.cls_(flatten)

        if return_extra:
            return classes, blocks

        return classes