import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Union


class MyConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        # В статье явно указано использование Conv-BN-ReLU. Bias=False, т.к. есть BN
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)  # Отключаем inplace для корректной работы Grad-CAM
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomClassifierFromConfig(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        if 'layers' not in config:
            raise ValueError("Configuration dictionary must contain 'layers' key.")

        self.image_size = config.get('image_size', 28)
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 33)

        layers_list = []
        for layer in config['layers']:
            if layer['type'] == 'conv':
                layers_list.append(
                    MyConv2d(layer['in'], layer['out'], kernel_size=layer['kernel_size'], padding=layer['padding'])
                )
            elif layer['type'] == 'pool':
                layers_list.append(
                    nn.MaxPool2d(kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding'])
                )
            else:
                raise NotImplementedError(f"Layer type {layer['type']} is not implemented.")

        self.feature_extractor = nn.Sequential(*layers_list)
        self.fc_flat = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.cls_ = nn.Linear(flattened_size, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor, return_extra: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        blocks = []
        for layer in self.feature_extractor:
            x = layer(x)
            if isinstance(layer, MyConv2d) and len(blocks) < 2:
                blocks.append(x)

        flatten = self.fc_flat(x)
        classes = self.cls_(flatten)

        if return_extra:
            return classes, blocks

        return classes