import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Union
from DL.layers import ArcFaceLayer


class MyConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomClassifierFromConfig(nn.Module):
    def __init__(self, config: Dict[str, Any], use_pixel_count: bool = False, use_arcface: bool = False):
        super().__init__()

        if 'layers' not in config:
            raise ValueError("Configuration dictionary must contain 'layers' key.")

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
                raise NotImplementedError(f"Layer type {layer['type']} is not implemented.")

        self.feature_extractor = nn.Sequential(*layers_list)
        self.fc_flat = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        # Увеличиваем вектор на 1, если используем счетчик пикселей
        if self.use_pixel_count:
            flattened_size += 1

        if self.use_arcface:
            self.cls_ = ArcFaceLayer(flattened_size, self.num_classes)
        else:
            self.cls_ = nn.Linear(flattened_size, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_extra: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        blocks = []
        features = x  # Сохраняем исходную картинку для подсчета пикселей

        for layer in self.feature_extractor:
            x = layer(x)
            if isinstance(layer, MyConv2d) and len(blocks) < 2:
                blocks.append(x)

        flatten = self.fc_flat(x)

        # Добавляем инженерный признак (доля белых пикселей на картинке)
        if self.use_pixel_count:
            pixel_density = features.view(features.size(0), -1).mean(dim=1, keepdim=True)
            flatten = torch.cat([flatten, pixel_density], dim=1)

        # Классификатор (ArcFace требует метки при обучении)
        if self.use_arcface:
            classes = self.cls_(flatten, labels)
        else:
            classes = self.cls_(flatten)

        if return_extra:
            return classes, blocks

        return classes


import torch.nn.functional as F


class RotateNet(nn.Module):
    """
    Сеть для оценки угла поворота.
    Предсказывает [sin(theta), cos(theta)] и отдает латентный вектор.
    По архитектуре статьи: [(1->8)->MP2] -> [(8->16)->MP2] ->[(16->32)]
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.flat = nn.Flatten()

        # Определяем размер вектора: 28 -> /2 -> 14 -> /2 -> 7. Итог: 32 * 7 * 7 = 1568
        self.regressor = nn.Linear(1568, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        flat_f = self.flat(f)
        sin_cos = self.regressor(flat_f)

        # Строго нормализуем, чтобы значения лежали на единичной окружности (sin^2 + cos^2 = 1)
        sin_cos = F.normalize(sin_cos, p=2, dim=1)
        return sin_cos, flat_f


class CascadeNet(nn.Module):
    """
    Ультимативная архитектура: RotateNet + LeNet с Feature Fusion.
    1. Оценивает угол через RotateNet.
    2. Программно вращает картинку в прямое положение (STN).
    3. Отдает выровненную картинку в LeNet.
    4. Приклеивает фичи RotateNet к Flatten слою LeNet (ваша идея!).
    """

    def __init__(self, classifier_config: Dict[str, Any], rotatenet: RotateNet):
        super().__init__()
        self.rotatenet = rotatenet

        # Замораживаем RotateNet (мы предполагаем, что он уже обучен)
        for param in self.rotatenet.parameters():
            param.requires_grad = False

        self.classifier = CustomClassifierFromConfig(classifier_config)
        self.feature_extractor = self.classifier.feature_extractor  # Для Grad-CAM

        # Получаем оригинальный размер Flatten из LeNet и прибавляем размер фичей RotateNet (1568)
        old_in_features = self.classifier.cls_.in_features
        new_in_features = old_in_features + 1568

        self.classifier.cls_ = nn.Linear(new_in_features, self.classifier.num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_extra: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        with torch.no_grad():
            self.rotatenet.eval()
            sin_cos, rot_f = self.rotatenet(x)

        # Вытаскиваем sin и cos
        sin_a = sin_cos[:, 0]
        cos_a = sin_cos[:, 1]

        # --- БЛОК ПРОСТРАНСТВЕННОГО ВЫРАВНИВАНИЯ (STN) ---
        # Чтобы повернуть картинку обратно на угол (-theta), нужно передать в affine_grid
        # матрицу поворота на (+theta), так как функция делает обратный маппинг.
        batch_size = x.size(0)
        theta = torch.zeros(batch_size, 2, 3, device=x.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        aligned_x = F.grid_sample(x, grid, align_corners=False)

        # --- БЛОК КЛАССИФИКАЦИИ ---
        blocks = []
        features = aligned_x
        for layer in self.classifier.feature_extractor:
            features = layer(features)
            if isinstance(layer, MyConv2d) and len(blocks) < 2:
                blocks.append(features)

        flat_clf = self.classifier.fc_flat(features)

        # --- БЛОК FEATURE FUSION ---
        # Склеиваем локальные признаки (LeNet) и глобальные геометрические (RotateNet)
        fused_features = torch.cat([flat_clf, rot_f], dim=1)

        out = self.classifier.cls_(fused_features)

        if return_extra:
            # Возвращаем aligned_x вместе с блоками для корректного расчета BG Loss
            return out, blocks, aligned_x

        return out