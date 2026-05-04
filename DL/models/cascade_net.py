import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Union
from DL.models.custom_classifier_from_config import CustomClassifierFromConfig
from DL.models.my_conv2d import MyConv2d


class CascadeNet(nn.Module):
    """
    Каскадная архитектура с итеративным STN и полярным слиянием.
    """

    def __init__(self, classifier_config: Dict[str, Any], rotatenet: nn.Module, use_polar: bool = False,
                 stn_iters: int = 2):
        super().__init__()
        self.rotatenet = rotatenet
        self.use_polar = use_polar
        self.stn_iters = stn_iters

        for param in self.rotatenet.parameters():
            param.requires_grad = False

        self.classifier = CustomClassifierFromConfig(classifier_config)
        self.feature_extractor = self.classifier.feature_extractor

        old_in_features = self.classifier.cls_.in_features
        new_in_features = old_in_features + self.rotatenet.out_features
        self.classifier.cls_ = nn.Linear(new_in_features, self.classifier.num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_extra: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        batch_size = x.size(0)

        # Начальный угол: 0 градусов (sin=0, cos=1)
        sin_a = torch.zeros(batch_size, device=x.device)
        cos_a = torch.ones(batch_size, device=x.device)

        aligned_x = x
        latent_f = None

        # === ИТЕРАТИВНЫЙ STN ===
        for i in range(self.stn_iters):
            # Прогон через RotateNet (градиенты текут только если RotateNet разморожен)
            sin_cos, latent_f = self.rotatenet(aligned_x)

            s_delta = sin_cos[:, 0]
            c_delta = sin_cos[:, 1]

            # Математическое сложение углов: a_new = a_old + delta
            if i == 0:
                sin_a, cos_a = s_delta, c_delta
            else:
                new_sin = sin_a * c_delta + cos_a * s_delta
                new_cos = cos_a * c_delta - sin_a * s_delta

                # Нормализация для защиты от дрейфа float32
                norm = torch.sqrt(new_sin ** 2 + new_cos ** 2 + 1e-8)
                sin_a, cos_a = new_sin / norm, new_cos / norm

            # Применяем STN всегда к ОРИГИНАЛЬНОМУ x на суммарный угол
            theta = torch.zeros(batch_size, 2, 3, device=x.device)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a

            grid = F.affine_grid(theta, x.size(), align_corners=False)
            aligned_x = F.grid_sample(x, grid, align_corners=False)

        # === ПОЛЯРНОЕ СЛИЯНИЕ ===
        if self.use_polar:
            num_pts = self.rotatenet.out_features // 2
            x_coords = latent_f[:, :num_pts]
            y_coords = latent_f[:, num_pts:]

            r = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2) + 1e-8)
            phi = torch.atan2(y_coords, x_coords)
            latent_f = torch.cat([r, phi], dim=1)

        # === КЛАССИФИКАЦИЯ ===
        blocks = []
        features = aligned_x
        for layer in self.classifier.feature_extractor:
            features = layer(features)
            if isinstance(layer, MyConv2d) and len(blocks) < 2:
                blocks.append(features)

        flat_clf = self.classifier.fc_flat(features)
        fused_features = torch.cat([flat_clf, latent_f], dim=1)
        out = self.classifier.cls_(fused_features)

        if return_extra:
            return out, blocks, aligned_x
        return out