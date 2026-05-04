import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLayer(nn.Module):
    """
    CosFace (Large Margin Cosine Loss).
    Математически пуленепробиваемая альтернатива ArcFace.
    Вычитает margin прямо из косинуса, избегая ЛЮБЫХ тригонометрических
    преобразований. Не может взорваться в принципе.
    (معбычно m для CosFace берут чуть больше, например 0.35)
    """

    def __init__(self, in_features: int, out_features: int, s: float = 15.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)

        cosine = F.linear(x_norm, w_norm)

        if not self.training or labels is None:
            return cosine * self.s

        # Просто итерируем margin из косинуса правильного класса
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Вычитаем margin (косинус падает -> сеть штрафуется -> классы расталкиваются)
        output = cosine - (one_hot * self.m)

        return output * self.s