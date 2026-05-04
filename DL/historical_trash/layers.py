import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLayer(nn.Module):
    """
    Ультра-стабильная реализация ArcFace.
    Использует прямое вычисление через acos с жестким клиппингом (clamp 1e-4),
    что гарантированно спасает от бесконечных градиентов при идеальном совпадении.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 15.0, m: float = 0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Нормализация
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)

        # Вычисление косинусного сходства
        cosine = F.linear(x_norm, w_norm)

        # Инференс: просто возвращаем масштабированные логиты
        if not self.training or labels is None:
            return cosine * self.s

        # ===============================================================
        # ГЛАВНЫЙ СЕКРЕТ СТАБИЛЬНОСТИ: ЖЕСТКИЙ CLAMP
        # Отступаем от краев на 1e-4. В float32 этого достаточно,
        # чтобы градиент arccos никогда не улетел в бесконечность.
        # ===============================================================
        cosine = cosine.clamp(-1.0 + 1e-4, 1.0 - 1e-4)

        # Переход к честным углам (наиболее читаемый и стабильный метод)
        theta = torch.acos(cosine)

        # Добавляем margin (угловой отступ) к правильному классу
        target_logits = torch.cos(theta + self.m)

        # Собираем финальные логиты
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = cosine * (1 - one_hot) + target_logits * one_hot

        # Масштабируем перед CrossEntropy
        return output * self.s


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


# Оставляем здесь наш LatentDiscriminator из предыдущего шага
class LatentDiscriminator(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
            # УБРАЛИ nn.Sigmoid()! Теперь дискриминатор выдает сырые логиты (от -inf до +inf)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)