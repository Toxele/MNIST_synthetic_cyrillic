import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLayer(nn.Module):
    """
    Стабильная реализация ArcFace.
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