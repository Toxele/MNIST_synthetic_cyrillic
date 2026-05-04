import torch
import math


def angular_cosine_loss(pred_sin_cos: torch.Tensor, target_sin_cos: torch.Tensor,
                        limit_deg: float = 150.0) -> torch.Tensor:
    """
    Геометрический лосс для углов поворота на многообразии SO(2).
    """
    # 1. Скалярное произведение (Косинусное расстояние между предсказанным и целевым углом)
    # 1 - cos(delta_alpha)
    cos_sim = (pred_sin_cos * target_sin_cos).sum(dim=1)
    loss_angle = 1.0 - cos_sim.mean()

    # 2. Penalty Limit (Штраф за выход за пределы допустимых углов)
    # Вычисляем предсказанный угол в радианах
    pred_angles = torch.atan2(pred_sin_cos[:, 0], pred_sin_cos[:, 1])
    limit_rad = math.radians(limit_deg)

    # Релу срежет все, что меньше лимита. Все, что больше - возведет в квадрат.
    penalty = torch.relu(torch.abs(pred_angles) - limit_rad).pow(2).mean()

    return loss_angle + penalty