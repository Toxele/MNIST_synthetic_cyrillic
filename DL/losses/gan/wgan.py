import torch
import torch.nn as nn
from DL.losses.gan.base import BaseGANLoss


class WGANLoss(BaseGANLoss):
    """Wasserstein GAN Loss. Требует обрезки весов дискриминатора."""

    def __init__(self, clip_value: float = 0.01):
        self.clip_value = clip_value

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        # Дискриминатор максимизирует разницу (поэтому возвращаем с минусом для минимизатора)
        return -real_logits.mean() + fake_logits.mean()

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def apply_d_constraints(self, discriminator: nn.Module):
        # Weight Clipping (обязательное условие для WGAN)
        with torch.no_grad():
            for param in discriminator.parameters():
                param.clamp_(-self.clip_value, self.clip_value)