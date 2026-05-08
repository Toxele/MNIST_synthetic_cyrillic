import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseGANLoss(ABC):
    """Базовый класс для всех GAN лоссов."""

    @abstractmethod
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Лосс дискриминатора."""
        pass

    @abstractmethod
    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Лосс генератора (в нашем случае - классификатора, генерирующего латентный вектор)."""
        pass

    def apply_d_constraints(self, discriminator: nn.Module):
        """Метод для применения ограничений на веса (например, Weight Clipping для WGAN)."""
        pass