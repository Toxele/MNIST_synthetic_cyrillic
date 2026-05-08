import torch
import torch.nn.functional as F
from DL.losses.gan.base import BaseGANLoss

class LSGANLoss(BaseGANLoss):
    """Least Squares GAN Loss (MSE). Сглаживает градиенты."""
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
        fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) / 2.0

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))