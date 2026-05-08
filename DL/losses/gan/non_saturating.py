import torch
import torch.nn.functional as F
from DL.losses.gan.base import BaseGANLoss

class NonSaturatingLoss(BaseGANLoss):
    """Классический ненасыщающийся лосс (Vanilla GAN с эвристикой)."""
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) / 2.0

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))