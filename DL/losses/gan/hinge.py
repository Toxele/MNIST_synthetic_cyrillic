import torch
import torch.nn.functional as F
from DL.losses.gan.base import BaseGANLoss

class HingeLoss(BaseGANLoss):
    """Hinge GAN Loss. Использует Max-Margin подход."""
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()