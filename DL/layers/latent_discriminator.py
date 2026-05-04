import torch
import torch.nn as nn


class LatentDiscriminator(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)