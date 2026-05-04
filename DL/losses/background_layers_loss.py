import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List


def background_layers_loss(x: torch.Tensor, blocks: List[torch.Tensor], feature_extractor: nn.Module, alpha: float,
                           n: int = 1) -> torch.Tensor:
    """Вычисляет пространственный штраф за активацию фильтров на фоне."""
    batch_size = x.shape[0]
    loss = torch.tensor(0.0, device=x.device)

    digit_mask = (x >= 0.01).float()

    if n > 0:
        digit_mask = F.max_pool2d(digit_mask, kernel_size=n * 2 + 1, stride=1, padding=n)

    current_mask = digit_mask
    block_idx = 0

    for layer in feature_extractor:
        layer_name = layer.__class__.__name__

        if layer_name == 'MyConv2d':
            conv = layer.block[0]
            k = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
            p = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
            s = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)

            current_mask = F.max_pool2d(current_mask, kernel_size=k, stride=s, padding=p)

            if block_idx < len(blocks):
                bg_mask = 1.0 - current_mask
                block_tensor = blocks[block_idx]

                spatial_size = block_tensor.shape[2] * block_tensor.shape[3]
                loss += (block_tensor * bg_mask).sum() / (batch_size * spatial_size)
                block_idx += 1

        elif layer_name == 'MaxPool2d':
            k = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            p = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            s = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)

            current_mask = F.max_pool2d(current_mask, kernel_size=k, stride=s, padding=p)

    return alpha * loss