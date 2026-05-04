import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Универсальный движок для вычисления тепловых карт Grad-CAM.
    Поддерживает фоллбэк на сырые активации, если градиенты обнулились.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Обязательно вызывать после завершения визуализации."""
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()

        # Обработка входа (учитываем return_extra у нашей модели)
        if 'return_extra' in self.model.forward.__code__.co_varnames:
            output = self.model(x, return_extra=False)
        else:
            output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Математика Grad-CAM
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))

        heatmap = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * acts[i]

        heatmap = np.maximum(heatmap, 0)  # ReLU для тепловой карты

        # Защита от пустых экранов (если градиенты обнулили карту)
        if np.max(heatmap) == 0:
            heatmap = np.mean(acts, axis=0)
            heatmap = np.maximum(heatmap, 0)

        # Нормализация и ресайз до размера входа
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (x.size(-1), x.size(-2)))
        return heatmap