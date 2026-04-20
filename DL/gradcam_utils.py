import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()

        # Получаем предсказания модели (нужно использовать return_extra=False для совместимости)
        if 'return_extra' in self.model.forward.__code__.co_varnames:
            output = self.model(x, return_extra=False)
        else:
            output = self.model(x)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Global average pooling градиентов
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Взвешивание активаций
        activations = self.activations[0].detach()
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # фоллбэк (не помогло): Если градиенты обнулили карту (черный экран),
        # берем просто среднюю карту активаций (куда реально смотрят фильтры)
        if np.max(heatmap) == 0:
            raw_activations = self.activations[0].detach()
            heatmap = torch.mean(raw_activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) == 0:
                return heatmap  # Если и тут нули, сеть выдала абсолютно пустой тензор

        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (x.size(-1), x.size(-2)))
        return heatmap