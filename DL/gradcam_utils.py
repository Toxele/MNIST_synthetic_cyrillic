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

        # Сохраняем ссылки на хуки, чтобы потом их удалить
        self.hook_handles = []

        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Важно вызывать эту функцию после получения тепловой карты"""
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()

        if 'return_extra' in self.model.forward.__code__.co_varnames:
            output = self.model(x, return_extra=False)
        else:
            output = self.model(x)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        activations = self.activations[0].detach()
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) == 0:
            raw_activations = self.activations[0].detach()
            heatmap = torch.mean(raw_activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) == 0:
                heatmap = cv2.resize(heatmap, (x.size(-1), x.size(-2)))
                return heatmap

        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (x.size(-1), x.size(-2)))
        return heatmap