import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List
from DL.visualization.gradcam import GradCAM


def plot_confusion_matrices(model_base, model_bg, dataloader, dataset, device):
    """Отрисовка матриц ошибок с безопасной распаковкой батча."""
    model_base.eval()
    model_bg.eval()
    all_labels, base_preds, bg_preds = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1]  # Берем только метку, игнорируем sin/cos

            out_base = model_base(images, return_extra=False)
            out_bg = model_bg(images, return_extra=False)

            base_preds.extend(out_base.argmax(dim=1).cpu().numpy())
            bg_preds.extend(out_bg.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    chars = [dataset.idx_to_char[i] for i in range(len(dataset.char_to_idx))]

    for preds, title in [(base_preds, "Baseline"), (bg_preds, "BG Loss / Cascade")]:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=chars, yticklabels=chars, cbar=False)
        plt.title(f"Матрица ошибок: {title}")
        plt.show()


def visualize_multilayer_gradcam(model_base, model_bg, dataset, device, num_samples=5):
    """Визуализация Conv1 и Conv2 (5 колонок)."""
    samples = {}
    for item in dataset:
        img, lbl = item[0], item[1]
        if lbl not in samples: samples[lbl] = img
        if len(samples) == num_samples: break

    # Проверяем архитектуру для выбора слоев
    def get_layers(m):
        f = m.classifier.feature_extractor if hasattr(m, 'classifier') else m.feature_extractor
        return [f[0], f[2]] if len(f) > 2 else [f[0], None]

    fig, axes = plt.subplots(num_samples, 5, figsize=(18, num_samples * 3.5))

    for i, (lbl, img_tensor) in enumerate(samples.items()):
        char = dataset.idx_to_char[lbl]
        axes[i, 0].imshow(img_tensor.squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Оригинал: {char}")

        for j, m in enumerate([model_base, model_bg]):
            layers = get_layers(m)
            for k, layer in enumerate(layers):
                ax = axes[i, 1 + j * 2 + k]
                if layer is None:
                    ax.text(0.5, 0.5, "Нет слоя", ha='center')
                else:
                    engine = GradCAM(m, layer)
                    hm = engine(img_tensor.unsqueeze(0).to(device))
                    engine.remove_hooks()
                    ax.imshow(img_tensor.squeeze(), cmap='gray')
                    ax.imshow(hm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                ax.axis('off')
    plt.tight_layout()
    plt.show()