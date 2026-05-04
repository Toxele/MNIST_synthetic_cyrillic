import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from DL.visualization.gradcam import GradCAM


def visualize_classic_gradcam(model, dataset, device, title="Grad-CAM Visualization"):
    """Сетка всех классов в стиле Alpha-Blending (наложение на саму букву)."""
    model.eval()
    samples = {}
    num_classes = len(dataset.char_to_idx)

    for item in dataset:
        img, lbl = item[0], item[1]
        if lbl not in samples: samples[lbl] = img.unsqueeze(0).to(device)
        if len(samples) == num_classes: break

    # Ищем последний сверточный слой
    feat = model.classifier.feature_extractor if hasattr(model, 'classifier') else model.feature_extractor
    target = feat[-2].block[0] if hasattr(feat[-2], 'block') else feat[-2]

    engine = GradCAM(model, target)

    cols = 8
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), facecolor='black')
    axes = axes.flatten()

    for idx in range(num_classes):
        if idx in samples:
            img_t = samples[idx]
            hm = engine(img_t)
            img_np = img_t.cpu().squeeze().numpy()

            # Окрашивание
            hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            img_rgb = (img_np * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)

            overlay = cv2.addWeighted(img_rgb, 0.5, hm_color, 0.5, 0)

            axes[idx].imshow(overlay)
            axes[idx].set_title(dataset.idx_to_char[idx], color='white')
        axes[idx].axis('off')

    engine.remove_hooks()
    plt.suptitle(title, color='white', fontsize=16)
    plt.tight_layout()
    plt.show()