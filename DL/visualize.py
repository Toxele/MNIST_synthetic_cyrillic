import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
from sklearn.metrics import confusion_matrix
from scipy.ndimage import zoom

from DL.gradcam_utils import GradCAM, OverlayGradCAM


def apply_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device,
                  layer_idx: int) -> torch.Tensor:
    target_layer = model.feature_extractor[layer_idx]
    cam = GradCAM(model=model, target_layer=target_layer)
    heatmap = cam(image_tensor.unsqueeze(0).to(device))

    # Удаляем хуки, чтобы не засорять модель
    cam.remove_hooks()
    return heatmap


def show_batch(dataset, num_samples: int = 25):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(dataset):
            break
        img, label_idx = dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(dataset.idx_to_char[label_idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_all_classes_gradcam(model_base: torch.nn.Module, model_bg: torch.nn.Module, dataset,
                                  device: torch.device):
    class_samples = {}
    for img, lbl in dataset:
        if lbl not in class_samples:
            class_samples[lbl] = img
        if len(class_samples) == len(dataset.char_to_idx):
            break

    num_classes = len(class_samples)
    cols_per_row = 2
    rows = math.ceil(num_classes / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row * 3, figsize=(16, rows * 3.5))

    for idx, (label_idx, img_tensor) in enumerate(class_samples.items()):
        row = idx // cols_per_row
        col_offset = (idx % cols_per_row) * 3

        char_name = dataset.idx_to_char[label_idx]

        hm_base = apply_gradcam(model_base, img_tensor, device, layer_idx=2)
        hm_bg = apply_gradcam(model_bg, img_tensor, device, layer_idx=2)

        axes[row, col_offset].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, col_offset].set_title(f"Символ: '{char_name}'")
        axes[row, col_offset].axis('off')

        axes[row, col_offset + 1].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, col_offset + 1].imshow(hm_base, cmap='jet', alpha=0.5)
        axes[row, col_offset + 1].set_title("Base Grad-CAM")
        axes[row, col_offset + 1].axis('off')

        axes[row, col_offset + 2].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, col_offset + 2].imshow(hm_bg, cmap='jet', alpha=0.5)
        axes[row, col_offset + 2].set_title("BG Loss Grad-CAM")
        axes[row, col_offset + 2].axis('off')

    for col in range(num_classes * 3, rows * cols_per_row * 3):
        r = col // (cols_per_row * 3)
        c = col % (cols_per_row * 3)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


def plot_single_confusion_matrix(all_labels: list, all_preds: list, chars: list, title: str):
    cm = confusion_matrix(all_labels, all_preds)

    # Создаем одну большую фигуру для матрицы
    plt.figure(figsize=(14, 12))

    # annot_kws управляет размером шрифта цифр внутри ячеек
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=chars, yticklabels=chars,
                cbar=False, annot_kws={"size": 10})

    plt.title(title, fontsize=16, pad=15)
    plt.ylabel("Истинный класс", fontsize=14, labelpad=10)
    plt.xlabel("Предсказанный класс", fontsize=14, labelpad=10)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(model_base: torch.nn.Module, model_bg: torch.nn.Module, dataloader, dataset,
                            device: torch.device):
    model_base.eval()
    model_bg.eval()

    all_labels = []
    base_preds = []
    bg_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            out_base = model_base(images, return_extra=False)
            out_bg = model_bg(images, return_extra=False)

            _, p_base = torch.max(out_base, 1)
            _, p_bg = torch.max(out_bg, 1)

            all_labels.extend(labels.cpu().numpy())
            base_preds.extend(p_base.cpu().numpy())
            bg_preds.extend(p_bg.cpu().numpy())

    chars = [dataset.idx_to_char[i] for i in range(len(dataset.char_to_idx))]

    # Вызываем отрисовку дважды, чтобы они были в разных окнах по вертикали
    plot_single_confusion_matrix(all_labels, base_preds, chars, "Матрица ошибок: Baseline")
    plot_single_confusion_matrix(all_labels, bg_preds, chars, "Матрица ошибок: С Background Loss")


def visualize_failure_cases(model: torch.nn.Module, dataloader, dataset, device: torch.device, num_cases: int = 5):
    model.eval()
    failures = []

    with torch.no_grad():
        for images, labels in dataloader:
            images_device = images.to(device)
            outputs = model(images_device, return_extra=False)
            _, preds = torch.max(outputs, 1)

            mask = preds != labels.to(device)

            if mask.any():
                failed_imgs = images[mask.cpu()]
                failed_preds = preds[mask].cpu()
                true_labels = labels[mask.cpu()]

                for img, p, t in zip(failed_imgs, failed_preds, true_labels):
                    failures.append((img, p.item(), t.item()))
                    if len(failures) >= num_cases:
                        break
            if len(failures) >= num_cases:
                break

    if not failures:
        print("Ошибок не найдено!")
        return

    fig, axes = plt.subplots(len(failures), 2, figsize=(8, len(failures) * 3))
    if len(failures) == 1:
        axes = [axes]

    for idx, (img_tensor, pred_idx, true_idx) in enumerate(failures):
        true_char = dataset.idx_to_char[true_idx]
        pred_char = dataset.idx_to_char[pred_idx]

        hm = apply_gradcam(model, img_tensor, device, layer_idx=2)

        axes[idx][0].imshow(img_tensor.squeeze(), cmap='gray')
        axes[idx][0].set_title(f"Истина: {true_char}")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(img_tensor.squeeze(), cmap='gray')
        axes[idx][1].imshow(hm, cmap='jet', alpha=0.5)
        axes[idx][1].set_title(f"Предсказано: {pred_char} (Grad-CAM)")
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_multilayer_gradcam(model_base: torch.nn.Module, model_bg: torch.nn.Module, dataset, device: torch.device,
                                 num_samples: int = 6):
    """
    Визуализирует работу Grad-CAM на разных слоях сети.
    Автоматически адаптируется под глубину модели (1 или 2 сверточных слоя).
    """
    class_samples = {}
    for img, lbl in dataset:
        if lbl not in class_samples:
            class_samples[lbl] = img
        if len(class_samples) == num_samples:
            break

    fig, axes = plt.subplots(num_samples, 5, figsize=(18, num_samples * 3.5))
    if num_samples == 1:
        axes = [axes]

    # Проверяем, есть ли в модели второй сверточный слой (индекс 2)
    has_two_layers = len(model_base.feature_extractor) > 2

    for row, (label_idx, img_tensor) in enumerate(class_samples.items()):
        char_name = dataset.idx_to_char[label_idx]

        # Слой 1 всегда есть (index 0)
        hm_base_l1 = apply_gradcam(model_base, img_tensor, device, layer_idx=0)
        hm_bg_l1 = apply_gradcam(model_bg, img_tensor, device, layer_idx=0)

        if has_two_layers:
            hm_base_l2 = apply_gradcam(model_base, img_tensor, device, layer_idx=2)
            hm_bg_l2 = apply_gradcam(model_bg, img_tensor, device, layer_idx=2)

        # 0. Оригинал
        axes[row, 0].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, 0].set_title(f"Оригинал: '{char_name}'")
        axes[row, 0].axis('off')

        # 1. Base Conv1
        axes[row, 1].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, 1].imshow(hm_base_l1, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].set_title("Base Conv1 (Локальные)")
        axes[row, 1].axis('off')

        # 2. Base Conv2 (если есть)
        if has_two_layers:
            axes[row, 2].imshow(img_tensor.squeeze(), cmap='gray')
            axes[row, 2].imshow(hm_base_l2, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[row, 2].set_title("Base Conv2 (Паттерны)")
        else:
            axes[row, 2].text(0.5, 0.5, 'Слой отсутствует\n(2KB Модель)',
                              ha='center', va='center', fontsize=12)
        axes[row, 2].axis('off')

        # 3. BG Loss Conv1
        axes[row, 3].imshow(img_tensor.squeeze(), cmap='gray')
        axes[row, 3].imshow(hm_bg_l1, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[row, 3].set_title("BG Loss Conv1 (Локальные)")
        axes[row, 3].axis('off')

        # 4. BG Loss Conv2 (если есть)
        if has_two_layers:
            axes[row, 4].imshow(img_tensor.squeeze(), cmap='gray')
            axes[row, 4].imshow(hm_bg_l2, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[row, 4].set_title("BG Loss Conv2 (Паттерны)")
        else:
            axes[row, 4].text(0.5, 0.5, 'Слой отсутствует\n(2KB Модель)',
                              ha='center', va='center', fontsize=12)
        axes[row, 4].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_overlay_gradcam(model: torch.nn.Module, dataset, device: torch.device, title: str):
    """
    Визуализация Grad-CAM методом "наложения" (overlay) с использованием cv2.addWeighted.
    Показывает один пример для каждого класса.
    """
    model.eval()
    samples_per_class = {}
    num_classes = len(dataset.char_to_idx)

    for img, label_idx in dataset:
        if label_idx not in samples_per_class:
            samples_per_class[label_idx] = img.unsqueeze(0).to(device)
        if len(samples_per_class) == num_classes:
            break

    # Динамически берем последний сверточный блок (Conv2d)
    target_layer = model.feature_extractor[-2].block[0]
    grad_cam = OverlayGradCAM(model, target_layer)

    cols = 8
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for class_idx in range(num_classes):
        if class_idx in samples_per_class:
            img_tensor = samples_per_class[class_idx]
            cam, pred_class = grad_cam.generate_cam(img_tensor)
            img = img_tensor.cpu().detach().numpy()[0, 0]

            # Избегаем деления на ноль, если размерности нулевые
            if cam.shape[0] > 0 and cam.shape[1] > 0:
                cam_resized = zoom(cam, (28 / cam.shape[0], 28 / cam.shape[1]))
            else:
                cam_resized = np.zeros((28, 28))

            cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted((img * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), 0.5,
                                      cam_colored, 0.5, 0)

            true_char = dataset.idx_to_char[class_idx]
            pred_char = dataset.idx_to_char[pred_class]

            ax = axes[class_idx]
            ax.imshow(overlay)

            title_color = 'white' if true_char == pred_char else 'red'
            ax.set_title(f"{true_char} (Pred: {pred_char})", fontsize=12, color=title_color)
            ax.axis('off')

    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')

    grad_cam.remove_hooks()
    plt.suptitle(title, fontsize=16, color='white', y=1.02)

    # Настраиваем цвет фона для фигуры, чтобы текст был читаем
    fig.patch.set_facecolor('black')

    plt.tight_layout()
    plt.show()