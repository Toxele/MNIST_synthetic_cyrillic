import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from DL.gradcam_utils import GradCAM


def apply_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    target_layer = model.feature_extractor[-2]
    cam = GradCAM(model=model, target_layer=target_layer)
    heatmap = cam(image_tensor.unsqueeze(0).to(device))
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

        hm_base = apply_gradcam(model_base, img_tensor, device)
        hm_bg = apply_gradcam(model_bg, img_tensor, device)

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

        hm = apply_gradcam(model, img_tensor, device)

        axes[idx][0].imshow(img_tensor.squeeze(), cmap='gray')
        axes[idx][0].set_title(f"Истина: {true_char}")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(img_tensor.squeeze(), cmap='gray')
        axes[idx][1].imshow(hm, cmap='jet', alpha=0.5)
        axes[idx][1].set_title(f"Предсказано: {pred_char} (Grad-CAM)")
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()