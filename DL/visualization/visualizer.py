import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import clear_output
from DL.loggers.experiment_logger import ExperimentLogger
from DL.visualization.classic_gradcam import visualize_classic_gradcam


class ExperimentVisualizer:
    """
    Объект для инкапсуляции всей логики визуализации.
    Автоматически сохраняет графики в нужные папки логгера и выводит их в output.
    """

    def __init__(self, logger: ExperimentLogger):
        self.logger = logger

    def plot_live_losses(self, history: dict, model_name: str, current_epoch: int, total_epochs: int):
        """
        Динамическая отрисовка графиков обучения.
        """
        from IPython.display import clear_output
        clear_output(wait=True)

        # Настраиваем стиль
        sns.set_theme(style="ticks")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Настройка цветов
        c_train = '#4C72B0'  # Приглушенный синий
        c_val = '#DD8452'  # Приглушенный оранжевый

        # График Loss
        axes[0].plot(history['train_loss'], label='Train Loss', color=c_train, linewidth=2, marker='o', markersize=6,
                     alpha=0.9)
        axes[0].plot(history['val_loss'], label='Val Loss', color=c_val, linewidth=2, marker='s', markersize=6,
                     alpha=0.9)
        axes[0].set_title(f"{model_name}: Loss (Epoch {current_epoch}/{total_epochs})", fontsize=14, pad=10)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].legend(frameon=True, shadow=True)
        axes[0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # График Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc', color=c_train, linewidth=2, marker='o', markersize=6,
                     alpha=0.9)
        axes[1].plot(history['val_acc'], label='Val Acc', color=c_val, linewidth=2, marker='s', markersize=6, alpha=0.9)
        axes[1].set_title(f"{model_name}: Accuracy", fontsize=14, pad=10)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].legend(frameon=True, shadow=True)
        axes[1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        sns.despine()  # Убираем лишние рамки
        plt.tight_layout()

        save_path = os.path.join(self.logger.plots_loss_dir, f"loss_{model_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Сохраняем в высоком качестве
        plt.show()


    def plot_losses(self, history: dict, model_name: str):
        """Отрисовка и сохранение графиков обучения (Loss и Accuracy)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # График Loss
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0].set_title(f"{model_name}: Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # График Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
        axes[1].plot(history['val_acc'], label='Val Acc', color='red')
        axes[1].set_title(f"{model_name}: Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.logger.plots_loss_dir, f"loss_{model_name}.png")
        plt.savefig(save_path)
        plt.show()

    def plot_confusion_matrix(self, model: torch.nn.Module, dataloader, dataset, device: torch.device, model_name: str):
        """Отрисовка матрицы ошибок для одной модели"""
        model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(device)
                labels = batch[1]

                outputs = model(images, return_extra=False)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())

        chars = [dataset.idx_to_char[i] for i in range(len(dataset.char_to_idx))]
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=chars, yticklabels=chars, cbar=False,
                    annot_kws={"size": 10})

        title = f"Матрица ошибок: {model_name}"
        plt.title(title, fontsize=16, pad=15)
        plt.ylabel("Истинный класс", fontsize=14)
        plt.xlabel("Предсказанный класс", fontsize=14)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11, rotation=0)

        plt.tight_layout()
        save_path = os.path.join(self.logger.plots_cm_dir, f"cm_{model_name}.png")
        plt.savefig(save_path)
        plt.show()

    def plot_gradcam(self, model: torch.nn.Module, dataset, device: torch.device, model_name: str):
        """Вызов классического Grad-CAM и сохранение графика"""
        # Эта функция уже отрисовывает и делает plt.show() внутри себя.
        # Мы адаптируем вызов так, чтобы можно было сохранить результат.

        # Используем существующую логику
        visualize_classic_gradcam(model, dataset, device, f"Grad-CAM: {model_name}")

        # Получаем текущую активную фигуру и сохраняем ее
        fig = plt.gcf()
        save_path = os.path.join(self.logger.plots_gradcam_dir, f"gradcam_{model_name}.png")
        fig.savefig(save_path, facecolor='black')