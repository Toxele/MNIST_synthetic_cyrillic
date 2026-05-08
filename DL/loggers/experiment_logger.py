import os
import json
import logging
from typing import Dict, Any


class ExperimentLogger:
    """
    Менеджер эксперимента. Создает структуру директорий,
    сохраняет логи, метрики и конфигурации.
    """

    def __init__(self, experiment_name: str, base_dir: str = "DL/notebooks/"):
        self.exp_dir = os.path.join(base_dir, experiment_name)

        # Определяем пути для сохранения
        self.logs_dir = os.path.join(self.exp_dir, "logs")
        self.plots_loss_dir = os.path.join(self.exp_dir, "plots", "losses")
        self.plots_cm_dir = os.path.join(self.exp_dir, "plots", "confusion_matrices")
        self.plots_gradcam_dir = os.path.join(self.exp_dir, "plots", "gradcams")

        # Создаем директории
        for d in [self.logs_dir, self.plots_loss_dir, self.plots_cm_dir, self.plots_gradcam_dir]:
            os.makedirs(d, exist_ok=True)

        # Настройка текстового логгера
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # Избегаем дублирования хэндлеров при перезапусках ячеек
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(self.logs_dir, f"{experiment_name}.log"), encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)

    def log(self, message: str):
        """Записывает сообщение в файл (дублирование в консоль идет через print в ноутбуке)"""
        self.logger.info(message)

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Сохраняет конфигурацию эксперимента"""
        path = os.path.join(self.exp_dir, filename)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)