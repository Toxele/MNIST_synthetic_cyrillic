import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from DL.loggers.experiment_logger import ExperimentLogger


class RotateNetTrainer:
    """
    Тренер для моделей оценки угла (RotateNet).
    Поддерживает кастомные лоссы регрессии (MSE, AngularCosine).
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            logger: ExperimentLogger = None,
            model_name: str = "RotateNet"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.model_name = model_name

        self.history = {'train_loss': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Обучение {self.model_name}", leave=False)
        for batch in pbar:
            images = batch[0].to(self.device)
            # В датасете с вращениями batch[2] содержит таргет (sin_cos)
            targets = batch[2].to(self.device)

            self.optimizer.zero_grad()

            pred_sin_cos, _ = self.model(images)
            loss = self.criterion(pred_sin_cos, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / total_samples

    def fit(self, train_loader: DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            epoch_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - start_time

            self.history['train_loss'].append(epoch_loss)

            log_str = f"Epoch {epoch}/{epochs}[{epoch_time:.1f}s] - Train Loss: {epoch_loss:.4f}"
            print(log_str)

            if self.logger:
                self.logger.log(log_str)

        return self.history