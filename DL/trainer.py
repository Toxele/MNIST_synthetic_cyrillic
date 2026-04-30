import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Any, Optional

from DL.losses import background_layers_loss


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            use_background_loss: bool = False,
            bg_loss_alpha: float = 1.0,
            bg_loss_n: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.use_background_loss = use_background_loss
        self.bg_loss_alpha = bg_loss_alpha
        self.bg_loss_n = bg_loss_n

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_bg_loss': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_bg_loss = 0.0
        correct_preds = 0
        total_preds = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_background_loss:
                outputs, blocks = self.model(images, return_extra=True)
                loss_main = self.criterion(outputs, labels)
                loss_bg = background_layers_loss(
                    images,
                    blocks,
                    feature_extractor=self.model.feature_extractor,
                    alpha=self.bg_loss_alpha,
                    n=self.bg_loss_n
                )
                loss = loss_main + loss_bg
                running_bg_loss += loss_bg.item() * images.size(0)
            else:
                outputs = self.model(images, return_extra=False)
                loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds
        epoch_bg_loss = running_bg_loss / total_preds if self.use_background_loss else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc, 'bg_loss': epoch_bg_loss}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images, return_extra=False)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_bg_loss'].append(train_metrics['bg_loss'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            print(
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | BG Loss: {train_metrics['bg_loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}\n")

        return self.history