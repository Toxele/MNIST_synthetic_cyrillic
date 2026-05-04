import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
from DL.losses import background_layers_loss
from DL.layers import LatentDiscriminator


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            use_background_loss: bool = False,
            bg_loss_alpha: float = 1.0,
            bg_loss_n: int = 1,
            use_gan_loss: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.use_background_loss = use_background_loss
        self.bg_loss_alpha = bg_loss_alpha
        self.bg_loss_n = bg_loss_n

        self.use_gan_loss = use_gan_loss
        if self.use_gan_loss:
            # Динамическое определение размерности для дискриминатора
            if hasattr(self.model, 'classifier'):
                # Для CascadeNet
                clf = self.model.classifier
            else:
                # Для обычных моделей
                clf = self.model

            latent_dim = clf.cls_.in_features if hasattr(clf.cls_, 'in_features') else clf.cls_.weight.shape[1]
            self.discriminator = LatentDiscriminator(latent_dim).to(device)
            self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
            self.bce_loss = nn.BCEWithLogitsLoss()

        self.history = {'train_loss': [], 'train_acc': [], 'train_bg_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        if self.use_gan_loss:
            self.discriminator.train()

        running_loss = 0.0
        running_bg_loss = 0.0
        correct_preds = 0
        total_preds = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            batch_size = images.size(0)

            # === 1. ШАГ GAN-ДИСКРИМИНАТОРА (если включен) ===
            if self.use_gan_loss:
                self.opt_disc.zero_grad()
                # Извлекаем латентный вектор (зависит от архитектуры)
                if hasattr(self.model, 'classifier'):
                    features = self.model.classifier.feature_extractor(images)
                    latent = self.model.classifier.fc_flat(features)
                else:
                    features = self.model.feature_extractor(images)
                    latent = self.model.fc_flat(features)

                real_latent = torch.randn_like(latent).to(self.device)
                loss_d = (self.bce_loss(self.discriminator(real_latent),
                                        torch.ones(batch_size, 1, device=self.device)) +
                          self.bce_loss(self.discriminator(latent.detach()),
                                        torch.zeros(batch_size, 1, device=self.device))) / 2
                loss_d.backward()
                self.opt_disc.step()

            # === 2. ШАГ ОСНОВНОЙ МОДЕЛИ ===
            self.optimizer.zero_grad()

            # Универсальный проброс: модель может вернуть (out, blocks) или (out, blocks, aligned_images)
            forward_res = self.model(images, labels=labels, return_extra=True)

            if len(forward_res) == 3:
                outputs, blocks, bg_input_images = forward_res
            else:
                outputs, blocks = forward_res
                bg_input_images = images  # Если модель не Cascade, используем оригинал

            loss_main = self.criterion(outputs, labels)
            loss = loss_main

            if self.use_background_loss:
                # Берем feature_extractor в зависимости от типа модели
                f_ext = self.model.feature_extractor if hasattr(self.model,
                                                                'feature_extractor') else self.model.classifier.feature_extractor

                loss_bg = background_layers_loss(bg_input_images, blocks, f_ext, self.bg_loss_alpha, self.bg_loss_n)
                loss = loss + loss_bg
                running_bg_loss += loss_bg.item() * batch_size

            if self.use_gan_loss:
                # Повторный проход для латентного вектора не нужен, берем из логики выше если нужно
                # Но для простоты используем текущие активации
                loss_g = 0.1 * self.bce_loss(self.discriminator(latent), torch.ones(batch_size, 1, device=self.device))
                loss = loss + loss_g

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss_main.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_preds += batch_size
            correct_preds += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss_main.item()})

        return {'loss': running_loss / total_preds, 'acc': correct_preds / total_preds,
                'bg_loss': running_bg_loss / total_preds}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch in dataloader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            # При инференсе CascadeNet возвращает только 1 значение (логиты)
            outputs = self.model(images, labels=labels, return_extra=False)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        return {'loss': running_loss / total_preds, 'acc': correct_preds / total_preds}

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