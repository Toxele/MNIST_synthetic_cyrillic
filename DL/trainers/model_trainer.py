import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Optional
from DL.losses.background_layers_loss import background_layers_loss
from DL.layers.latent_discriminator import LatentDiscriminator
from DL.loggers.experiment_logger import ExperimentLogger
from DL.visualization.visualizer import ExperimentVisualizer
from DL.losses.gan.base import BaseGANLoss

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
            gan_loss_fn: Optional[BaseGANLoss] = None,
            logger: Optional[ExperimentLogger] = None,
            visualizer: Optional[ExperimentVisualizer] = None,
            model_name: str = "model"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.use_background_loss = use_background_loss
        self.bg_loss_alpha = bg_loss_alpha
        self.bg_loss_n = bg_loss_n

        self.gan_loss_fn = gan_loss_fn

        self.logger = logger
        self.visualizer = visualizer
        self.model_name = model_name

        if self.gan_loss_fn is not None:
            clf = self.model.classifier if hasattr(self.model, 'classifier') else self.model
            latent_dim = clf.cls_.in_features if hasattr(clf.cls_, 'in_features') else clf.cls_.weight.shape[1]
            self.discriminator = LatentDiscriminator(latent_dim).to(device)
            self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        self.history = {'train_loss': [], 'train_acc': [], 'train_bg_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        if self.gan_loss_fn is not None:
            self.discriminator.train()

        running_loss, running_bg_loss, correct_preds, total_preds = 0.0, 0.0, 0, 0
        pbar = tqdm(dataloader, desc="Обучение", leave=False)

        for batch in pbar:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            batch_size = images.size(0)

            # Шаг GAN-дискриминатора (если есть)
            if self.gan_loss_fn is not None:
                self.opt_disc.zero_grad()

                if hasattr(self.model, 'classifier'):
                    features = self.model.classifier.feature_extractor(images)
                    latent = self.model.classifier.fc_flat(features)
                else:
                    features = self.model.feature_extractor(images)
                    latent = self.model.fc_flat(features)

                real_latent = torch.randn_like(latent).to(self.device)

                # ООП вызов лосса дискриминатора
                d_real = self.discriminator(real_latent)
                d_fake = self.discriminator(latent.detach())

                loss_d = self.gan_loss_fn.d_loss(d_real, d_fake)
                loss_d.backward()
                self.opt_disc.step()

                # ООП применение ограничений (например, для WGAN)
                self.gan_loss_fn.apply_d_constraints(self.discriminator)

            # Шаг основной модели
            self.optimizer.zero_grad()
            forward_res = self.model(images, labels=labels, return_extra=True)

            if len(forward_res) == 3:
                outputs, blocks, bg_input_images = forward_res
            else:
                outputs, blocks = forward_res
                bg_input_images = images

            loss_main = self.criterion(outputs, labels)
            loss = loss_main

            if self.use_background_loss:
                f_ext = self.model.feature_extractor if hasattr(self.model,
                                                                'feature_extractor') else self.model.classifier.feature_extractor
                loss_bg = background_layers_loss(bg_input_images, blocks, f_ext, self.bg_loss_alpha, self.bg_loss_n)
                loss = loss + loss_bg
                running_bg_loss += loss_bg.item() * batch_size

            if self.gan_loss_fn is not None:
                # ООП вызов лосса генератора
                d_fake_for_g = self.discriminator(latent)
                loss_g = 0.1 * self.gan_loss_fn.g_loss(d_fake_for_g)
                loss = loss + loss_g

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss_main.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_preds += batch_size
            correct_preds += (predicted == labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {'loss': running_loss / total_preds, 'acc': correct_preds / total_preds,
                'bg_loss': running_bg_loss / total_preds}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss, correct_preds, total_preds = 0.0, 0, 0

        for batch in dataloader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(images, labels=labels, return_extra=False)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        return {'loss': running_loss / total_preds, 'acc': correct_preds / total_preds}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            epoch_time = time.time() - start_time

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_bg_loss'].append(train_metrics['bg_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            # Форматируем строку лога
            log_str = (f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s] - "
                       f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, BG: {train_metrics['bg_loss']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")

            # Выводим в консоль и пишем в файл
            print(log_str)
            if self.logger:
                self.logger.log(log_str)

            # Обновляем живой график
            if self.visualizer:
                self.visualizer.plot_live_losses(self.history, self.model_name, epoch, epochs)

        return self.history