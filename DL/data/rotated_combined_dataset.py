import torch
import math
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from mnist_synthetic.lower_case_letters_generator import LowerCaseLettersGenerator
from mnist_synthetic.generator import NumbersGenerator


class RotatedCombinedDataset(Dataset):
    """
    Универсальный датасет, использующий процедурные генераторы для букв и цифр.
    Вращает изображения на лету, подготавливает целевые значения (sin, cos)
    и опционально применяет перспективные искажения.
    """

    def __init__(self, num_samples: int, seed: int, max_angle: int = 150, apply_perspective: bool = False):
        self.num_samples = num_samples
        self.max_angle = max_angle
        self.apply_perspective = apply_perspective
        self.rng = torch.Generator().manual_seed(seed)

        self.cyrillic_gen = LowerCaseLettersGenerator(seed=seed)
        self.digit_gen = NumbersGenerator(seed=seed)

        self.cyr_chars = self.cyrillic_gen.ALLOWS_SYMBOLS
        self.dig_chars = [str(i) for i in range(10)]

        self.char_to_idx = {char: idx for idx, char in enumerate(self.cyr_chars)}
        for idx, char in enumerate(self.dig_chars):
            self.char_to_idx[char] = idx + len(self.cyr_chars)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Инициализация модуля перспективы (50% шанс применения к картинке)
        if self.apply_perspective:
            self.perspective = T.RandomPerspective(distortion_scale=0.5, p=0.5)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples // 2:
            img_np, label_char = self.cyrillic_gen.generate()
        else:
            img_np, label_char = self.digit_gen.generate()
            label_char = str(label_char)

        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0

        # 1. Применяем перспективу (опционально)
        if self.apply_perspective:
            img_tensor = self.perspective(img_tensor)

        # 2. Применяем случайное вращение
        angle = torch.empty(1).uniform_(-self.max_angle, self.max_angle, generator=self.rng).item()
        img_rotated = TF.rotate(img_tensor, angle)

        # 3. Подготавливаем таргеты для RotateNet
        angle_rad = math.radians(angle)
        sin_cos_target = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)

        return img_rotated, self.char_to_idx[label_char], sin_cos_target