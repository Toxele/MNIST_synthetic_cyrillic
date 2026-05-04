import torch
import math
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from mnist_synthetic.lower_case_letters_generator import LowerCaseLettersGenerator


class RotatedCyrillicDataset(Dataset):
    """Датасет для генерации символов с жестко заданным углом поворота (для CascadeNet)."""

    def __init__(self, num_samples: int, seed: int, max_angle: int = 150):
        self.num_samples = num_samples
        self.generator = LowerCaseLettersGenerator(seed=seed)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.generator.ALLOWS_SYMBOLS)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.max_angle = max_angle
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_np, label_char = self.generator.generate()
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0

        angle = torch.empty(1).uniform_(-self.max_angle, self.max_angle, generator=self.rng).item()
        img_rotated = TF.rotate(img_tensor, angle)

        angle_rad = math.radians(angle)
        sin_cos_target = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)

        return img_rotated, self.char_to_idx[label_char], sin_cos_target