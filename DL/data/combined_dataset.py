import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from mnist_synthetic.torch.datasets import MNISTSynthetic
from DL.data.synthetic_cyrillic_dataset import SyntheticCyrillicDataset


class CombinedDataset(Dataset):
    """Датасет, объединяющий 33 буквы кириллицы и 10 цифр (43 класса)."""

    def __init__(self, num_samples: int, seed: int, transform=None):
        self.num_samples = num_samples
        self.transform = transform

        self.cyrillic_dataset = SyntheticCyrillicDataset(num_samples=num_samples // 2, seed=seed, transform=None)
        self.mnist_dataset = MNISTSynthetic(num_samples // 2, seed=seed)

        self.char_to_idx = {**self.cyrillic_dataset.char_to_idx,
                            **{str(i): i + len(self.cyrillic_dataset.char_to_idx) for i in range(10)}}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples // 2:
            img, label_idx = self.cyrillic_dataset[idx]
        else:
            img, label = self.mnist_dataset[idx - self.num_samples // 2]

            if isinstance(img, Image.Image):
                img = TF.to_tensor(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                if img.ndim == 2: img = img.unsqueeze(0)
                if img.max() > 1.0: img /= 255.0
            elif isinstance(img, torch.Tensor):
                if img.ndim == 2: img = img.unsqueeze(0)
                if img.max() > 1.0: img /= 255.0

            label_idx = self.char_to_idx[str(label)]

        if self.transform:
            img = self.transform(img)

        return img, label_idx