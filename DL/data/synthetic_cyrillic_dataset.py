import torch
from torch.utils.data import Dataset
from mnist_synthetic.lower_case_letters_generator import LowerCaseLettersGenerator


class SyntheticCyrillicDataset(Dataset):
    """Датасет для генерации синтетической кириллицы (33 класса)."""

    def __init__(self, num_samples: int, seed: int, transform=None):
        self.num_samples = num_samples
        self.generator = LowerCaseLettersGenerator(seed=seed)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.generator.ALLOWS_SYMBOLS)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_np, label_char = self.generator.generate()
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label_idx = self.char_to_idx[label_char]
        return img_tensor, label_idx