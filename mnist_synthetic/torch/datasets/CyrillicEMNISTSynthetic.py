from typing import Any, Optional, Callable, List
from pathlib import Path

import numpy as np
from PIL import Image
from numpy._typing import NDArray
from torchvision.datasets import VisionDataset

from mnist_synthetic.lower_case_letters_generator import LowerCaseLettersGenerator

class CyrillicEMNISTSynthetic(VisionDataset):
    def __init__(
        self,
        size: int,
        root: str | Path | None = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int | None = None,
        include: List[str] | None = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        assert isinstance(size, int) and size > 0, 'size must be a positive integer'
        self.size: int = size
        self.seed: int | None = seed
        self.include: List[str] | None = include

        data, labels, extra = self._load_data()

        self.data: NDArray[np.uint8] = data
        self.targets: List[str] = labels
        self.extra: List[dict[str, Any] | None] = extra

    def __len__(self) -> int:
        return self.size

    def _load_data(self) -> tuple[NDArray[np.uint8], List[str], List[dict[str, Any] | None]]:
        data = np.zeros((self.size, 28, 28), dtype=np.uint8)
        labels: List[str] = []
        extra: List[dict[str, Any] | None] = []

        generator = LowerCaseLettersGenerator(seed=self.seed)
        for i in range(self.size):
            img, label, extra_ = self._generate_image(generator)

            data[i, :, :] = img
            labels.append(label)
            extra.append(extra_)

        return data, labels, extra

    def _generate_image(
        self,
        generator: LowerCaseLettersGenerator
    ) -> tuple[NDArray[np.uint8], str, dict[str, Any] | None]:
        img, label = generator.generate(include=self.include)

        return img, label, None

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        img_np = self.data[idx]
        target: str = self.targets[idx]

        # Many transform methods require Pillow
        img = Image.fromarray(img_np, mode="L")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target