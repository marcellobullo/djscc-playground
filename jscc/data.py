"""Image-folder dataset + loaders for training (ported from djscc-demo).

Works with flat image directories (DIV2K, Flickr2K, CLIC, ...) — one or more
roots are concatenated. Images are resized to the model's training resolution.
"""

from __future__ import annotations

import os
from typing import List

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDirectoryDataset(Dataset):
    """A flat directory of images (recursively discovered)."""

    def __init__(self, root: str, transform=None) -> None:
        self.transform = transform
        self.paths = sorted(
            os.path.join(dp, f)
            for dp, _, files in os.walk(root)
            for f in files
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_loader(dirs: List[str], height: int, width: int, batch_size: int,
                train: bool = True, num_workers: int = 4) -> DataLoader:
    ops = [transforms.Resize((height, width))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    tfm = transforms.Compose(ops)
    ds = ConcatDataset([ImageDirectoryDataset(d, tfm) for d in dirs])
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, drop_last=train, pin_memory=True)
