# ==============================================================================
# dataset.py
# GCN transform, train/eval transforms, and CellDataset class.
# Taken directly from Cells 5 & 6 of the project notebook.
# ==============================================================================

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import config


# ==============================================================================
# CELL 5 — Global Contrast Normalisation transform
# ==============================================================================
"""
Global Contrast Normalization (GCN)

Blood smear images suffer from inconsistent Wright-Giemsa staining intensity
slide-to-slide. GCN removes per-channel mean and scales by std so staining
intensity doesn't dominate early convolutional features.

Formula per channel c:
    x_gcn[c] = (x[c] - mean(x[c])) / (std(x[c]) + eps)
"""

class GlobalContrastNormalization(torch.nn.Module):
    """Per-image, per-channel GCN as a composable torchvision transform."""

    def __init__(self, eps: float = config.GCN_EPS):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, H, W) float32 in [0, 1]
        mean = x.mean(dim=[-2, -1], keepdim=True)   # (C, 1, 1)
        std  = x.std(dim=[-2, -1],  keepdim=True)   # (C, 1, 1)
        return (x - mean) / (std + self.eps)


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

# Training: augmentation + GCN (cells are radially symmetric — full 360° rotation)
train_transform = transforms.Compose([
    transforms.Resize((config.resize_y, config.resize_x)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),                   # (C, H, W) float32 in [0, 1]
    GlobalContrastNormalization(),
])

# Val / test: only resize + GCN, no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((config.resize_y, config.resize_x)),
    transforms.ToTensor(),
    GlobalContrastNormalization(),
])


# ==============================================================================
# CELL 6 — Dataset class
# ==============================================================================

class CellDataset(Dataset):
    """
    Reads images from:
        root/
            LYMPHOCYTE/   *.jpg / *.jpeg / *.png
            NEUTROPHIL/
            MONOCYTE/
            EOSINOPHIL/

    Returns (image_tensor, label_int) where label_int indexes config.CLASS_NAMES.
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples   = []   # list of (path_str, class_index)

        for idx, cls in enumerate(config.CLASS_NAMES):
            cls_dir = Path(root) / cls
            if not cls_dir.exists():
                print(f'  Warning: {cls_dir} not found — skipping.')
                continue
            imgs = (list(cls_dir.glob('*.jpg'))
                  + list(cls_dir.glob('*.jpeg'))
                  + list(cls_dir.glob('*.png')))
            self.samples.extend([(str(p), idx) for p in imgs])

        print(f'  {len(self.samples)} images from {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')   # force 3 channels
        if self.transform:
            img = self.transform(img)
        return img, label


# ==============================================================================
# DataLoader factory  (used by interface.py)
# ==============================================================================

def the_dataloader(
    root:        str  = config.DATA_DIR,
    train:       bool = True,
    batch_size:  int  = config.batch_size,
    num_workers: int  = 2,
    pin_memory:  bool = True,
) -> DataLoader:
    """
    Build and return a DataLoader for CellDataset.

    Parameters
    ----------
    root        : path to split directory (contains class sub-folders)
    train       : True  → apply train_transform + shuffle
                  False → apply eval_transform, no shuffle
    batch_size  : mini-batch size
    num_workers : parallel data-loading workers
    pin_memory  : speeds up CPU→GPU transfer
    """
    transform = train_transform if train else eval_transform
    dataset   = CellDataset(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = train,
        num_workers = num_workers,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        drop_last   = False,
    )
