import os
from .dataset import Arc2FaceDataset
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    transforms: torchvision.transforms.Compose | None = None,
    num_workers: int = NUM_WORKERS,
    train_length: int = -1,
) -> tuple[DataLoader, DataLoader]:
    train_data = Arc2FaceDataset(train_dir, transforms=transforms, length=train_length)
    val_data = Arc2FaceDataset(val_dir)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
