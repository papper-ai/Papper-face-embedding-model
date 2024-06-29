import os
from .dataset import Arc2FaceDataset
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    transform: v2,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    train_length: int = -1,
) -> tuple[DataLoader, DataLoader]:
    train_data = Arc2FaceDataset(train_dir, transform=transform, length=train_length)
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
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
