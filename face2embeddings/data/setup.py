import os
from .dataset import Arc2FaceDataset
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
    transform: v2,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    train_data = Arc2FaceDataset(train_dir, transform=transform)
    test_data = Arc2FaceDataset(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
