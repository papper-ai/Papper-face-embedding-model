from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import random
from torchvision.models import Swin_V2_S_Weights
from torchvision.transforms import v2
import torch


class Arc2FaceDataset(Dataset):
    def __init__(self, path_to_dataset: Path, transform: v2 = None):
        self.image_paths, self.labels = self.found_images(path_to_dataset)
        self.transform = (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    Swin_V2_S_Weights.DEFAULT.transforms(),
                ]
            )
            if transform is None
            else v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    transform,
                    Swin_V2_S_Weights.DEFAULT.transforms(),
                ]
            )
        )

    @staticmethod
    def found_images(path_to_dataset: Path) -> tuple[list[list[Path]], list[Path]]:
        temp_paths = list()
        previous_folder = None
        paths = list()
        folders = list()
        total_images = 0
        for image in path_to_dataset.glob("**/*.jpg"):
            if image.parent == previous_folder:
                temp_paths.append(image)
                previous_folder = image.parent
            else:
                if previous_folder is not None:
                    paths.append(temp_paths)
                    folders.append(previous_folder)
                    total_images += len(temp_paths)
                    temp_paths.clear()
                temp_paths.append(image)
                previous_folder = image.parent

        return paths, folders

    def __getitem__(self, idx) -> tuple[Image, Image, int]:
        images_class = random.choice([0, 1])

        # if image class is 1, then select two random images from the one folder, else select two full random image
        if images_class:
            image_folder1 = self.image_paths[idx]
            image1_path = random.choice(image_folder1)
            while image1_path == (image2_path := random.choice(image_folder1)):
                continue
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
        else:
            image_folder1 = self.image_paths[idx]
            image_folder2 = self.image_paths[
                random.randint(0, len(self.image_paths) - 1)
            ]
            image1_path = random.choice(image_folder1)
            image2_path = random.choice(image_folder2)
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, images_class

    def __len__(self):
        return len(self.image_paths)
