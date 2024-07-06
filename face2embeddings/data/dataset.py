import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import random
from torchvision.models import Swin_V2_B_Weights
from torchvision.transforms import v2
import torch
import copy


class Arc2FaceDataset(Dataset):
    def __init__(
        self, path_to_dataset: Path, transforms: v2.Compose = None, length: int = -1
    ):
        self.image_paths, self.folders = self.found_images(path_to_dataset, length)
        swin_transforms = Swin_V2_B_Weights.DEFAULT.transforms()
        self.transform = (
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    swin_transforms,
                ]
            )
            if transforms is None
            else v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    transforms,
                    swin_transforms,
                ]
            )
        )

    @staticmethod
    def found_images(
        path_to_dataset: Path, length: int = -1
    ) -> tuple[list[np.ndarray], np.ndarray]:
        folder_img_paths = []
        dataset_img_paths = []
        folders = []
        previous_folder = None
        for image in path_to_dataset.glob("**/*.jpg"):
            if length != -1:
                if len(dataset_img_paths) <= length:
                    if image.parent == previous_folder:
                        folder_img_paths.append(image)
                        previous_folder = image.parent
                    else:
                        if previous_folder is not None:
                            dataset_img_paths.append(
                                copy.deepcopy(
                                    np.array(list(map(str, folder_img_paths)))
                                )
                            )
                            folders.append(str(previous_folder))
                            folder_img_paths.clear()
                        folder_img_paths.append(image)
                        previous_folder = image.parent
                else:
                    break
            else:
                if image.parent == previous_folder:
                    folder_img_paths.append(image)
                    previous_folder = image.parent
                else:
                    if previous_folder is not None:
                        dataset_img_paths.append(
                            copy.deepcopy(np.array(list(map(str, folder_img_paths))))
                        )
                        folders.append(str(previous_folder))
                        folder_img_paths.clear()
                    folder_img_paths.append(image)
                    previous_folder = image.parent

        return dataset_img_paths, np.array(folders)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_folder1 = self.image_paths[idx]
        filtered_indices = [i for i in range(len(self.image_paths)) if i != idx]
        image_folder2 = self.image_paths[random.choice(filtered_indices)]

        image1_path = np.random.choice(image_folder1)
        image2_path = (
            np.random.choice(image_folder1[image_folder1 != image1_path])
            if len(image_folder1) > 1
            else image1_path
        )
        image3_path = np.random.choice(image_folder2)

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        image3 = Image.open(image3_path)

        t_image1 = self.transform(image1)
        t_image2 = self.transform(image2)
        t_image3 = self.transform(image3)

        return t_image1, t_image2, t_image3

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return "Arc2FaceDataset"
