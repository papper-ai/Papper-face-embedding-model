import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs: torch.Tensor | list[torch.Tensor], labels: list | str | None = None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(labels, list) and labels is not None:
        labels = [labels]
        assert len(imgs) == len(labels)
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, ax in enumerate(axs.flat):
        img = imgs[i]
        img = img.detach()
        img = F.to_pil_image(img)
        ax.imshow(np.asarray(img))
        if labels is not None:
            ax.set_title(str(i))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> torch.utils.tensorboard.writer.SummaryWriter():
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
