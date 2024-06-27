import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

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
