from torch import nn
import torch
from pathlib import Path
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class FaceSwin(nn.Module):
    def __init__(self, train_from_default: bool = False):
        super().__init__()
        self.train_from_default = train_from_default
        if self.train_from_default:
            base_swin = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        else:
            base_swin = swin_v2_b()
            self.reduction = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
            )

        self.encoder = nn.Sequential(
            base_swin.features,
            base_swin.norm,
            base_swin.permute,
            base_swin.avgpool,
            base_swin.flatten,
        )

    def load_encoder(self, path: Path):
        self.encoder.load_state_dict(torch.load(path))

    def save_encoder(self, path: Path):
        torch.save(self.encoder.state_dict(), path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduction(self.encoder(x))
        return x

    def train_forward(
        self, anchor: torch.Tensor, pos_example: torch.Tensor, neg_example: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = (
            self.encoder(anchor)
            if self.train_from_default
            else self.reduction(self.encoder(anchor))
        )
        pos_example = (
            self.encoder(pos_example)
            if self.train_from_default
            else self.reduction(self.encoder(pos_example))
        )
        neg_example = (
            self.encoder(neg_example)
            if self.train_from_default
            else self.reduction(self.encoder(neg_example))
        )
        return anchor, pos_example, neg_example
