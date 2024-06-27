from torch import nn
import torch
from pathlib import Path
from torchvision.models import Swin_V2_S_Weights, swin_v2_s


class FaceSwin(nn.Module):
    def __init__(self, train_from_default: bool = False):
        super().__init__()

        if train_from_default:
            base_swin = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
            self.features_extractor = nn.Sequential(
                base_swin.features,
                base_swin.norm,
                base_swin.permute,
                base_swin.avgpool,
                base_swin.flatten,
            )
            self.head = nn.Sequential(
                nn.Linear(in_features=1536, out_features=512),
                nn.GELU(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=512, out_features=128),
                nn.GELU(),
                nn.Linear(in_features=128, out_features=1),
                nn.Sigmoid(),
            )
        else:
            base_swin = swin_v2_s()
            self.features_extractor = nn.Sequential(
                base_swin.features,
                base_swin.norm,
                base_swin.permute,
                base_swin.avgpool,
                base_swin.flatten,
            )

    def load_feature_extractor(self, path: Path):
        self.features_extractor.load_state_dict(torch.load(path))

    def save_feature_extractor(self, path: Path):
        torch.save(self.features_extractor.state_dict(), path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_extractor(x)
        return x

    def train_forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.features_extractor(x1)
        x2 = self.features_extractor(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.head(x)
        return x
