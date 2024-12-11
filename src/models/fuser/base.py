from typing import List

import torch
import torch.nn as nn


class Fuser(nn.Module):
    def __init__(self, n_auxiliary_features: int):
        super().__init__()

        self.n_auxiliary_features = n_auxiliary_features

    def forward(
        self, image_feature: torch.Tensor, auxiliary_features: List[torch.Tensor]
    ) -> torch.Tensor:
        pass
