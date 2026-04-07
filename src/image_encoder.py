from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class SmallImageEncoder(nn.Module):
    def __init__(self, embed_dim = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1), #B, C, H, W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Linear(128, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        x = self.backbone(image)   # x: [B, 128, 1, 1]
        x = x.flatten(start_dim=1)  # x: [B, 128]
        x = self.projector(x) # [B, E]
        x = F.normalize(x, dim=-1)
        return x