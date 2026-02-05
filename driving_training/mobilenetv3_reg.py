import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class RegDrivingModel(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = mobilenet_v3_small(weights=None)
        self.backbone = mobilenet.features

        input_dim = 576  # mobilenetv3 small last channel
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [linear_x, angular_z]
        )

    def forward(self, x):
        x = self.backbone(x)     # [B,576,H,W]
        x = self.gap(x)          # [B,576,1,1]
        x = torch.flatten(x, 1)  # [B,576]
        control_out = self.regression_head(x)  # [B,2]
        return control_out
