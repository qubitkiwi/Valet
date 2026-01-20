# models/pilotnet6ch.py
import torch
import torch.nn as nn

class PilotNet6Ch(nn.Module):
    """
    입력:
      - x: (B, 6, H, W)  front+rear concat
      - p_oh: (B, 3)     [p1,p2,p3] one-hot (float)
    출력:
      - (B, 2)           [linear_x, angular_z]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc1 = nn.Linear(1, 100)  # placeholder (lazy init)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 2),
        )
        self._fc_inited = False

    def _init_fc(self, n_flat: int, device):
        # ✅ conv flatten + p1,p2,p3(3)
        self.fc1 = nn.Linear(n_flat + 3, 100).to(device)
        self._fc_inited = True

    def forward(self, x: torch.Tensor, p_oh: torch.Tensor):
        z = self.conv(x)
        z = z.reshape(z.size(0), -1)

        if not self._fc_inited:
            self._init_fc(z.size(1), x.device)

        # p_oh: (B,3) float
        if p_oh.dim() == 1:
            p_oh = p_oh.unsqueeze(0)
        p_oh = p_oh.to(device=x.device, dtype=z.dtype)

        z = torch.cat([z, p_oh], dim=1)
        z = self.fc1(z)
        out = self.fc2(z)
        return out
