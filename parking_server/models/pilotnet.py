# models/pilotnet.py
import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(48*19*19 + 3, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x, slot):
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        slot_onehot = torch.zeros(x.size(0), 3, device=x.device)
        slot_onehot[:, slot-1] = 1.0
        z = torch.cat([z, slot_onehot], dim=1)
        return self.fc(z)
