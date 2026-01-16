import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Fully Connected Layers
        # (66x200) 입력 시 마지막 Conv 이후 크기는 1x18x64 = 1152입니다.
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2) # [linear_x, angular_z]
        
        self.elu = nn.ELU()

    def forward(self, x):
        # 1. Normalization (Lambda 대신 수행)
        x = x.float() / 127.5 - 1.0
        
        # 2. Convolutional Layers
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        
        # 3. Flatten
        x = x.view(x.size(0), -1)
        
        # 4. Fully Connected Layers
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)
        
        return x


# ==================================
# 추가 모델 예시: MobileNet (간단한 버전)
# ==================================
try:
    from torchvision.models import mobilenet_v2, mobilenet_v3_small
    from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
except Exception as e:
    raise ImportError(
        "torchvision이 필요합니다. 예: pip install torchvision\n"
        f"원인: {e}"
    )

class MobileNet(nn.Module):
    """
    MobileNet backbone + regression head for [linear_x, angular_z].

    Args:
        backbone: "v2" or "v3s" (v3 small)
        pretrained: True면 ImageNet weights 사용 (입력 정규화/색공간 주의)
        freeze_backbone: True면 feature extractor freeze
        input_norm:
            - "minus1_1": x/127.5 - 1.0 (현재 PilotNet과 동일)
            - "imagenet": ImageNet mean/std (pretrained 사용할 때 권장, RGB 입력 권장)
    """
    def __init__(
        self,
        backbone: str = "v2",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        input_norm: str = "minus1_1",
        dropout_p: float = 0.2,
        out_dim: int = 2,
    ):
        super().__init__()

        self.input_norm = input_norm

        if backbone == "v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = mobilenet_v2(weights=weights)
            self.features = net.features
            feat_dim = 1280  # mobilenet_v2 last channel
        elif backbone in ["v3s", "v3_small", "v3-small"]:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_small(weights=weights)
            self.features = net.features
            feat_dim = 576  # mobilenet_v3_small last channel
        else:
            raise ValueError("backbone must be one of: 'v2', 'v3s'")

        # Global pooling to handle arbitrary input resolution (e.g., 66x200)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # If using imagenet norm, store mean/std as buffers (on same device)
        # Note: ImageNet weights assume RGB order typically.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (B,3,H,W), dtype uint8 or float
        x = x.float()

        if self.input_norm == "minus1_1":
            # same as your PilotNet
            return x / 127.5 - 1.0

        if self.input_norm == "imagenet":
            # expects x in [0,1]
            x = x / 255.0
            return (x - self.imagenet_mean) / self.imagenet_std

        raise ValueError("input_norm must be 'minus1_1' or 'imagenet'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x