import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MultiCamParkingModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, num_classes=2):
        super().__init__()
        
        # 1. Pretrained Weights 설정
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            # features까지만 가져오면 마지막 pooling과 classifier는 제외됨
            self.backbone = mobilenet_v3_small(weights=weights).features
            print("✅ ImageNet Pretrained Weights Loaded!")
        else:
            self.backbone = mobilenet_v3_small(weights=None).features
            print("✨ Training from Scratch (No Pretraining)")

        # 2. Backbone Freeze (선택 사항)
        if pretrained and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # MobileNetV3-Small의 마지막 채널 수 = 576
        # 기존에는 feat_dim * 4 (2304)였으나, 이제는 Conv로 압축하므로 줄어듭니다.
        # Fusion Conv를 거쳐 나온 채널 수 = 256
        fused_dim = 256

        # 3. Spatial Fusion Layer (핵심 변경 사항)
        # 4개의 7x7 맵을 2x2로 붙이면 14x14가 됩니다.
        # 이를 훑어서 공간적 연관성을 학습하는 Convolution 층입니다.
        self.fusion_conv = nn.Sequential(
            # (B, 576, 14, 14) -> (B, 576, 14, 14) : 공간 정보 섞기
            nn.Conv2d(576, 576, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            
            # (B, 576, 14, 14) -> (B, 256, 7, 7) : 채널 압축 및 사이즈 복원(다운샘플링)
            nn.Conv2d(576, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 4. Global Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 5. Multi-Task Heads (입력 차원이 2304 -> 256으로 변경됨)
        # Head 1: Regression Head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [linear_x, angular_z]
        )

        # Head 2: Classification Head
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 6. Normalization 상수
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.pretrained = pretrained

    def forward(self, x):
        """
        Input x: (Batch, 4, 3, 224, 224) 
        Dataset Order: [0:Front, 1:Rear, 2:Left, 3:Right]
        """
        B, N, C, H, W = x.shape
        
        # (Step 1) 이미지 전처리
        x = x.view(B * N, C, H, W) # (B*4, 3, 224, 224)
        x = x.float() / 255.0
        
        if self.pretrained:
            x = (x - self.mean) / self.std
        else:
            x = (x - 0.5) / 0.5

        # (Step 2) Backbone 통과 (공간 정보 유지)
        # 결과: (B*4, 576, 7, 7) - 224 입력 기준
        feat = self.backbone(x) 
        
        # 다시 배치와 카메라 차원으로 분리
        # (B, 4, 576, 7, 7)
        _, fC, fH, fW = feat.shape
        feat = feat.view(B, N, fC, fH, fW)
        
        # (Step 3) Spatial Tiling (공간 이어붙이기)
        # Dataset 순서: 0:Front, 1:Rear, 2:Left, 3:Right
        # 목표: 차량을 중심으로 2x2 격자 형성
        # [ Front(0) | Right(3) ]
        # [ Left(2)  | Rear(1)  ] 
        # 이렇게 배치하면 인접한 카메라끼리 경계면이 맞닿게 됩니다.
        
        row1 = torch.cat([feat[:, 0], feat[:, 3]], dim=3) # 위쪽 줄: Front + Right (가로 결합)
        row2 = torch.cat([feat[:, 2], feat[:, 1]], dim=3) # 아래 줄: Left + Rear (가로 결합)
        
        # 최종 맵: (B, 576, 14, 14)
        combined_map = torch.cat([row1, row2], dim=2)     # 위 + 아래 (세로 결합)
        
        # (Step 4) Fusion Convolution (공간 상관관계 학습)
        # (B, 576, 14, 14) -> (B, 256, 7, 7)
        fused_feat = self.fusion_conv(combined_map)
        
        # (Step 5) Global Pooling & Flatten
        # (B, 256, 7, 7) -> (B, 256, 1, 1) -> (B, 256)
        fused_vec = self.pool(fused_feat)
        fused_vec = fused_vec.view(B, -1)
        
        # (Step 6) Heads
        control_out = self.regression_head(fused_vec)
        classification_out = self.classification_head(fused_vec)

        return {
            "control": control_out,
            "class": classification_out
        }