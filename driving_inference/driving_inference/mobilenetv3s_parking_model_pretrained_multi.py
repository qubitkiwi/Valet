import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MultiCamParkingModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, num_classes=2):
        super().__init__()
        
        # 1. Pretrained Weights 설정
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights).features
            print("✅ ImageNet Pretrained Weights Loaded!")
        else:
            self.backbone = mobilenet_v3_small(weights=None).features
            print("✨ Training from Scratch (No Pretraining)")

        # 2. Backbone Freeze (선택 사항: 초반에는 특징 추출기 고정하고 싶을 때)
        if pretrained and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # MobileNetV3-Small의 마지막 채널 수 = 576
        feat_dim = 576
        # 4개 카메라의 특징을 합치므로 입력 차원은 feat_dim * 4
        combined_dim = feat_dim * 4

        # 3. Global Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Multi-Task Heads
        # Head 1: Regression Head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [linear_x, angular_z]
        )

        # Head 2: Classification (표지판/상태 분류용) -> [class_score 1, ..., class_score N]
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 5. ImageNet Normalization 상수 (GPU 연산을 위해 register_buffer 사용)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.pretrained = pretrained

    def forward(self, x):
        """
        Input x: (Batch, 4, 3, Height, Width) -> 4개의 카메라 이미지
        """
        B, N, C, H, W = x.shape
        
        # (Step 1) 이미지를 모델에 넣기 좋게 변형
        # 4개의 카메라를 배치 차원으로 합칩니다. (B*4, 3, H, W)
        # 이렇게 하면 모델은 마치 (B*4)개의 이미지가 들어온 것처럼 한 번에 처리합니다.
        x = x.view(B * N, C, H, W)
        
        # (Step 2) Normalization
        x = x.float() / 255.0  # 0~1 사이로 변환
        
        if self.pretrained:
            # ImageNet 방식 정규화
            x = (x - self.mean) / self.std
        else:
            # 기존 방식 (-1 ~ 1)
            x = (x - 0.5) / 0.5

        # (Step 3) Backbone 통과
        feat = self.backbone(x)  # (B*4, 576, h, w)
        feat = self.pool(feat)   # (B*4, 576, 1, 1)
        feat = feat.view(B, N, -1) # 다시 (B, 4, 576)으로 복구
        
        # (Step 4) 4개 카메라 특징 결합 (Flatten)
        # (B, 4 * 576) -> (B, 2304)
        feat_cat = feat.view(B, -1)
        
        # (Step 5) 최종 예측 (변경됨: Head 분기)
        control_out = self.regression_head(feat_cat)       # Shape: (B, 2)
        classification_out = self.classification_head(feat_cat) # Shape: (B, num_classes)

        # 두 가지 결과를 동시에 반환 (딕셔너리 형태가 관리하기 편함)
        return {
            "control": control_out,
            "class": classification_out
        }