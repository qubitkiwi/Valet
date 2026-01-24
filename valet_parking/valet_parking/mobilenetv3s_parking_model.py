import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MultiCamParkingModel(nn.Module):
    def __init__(self, 
                 out_dim=2, 
                 pretrained=False, 
                 freeze_backbone=False, 
                 input_norm="minus1_1"):
        """
        4방향 카메라(Front, Rear, Left, Right)를 사용하는 자율주행 주차 모델
        Late Fusion 방식: 4개의 이미지를 각각 MobileNet에 통과시킨 후 특징을 합침
        """
        super().__init__()
        self.input_norm = input_norm
        
        # ---------------------------------------------------------
        # 1. Backbone 설정 (MobileNetV3 Small)
        # ---------------------------------------------------------
        # pretrained=False 이므로 가중치는 랜덤 초기화 상태로 시작합니다.
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        
        # 전체 모델 불러오기
        full_mobilenet = mobilenet_v3_small(weights=weights)
        
        # Classifier(Head)를 제거하고 특징 추출기(Features) 부분만 가져옵니다.
        # MobileNetV3-Small의 마지막 Conv 출력 채널은 576개입니다.
        self.features = full_mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 백본 파라미터 고정 (Pretrained 사용 시 초반 학습 안정화를 위해 사용 가능)
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # ---------------------------------------------------------
        # 2. Regression Head 설정 (4개 카메라 정보 통합)
        # ---------------------------------------------------------
        # MobileNetV3-Small의 Feature Dim = 576
        # 카메라가 4대이므로 입력 차원은 576 * 4 = 2304가 됩니다.
        self.backbone_out_dim = 576
        self.num_cameras = 4
        
        total_input_dim = self.backbone_out_dim * self.num_cameras # 2304

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_input_dim, 128), # 2304 -> 128로 압축
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),               # 과적합 방지
            nn.Linear(128, out_dim)          # 최종 출력 (Linear_X, Angular_Z)
        )
        
        # ImageNet 정규화용 상수 (버퍼에 등록하여 GPU 이동 시 자동 처리)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def _normalize(self, x):
        """
        입력 이미지 정규화
        x shape: (Batch, 3, H, W) - 이미 view로 펼쳐진 상태로 들어옴
        """
        x = x.float()
        
        if self.input_norm == "minus1_1":
            # 픽셀값 0~255 -> -1 ~ 1 로 변환 (PilotNet 방식)
            return x / 127.5 - 1.0
            
        elif self.input_norm == "imagenet":
            # ImageNet 평균/표준편차 정규화 (0~1 변환 후 처리)
            x = x / 255.0
            return (x - self.imagenet_mean) / self.imagenet_std
            
        return x

    def forward(self, x):
        """
        Forward Pass
        Input x shape: (Batch, 4, 3, Height, Width)
        - 4는 카메라 개수 (순서: Front, Rear, Left, Right 가정)
        """
        B, N, C, H, W = x.shape
        
        # ---------------------------------------------------------
        # 단계 1: 병렬 처리를 위한 모양 변경 (Batch * 4)
        # (Batch, 4, 3, H, W) -> (Batch * 4, 3, H, W)
        # 이렇게 하면 GPU는 마치 이미지가 4배 많은 것처럼 인식하고 한 번에 연산합니다.
        # ---------------------------------------------------------
        x = x.view(B * N, C, H, W)
        
        # 정규화 적용
        x = self._normalize(x)
        
        # ---------------------------------------------------------
        # 단계 2: Backbone 통과 (Feature Extraction)
        # 결과: (Batch * 4, 576, h, w)
        # ---------------------------------------------------------
        feat = self.features(x)
        
        # Pooling: (Batch * 4, 576, 1, 1) -> Flatten -> (Batch * 4, 576)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        
        # ---------------------------------------------------------
        # 단계 3: 다시 카메라 별로 나누고 합치기 (Late Fusion)
        # (Batch * 4, 576) -> (Batch, 4, 576)
        # ---------------------------------------------------------
        feat = feat.view(B, N, -1)
        
        # (Batch, 4, 576) -> (Batch, 2304)
        # dim=1 방향으로 펼쳐서 [앞 특징, 좌 특징, 우 특징, 뒤 특징]을 이어 붙임
        feat = feat.view(B, -1)
        
        # ---------------------------------------------------------
        # 단계 4: Regression Head 통과 (최종 예측)
        # ---------------------------------------------------------
        out = self.head(feat)
        
        return out

# ---------------------------------------------------------
# 테스트 코드 (이 파일을 직접 실행했을 때만 동작)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 로컬 PC 환경 테스트용 (CPU/GPU 자동 감지)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 모델 생성 (Pretrained X)
    model = MultiCamParkingModel(pretrained=False).to(device)
    
    # 가상의 입력 데이터 (Batch=2, Camera=4, Channel=3, Height=224, Width=224)
    # 실제 이미지 크기는 66x200 등 달라도 상관없음 (AdaptiveAvgPool이 있어서)
    dummy_input = torch.zeros(2, 4, 3, 224, 224).to(device)
    
    # 추론 실행
    output = model(dummy_input)
    
    print("\n--- Model Summary ---")
    print(f"Input Shape: {dummy_input.shape}")     # [2, 4, 3, 224, 224]
    print(f"Output Shape: {output.shape}")        # [2, 2] -> (Linear_X, Angular_Z)
    print("Model forward pass successful!")
    print(f"Backbone weights loaded: {'Pretrained' if False else 'Random Init'}")