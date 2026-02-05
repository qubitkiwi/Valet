import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MultiTaskDrivingModel(nn.Module):
    def __init__(self, num_signs=3): # num_signs: 표지판 종류 수 (일반, 직진, 우회전, (주차))
        super(MultiTaskDrivingModel, self).__init__()

        # 1. Backbone: MobileNetV3 Small (이미지에서 특징 추출)
        # .features만 가져오면 마지막 GAP와 Classifier 레이어는 제외된 컨볼루션 층만 가져옵니다.
        mobilenet = mobilenet_v3_small(weights=None) # pretrained=False
        self.backbone = mobilenet.features

        # MobileNetV3 Small의 마지막 채널 수는 576이다. 모델 버전 변경 시 이걸 수정하면 됨
        input_dim = 576

        # 2. Neck: 데이터를 압축하는 Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 3. Head 1: Regression (angular_z, linear_x)
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 출력: [angular_z, linear_x]
        )

        # 4. Head 2: Classification (Signs)
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_signs) # 출력: 표지판 클래스별 점수
        )

    def forward(self, x):
        # [공통 단계] Backbone을 통해 특징 추출
        x = self.backbone(x)     # 결과: [Batch, 576, H, W]
        x = self.gap(x)          # 결과: [Batch, 576, 1, 1]
        x = torch.flatten(x, 1)  # 결과: [Batch, 576] (1차원으로 펴주기)

        # [분기 단계] 공통 특징을 각각의 Head로 전달
        # 1. 제어값 예측 (Regression)
        control_out = self.regression_head(x)

        # 2. 표지판 인식 (Classification)
        sign_out = self.classification_head(x)

        # 두 가지 결과를 동시에 반환 (딕셔너리 형태가 관리하기 편함)
        return {
            "control": control_out,
            "signs": sign_out
        }

