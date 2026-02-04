# model.py
import torch
import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

class DeepLabV3MultiTask(nn.Module):
    """
    DeepLabV3-MobileNetV3-Large backbone (pretrained) + 2 heads:
      - control regression head: outputs [B,2] = [linear_x, angular_z]
      - sign classification head: outputs [B,num_signs] logits

    Returns dict to match your training loop:
      {"control": ..., "signs": ...}
    """
    def __init__(
        self,
        num_signs: int,
        pretrained: bool = True,
        control_hidden: int = 256,
        sign_hidden: int = 256,
        dropout_p: float = 0.1,
        feat_key: str = "out",  # backbone feature dict key
    ):
        super().__init__()
        self.num_signs = num_signs
        self.feat_key = feat_key

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        seg_model = deeplabv3_mobilenet_v3_large(weights=weights)

        # We only use the backbone features (OrderedDict like {"out": feat})
        self.backbone = seg_model.backbone

        # infer feature dim robustly
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            if feat_key not in feats:
                raise KeyError(f"backbone() returned keys {list(feats.keys())}, but feat_key='{feat_key}' not found.")
            feat_dim = feats[feat_key].shape[1]

        # control head
        self.control_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feat_dim, control_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(control_hidden, 2),
        )

        # sign head
        self.sign_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feat_dim, sign_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(sign_hidden, num_signs),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feat = feats[self.feat_key]
        control = self.control_head(feat)
        signs   = self.sign_head(feat)
        return {"control": control, "signs": signs}


def set_backbone_trainable(model: DeepLabV3MultiTask, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
