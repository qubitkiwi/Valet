import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        gamma: focusing parameter (2.0 common)
        alpha: None, scalar, or tensor [num_classes] for class weighting
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        logp = F.log_softmax(logits, dim=1)  # [B, C]
        p = torch.exp(logp)                  # [B, C]

        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)        # [B]

        loss = -((1.0 - p_t) ** self.gamma) * logp_t              # [B]

        if self.alpha is not None:
            if not torch.is_tensor(self.alpha):
                alpha_t = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha_t.gather(0, targets)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss