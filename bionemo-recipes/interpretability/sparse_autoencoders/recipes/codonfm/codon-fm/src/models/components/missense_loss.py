import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MissenseLoss(nn.Module):
    """
    Custom loss function for missense variant prediction.
    
    This loss function maintains the exact same algorithm as the original 
    missense_loss function but adds numerical stability to prevent inf/nan values.
    """
    
    def __init__(self, eps: float = 1e-4, clip_negative_at_logit: float = 0.0, clip_positive_at_logit: float = -1.0):
        """
        Initialize MissenseLoss.
        
        Args:
            eps: Small epsilon value to prevent log(0) and ensure numerical stability
            clip_negative_at_logit: Logit threshold to clip at for the benign class (y=0)
            clip_positive_at_logit: Logit threshold to clip at for the pathogenic class (y=1)
        """
        super().__init__()
        self.eps = eps
        self.clip_negative_at_logit = clip_negative_at_logit
        self.clip_positive_at_logit = clip_positive_at_logit
    
    def forward(self, preds: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute missense loss with the original algorithm but numerical stability.
        
        Original algorithm:
        loss = y * log(sigmoid(clamp(preds, min=-1))) + (1-y) * log(1 - sigmoid(clamp(preds, min=0)))
        
        Args:
            preds: Predictions tensor of shape (...,)
            y: Target labels tensor of shape (...,)
            w: Optional sample weights tensor of shape (...,)
            
        Returns:
            loss: Computed loss scalar, guaranteed to be finite
        """
        eps = max(self.eps, torch.finfo(preds.dtype).eps)
        prob = torch.sigmoid(preds)
        prob = torch.clamp(prob, min=eps, max=1.0 - eps)
        loss = -y * torch.log(prob) - (1 - y) * torch.log(1 - prob)
        
        # Use softplus for numerical stability: log(exp(x) + 1) = softplus(x)
        # Apply negative clipping when y == 0 (negative class) AND preds < clip_negative_at_logit
        loss_at_clip = F.softplus(torch.tensor(0, device=preds.device, dtype=preds.dtype))
        negative_condition = (y == 0) & (preds < self.clip_negative_at_logit)
        loss = torch.where(negative_condition, loss_at_clip, loss)
        
        # Apply positive clipping when y == 1 (positive class) AND preds < clip_positive_at_logit
        loss_at_clip = F.softplus(torch.tensor(-self.clip_positive_at_logit, device=preds.device, dtype=preds.dtype))
        positive_condition = (y == 1) & (preds < self.clip_positive_at_logit)
        loss = torch.where(positive_condition, loss_at_clip, loss)
        if w is not None:
            loss = (w * loss).mean()
        else:
            loss = loss.mean()
        return loss
