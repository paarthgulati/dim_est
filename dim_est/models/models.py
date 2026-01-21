# models.py # currently only includes the deterministic DSIB, possibly add variational cersion later on
 
import torch
import torch.nn as nn
from .estimators import infonce_lower_bound, smile_lower_bound, clip_lower_bound


class DSIB(nn.Module):
    def __init__(self, estimator: str, critic: nn.Module, baseline_fn=None):
        super().__init__()
        self.estimator = estimator
        self.critic = critic

    def forward(self, dataX, dataY):
        """
        dataX: [B, Nx]
        dataY: [B, Ny]

        Returns:
            loss, mi_i1, mi_i2
        """


        scores, _ = self.critic(dataX, dataY)  # [B, B]

        if self.estimator == "infonce":
            mi, extras = infonce_lower_bound(scores)
        elif self.estimator == "smile_5":
            mi, extras = smile_lower_bound(scores, clip=5.0)
        elif self.estimator == "lclip":
            mi, extras = clip_lower_bound(scores)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        loss = -mi
        return loss, mi, extras

class DVSIB(nn.Module):
    def __init__(self, estimator: str, critic: nn.Module, beta: float = 128.0):
        super().__init__()
        self.estimator = estimator
        self.critic = critic
        self.beta = beta

    def forward(self, dataX, dataY):
        # Critic returns scores AND the sum of KL divergences
        scores, kl_loss = self.critic(dataX, dataY)

        # Calculate MI on the samples (z) used to generate scores
        if self.estimator == "infonce":
            mi, extras = infonce_lower_bound(scores)
        elif self.estimator == "smile_5":
            mi, extras = smile_lower_bound(scores, clip=5.0)
        elif self.estimator == "lclip":
            mi, extras = clip_lower_bound(scores)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        # DVSIB Objective: Minimize KL - beta * MI
        loss = kl_loss - (self.beta * mi)
        
        # Add KL to extras for logging
        if isinstance(extras, dict):
            extras['kl_loss'] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss

        return loss, mi, extras