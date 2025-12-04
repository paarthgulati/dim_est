# models.py # currently only includes the deterministic DSIB, possibly add variational cersion later on
 
import torch
import torch.nn as nn
from .estimators import infonce_lower_bound, smile_lower_bound, clip_lower_bound


class DSIB(nn.Module):
    def __init__(self, estimator: str, critic: nn.Module, baseline_fn=None):
        super().__init__()
        self.estimator = estimator
        self.critic = critic
        self.baseline_fn = baseline_fn  # currently unused unless you want to add it in

    def forward(self, dataX, dataY):
        """
        dataX: [B, Nx]
        dataY: [B, Ny]

        Returns:
            loss, mi_i1, mi_i2
        """


        scores = self.critic(dataX, dataY)  # [B, B]

        # TODO: hook baseline in if you want; for now assume None
        if self.estimator == "infonce":
            mi, mi_i1, mi_i2 = infonce_lower_bound(scores)
        elif self.estimator == "smile_5":
            mi, mi_i1, mi_i2 = smile_lower_bound(scores, clip=5.0)
        elif self.estimator == "lclip":
            mi, mi_i1, mi_i2 = clip_lower_bound(scores)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        loss = -mi
        return loss, mi_i1, mi_i2
