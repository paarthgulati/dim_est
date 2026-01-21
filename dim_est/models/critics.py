# critics.py
import torch
import torch.nn as nn
from ..utils.networks import mlp  # assuming your mlp(dim, hidden_dim, output_dim, layers, activation)

def _unpack_encoder_output(output):
    """
    Handles encoders that return tensor (deterministic) or tuple (variational).
    Returns: (z, kl_loss)
    """
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, 0.0

class SeparableCritic(nn.Module):
    """
    Separably encoded critic:
      zX = encoder_x(x), zY = encoder_y(y)
      scores = zY @ zX^T
    """
    def __init__(self, encoder_x: nn.Module, encoder_y: nn.Module):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y

    def forward(self, x, y):
        # x: [B, Nx], y: [B, Ny]
        zX, kl_x = _unpack_encoder_output(self.encoder_x(x))
        zY, kl_y = _unpack_encoder_output(self.encoder_y(y))
        
        scores = zY @ zX.t()
        
        # Return scores AND combined KL (DSIB will ignore KL, DVSIB will use it)
        return scores, kl_x + kl_y

class BiCritic(nn.Module):
    """
    Bilinear variant:
      zX = encoder_x(x), zY = encoder_y(y)
      zX' = B zX
      scores = zY @ zX'^T
    """
    def __init__(self, encoder_x: nn.Module, encoder_y: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.bilinear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, y):
        zX, kl_x = _unpack_encoder_output(self.encoder_x(x))
        zY, kl_y = _unpack_encoder_output(self.encoder_y(y))
        
        zX = self.bilinear(zX)
        scores = zY @ zX.t()
        return scores, kl_x + kl_y

class SeparableAugmentedCritic(nn.Module):
    """
    Separably encoded critic with quadratic augmentations in zX and zY:

      scores_ij = zY_j · zX_i + qX_i + qY_j

    where qX, qY depend on quad_kind ∈ {"full", "diag", "scalar"}.
    """
    def __init__(self, encoder_x: nn.Module, encoder_y: nn.Module,
                 embed_dim: int, quad_kind: str = "full"):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.quad_kind = quad_kind

        d = embed_dim
        if quad_kind == "full":
            self.A_raw = nn.Parameter(torch.zeros(d, d))
            self.B_raw = nn.Parameter(torch.zeros(d, d))
        elif quad_kind == "diag":
            self.A_diag = nn.Parameter(torch.zeros(d))
            self.B_diag = nn.Parameter(torch.zeros(d))
        elif quad_kind == "scalar":
            self.A_scalar = nn.Parameter(torch.tensor(0.0))
            self.B_scalar = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("quad_kind must be one of {'full','diag','scalar'}")

    def forward(self, x, y):
        zX, kl_x = _unpack_encoder_output(self.encoder_x(x))
        zY, kl_y = _unpack_encoder_output(self.encoder_y(y))
        cross = zY @ zX.t()     # [B, B]

        if self.quad_kind == "full":
            A = 0.5 * (self.A_raw + self.A_raw.t())
            B = 0.5 * (self.B_raw + self.B_raw.t())
            quadX = ((zX @ A) * zX).sum(dim=1)  # [B]
            quadY = ((zY @ B) * zY).sum(dim=1)  # [B]
        elif self.quad_kind == "diag":
            quadX = (zX.pow(2) * self.A_diag).sum(dim=1)
            quadY = (zY.pow(2) * self.B_diag).sum(dim=1)
        elif self.quad_kind == "scalar":
            quadX = self.A_scalar * zX.pow(2).sum(dim=1)
            quadY = self.B_scalar * zY.pow(2).sum(dim=1)
        else:
            raise ValueError("quad_kind must be one of {'full','diag','scalar'}")

        # broadcast into [B, B]
        scores = cross + quadY.unsqueeze(1) + quadX.unsqueeze(0)
        return scores, kl_x + kl_y

class HybridCritic(nn.Module):
    """
    Hybrid critic:
      zX = encoder_x(x)   # [B, d_x]
      zY = encoder_y(y)   # [B, d_y]
      form all pairs (zX_i, zY_j) and pass through pair_mlp: [d_x + d_y] -> 1
    """
    def __init__(self, encoder_x: nn.Module, encoder_y: nn.Module, pair_mlp: nn.Module):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.pair_mlp = pair_mlp

    def forward(self, x, y):
        B = x.shape[0]
        zX, kl_x = _unpack_encoder_output(self.encoder_x(x))
        zY, kl_y = _unpack_encoder_output(self.encoder_y(y))
        dx = zX.shape[-1]
        dy = zY.shape[-1]
        
        if dx == 0 or dy == 0:
            # encoders map to R^0 → no joint information possible
            # return a constant (zero) score matrix
            return x.new_zeros(B, B)

        # all pairs
        zX_expanded = zX.repeat(B, 1)                    # [B*B, d_x]
        zY_expanded = zY.repeat_interleave(B, dim=0)     # [B*B, d_y]
        pair_input = torch.cat([zX_expanded, zY_expanded], dim=1)  # [B*B, d_x+d_y]

        scores_vec = self.pair_mlp(pair_input)           # [B*B, 1] or [B*B]
        scores = scores_vec.view(B, B, -1).squeeze(-1)   # [B, B]
        return scores, kl_x + kl_y


class ConcatCritic(nn.Module):
    """
    Concatenated critic:
      directly form all pairs (x_i, y_j) and feed into pair_mlp: [Nx + Ny] -> 1
      No separate encoders.
    """
    def __init__(self, Nx: int, Ny: int, pair_mlp: nn.Module):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.pair_mlp = pair_mlp

    def forward(self, x, y):
        B = x.shape[0]
        # x: [B, Nx], y: [B, Ny]
        x_expanded = x.repeat(B, 1)                    # [B*B, Nx]
        y_expanded = y.repeat_interleave(B, dim=0)     # [B*B, Ny]
        pair_input = torch.cat([x_expanded, y_expanded], dim=1)  # [B*B, Nx+Ny]

        scores_vec = self.pair_mlp(pair_input)         # [B*B, 1] or [B*B]
        scores = scores_vec.view(B, B, -1).squeeze(-1) # [B, B]
        return scores, 0.0  # No KL divergence since no encoders
