
import numpy as np
import torch
import torch.nn.functional as F

# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"


def infonce_lower_bound(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi

# def clip_lower_bound(scores):
#     nll = scores.diag().mean() - 0.5 * scores.logsumexp(dim=1) - 0.5 * scores.logsumexp(dim=0)
#     mi = torch.tensor(scores.size(0)).float().log() + nll
#     mi = mi.mean()
#     return mi

def clip_lower_bound(scores):
    nll_1 = 0.5 * scores.diag().mean() - 0.5 * scores.logsumexp(dim=1) 
    nll_2 = 0.5 * scores.diag().mean() - 0.5 * scores.logsumexp(dim=0)
    nll = nll_1 + nll_2 #scores.diag().mean() - 0.5 * scores.logsumexp(dim=1) - 0.5 * scores.logsumexp(dim=0)

    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()

    mi_i1 = 0.5*torch.tensor(scores.size(0)).float().log() + nll_1
    mi_i1 = mi_i1.mean()
    mi_i2 = 0.5*torch.tensor(scores.size(0)).float().log() + nll_2
    mi_i2 = mi_i2.mean()

    return mi, mi_i1, mi_i2

# def clip_lower_bound(scores: torch.Tensor):
#     """
#     scores: [N, N] matrix with (i,i) the positive pair.
#     Returns: total symmetric MI lower bound, and the row/column halves.
#     """
#     N = scores.size(0)
#     diag = scores.diag()                  # shape [N]
#     lse_row = scores.logsumexp(dim=1)     # shape [N]
#     lse_col = scores.logsumexp(dim=0)     # shape [N]

#     # Each directional InfoNCE lower bound: log N + E[diag - logsumexp]
#     mi_row = 0.5 * (np.log(N) + (diag - lse_row).mean())
#     mi_col = 0.5 * (np.log(N) + (diag - lse_col).mean())

#     mi_total = mi_row + mi_col
#     return mi_total, mi_row, mi_col

def smile_lower_bound(f, clip=None):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js
def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term
    
def logmeanexp_diag(x, device=device):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)


def logmeanexp_nodiag(x, dim=None, device=device):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)