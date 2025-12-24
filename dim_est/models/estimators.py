import numpy as np
import torch
import torch.nn.functional as F

def infonce_lower_bound(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Use scores.device to ensure compatibility
    mi = torch.tensor(scores.size(0), device=scores.device).float().log() + nll
    mi = mi.mean()
    extras = {}
    return mi, extras

def clip_lower_bound(scores):
    # nll terms
    nll_1 = 0.5 * scores.diag().mean() - 0.5 * scores.logsumexp(dim=1) 
    nll_2 = 0.5 * scores.diag().mean() - 0.5 * scores.logsumexp(dim=0)
    nll = nll_1 + nll_2 

    # Use scores.device
    log_N = torch.tensor(scores.size(0), device=scores.device).float().log()
    
    mi = log_N + nll
    mi = mi.mean()

    mi_i1 = 0.5*log_N + nll_1
    mi_i1 = mi_i1.mean()
    mi_i2 = 0.5*log_N + nll_2
    mi_i2 = mi_i2.mean()

    extras = dict(mi_i1 = mi_i1, mi_i2 = mi_i2)
    return mi, extras

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

    extras = {}
    return js + dv_js, extras

def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term
    
def logmeanexp_diag(x):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)
    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size
    return logsumexp - torch.log(torch.tensor(num_elem, device=x.device).float())

def logmeanexp_nodiag(x, dim=None):
    batch_size = x.size(0)
    device = x.device
    
    if dim is None:
        dim = (0, 1)

    # Mask diagonal with -inf
    mask = torch.diag(torch.full((batch_size,), float('inf'), device=device))
    logsumexp = torch.logsumexp(x - mask, dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        # fallback if dim was scalar
        num_elem = batch_size - 1
        
    return logsumexp - torch.log(torch.tensor(num_elem, device=device))