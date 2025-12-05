import numpy as np
import torch
import math
from .transform_observation import build_observation_transform

# latent keys depend on dataset_type
_ALLOWED_DATASET_LATENT_KEYS = {
    "gaussian_mixture": {"latent_dim", "n_peaks", "mu", "sig", "mi_bits_peak"},
    "joint_gaussian":   {"latent_dim", "mi_bits"},
    "ring_with_spread": {"latent_dim", "mu", "radial_std"},
    "hyperspherical_shell": {"latent_dim", "mu", "radial_std"},
    "swiss_roll": {"latent_dim", "t_min_pi_units", "t_max_pi_units", "height_min", "height_max"},
}

# transform keys depend on transform["mode"]
_ALLOWED_TRANSFORM_KEYS = {
    "teacher": {"mode",  "observe_dim_x", "observe_dim_y", "sig_embed_x", "sig_embed_y", "noise_mode"},
    "identity": {"mode", "sig_embed_x", "sig_embed_y", "noise_mode"},
    "linear": {"mode", "observe_dim_x", "observe_dim_y", "sig_embed_x", "sig_embed_y", "noise_mode"},
}



def make_data_generator(dataset_type: str, dataset_cfg: dict, device="cuda", dtype=torch.float32, validate_keys: bool = True):
    """Return a function that generates batches."""

    ## make sure the passed keys are sensible, also catches any typos in parameter specification
    if validate_keys:
        _validate_dataset_cfg(dataset_type, dataset_cfg)

    latent_cfg   = dataset_cfg["latent"]
    transform_cfg = dataset_cfg["transform"]



    # check whether there is a function for the latent dataset required, catch typos
    try:
        sample_latent_pair = LATENT_SAMPLERS[dataset_type]
    except KeyError:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Available latent samplers: {list(LATENT_SAMPLERS.keys())}"
        )

    latent_dim_x = latent_cfg["latent_dim"]
    latent_dim_y = latent_cfg["latent_dim"]

    # build a *fixed* transform callable for this experiment -- all data batches are passed through the same transform
    transform = build_observation_transform(latent_dim_x=latent_dim_x, latent_dim_y=latent_dim_y, transform_cfg=transform_cfg, device=device)

    def data_generator(batch_size: int):
        x_latent, y_latent = sample_latent_pair(batch_size=batch_size, latent_cfg=latent_cfg, device=device, dtype=dtype)
        x_obs, y_obs = transform(x_latent, y_latent)
        return x_obs, y_obs

    return data_generator
    
def _validate_dataset_cfg(dataset_type: str, dataset_cfg: dict) -> None:
    """Check that latent/transform keys are valid for this dataset + mode."""
    try:
        allowed_latent = _ALLOWED_DATASET_LATENT_KEYS[dataset_type]
    except KeyError:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    if "latent" not in dataset_cfg or "transform" not in dataset_cfg:
        raise ValueError(
            f"dataset_cfg for '{dataset_type}' must have 'latent' and 'transform' sub-dicts"
        )

    latent_cfg = dataset_cfg["latent"]
    transform_cfg = dataset_cfg["transform"]

    # latent keys
    extra_latent = set(latent_cfg.keys()) - allowed_latent
    if extra_latent:
        raise ValueError(
            f"Invalid latent parameters {extra_latent} for dataset '{dataset_type}'. "
            f"Allowed latent keys: {allowed_latent}"
        )

    # transform keys
    mode = transform_cfg.get("mode", None)
    if mode is None:
        raise ValueError(f"'transform.mode' must be specified for dataset '{dataset_type}'")
    if mode not in _ALLOWED_TRANSFORM_KEYS:
        raise ValueError(
            f"Unknown transform mode '{mode}' for dataset '{dataset_type}'. "
            f"Allowed modes: {set(_ALLOWED_TRANSFORM_KEYS.keys())}"
        )

    allowed_transform = _ALLOWED_TRANSFORM_KEYS[mode]
    extra_transform = set(transform_cfg.keys()) - allowed_transform
    if extra_transform:
        raise ValueError(
            f"Invalid transform parameters {extra_transform} for dataset '{dataset_type}' "
            f"with mode '{mode}'. Allowed transform keys: {allowed_transform}"
        )

def sample_gaussian_mixture_latents(batch_size: int, latent_cfg: dict, device = "cuda", dtype=torch.float32):
    """
    n-peak mixture in R^2 with equal weights.
    Peak k is centered on a ring at angle φ_k = 2πk/n with mean m_k = mu*[cos φ_k, sin φ_k].
    By default, each component is oriented *tangentially* to the ring: θ_k = φ_k + π/2.

    Component k covariance:
        base C_k = [[σ_k^2, ρ_k σ_k^2],
                    [ρ_k σ_k^2, σ_k^2]]
        Σ_k = R(θ_k) · C_k · R(θ_k)^T

    Returns:
        x, y: [B, 1], [B, 1] 
    """

    #####

    expected_dim = latent_cfg["latent_dim"]
    if expected_dim != 1:
        raise ValueError(
            f"gaussian_mixture in this setup requires latent_dim=1, got latent_dim={expected_dim}"
        )

    n_peaks=latent_cfg["n_peaks"]
    mu=latent_cfg["mu"]
    sig=latent_cfg["sig"]
    mi_bits_peak=latent_cfg["mi_bits_peak"]

    mi_nats_peak = mi_bits_peak * math.log(2) # --> individual cluster mi in nats
    rho = math.sqrt(1.0 - math.exp(-2.0 * mi_nats_peak )) ## corresponding rho, note this distrubtion is only defined for 1d


    eps = 1e-7
    to1 = lambda v: torch.as_tensor(v, device=device, dtype=dtype)

    def to_vec_n(v):
        t = to1(v)
        if t.ndim == 0:
            return t.repeat(n_peaks)
        if t.numel() != n_peaks:
            raise ValueError(f"Expected scalar or length-{n_peaks} for per-component parameter.")
        return t

    # Angles for equally spaced means on the circle
    k = torch.arange(n_peaks, device=device, dtype=dtype)
    phi = 2 * torch.pi * k / n_peaks  # [n]

    mu = to1(mu)
    sig = to_vec_n(sig)
    rho = torch.clamp(to_vec_n(rho), -1 + eps, 1 - eps)

    # Tangential orientation by default
    theta = torch.tensor(0.0, device=phi.device)

    # if theta is None:
    #     theta = torch.zeros_like(phi)   # or 
    # elif isinstance(theta, str) and theta.lower() == "tangential":
    #     theta = phi + torch.pi / 4
    # else:
    #     theta = to_vec_n(theta)

    # Means on the ring (n, 2)
    means = torch.stack([mu * torch.cos(phi), mu * torch.sin(phi)], dim=-1)

    # Base covariances for each component (n, 2, 2)
    s2 = sig ** 2  # (n,)
    cov_base = torch.stack([
        torch.stack([s2,            rho * s2], dim=-1),
        torch.stack([rho * s2,      s2      ], dim=-1),
    ], dim=-2)  # (n, 2, 2)

    # Rotation matrices per component (n, 2, 2)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([
        torch.stack([c,  -s], dim=-1),
        torch.stack([s,   c], dim=-1),
    ], dim=-2)  # (n, 2, 2)

    # Oriented covariances Σ_k and Cholesky factors (n, 2, 2)
    Sigma = R @ cov_base @ R.transpose(-1, -2)
    eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)  # (1,2,2)
    L = torch.linalg.cholesky(Sigma + eps * eye2)  # (n, 2, 2)

    # Sample component uniformly and draw points
    comp = torch.randint(0, n_peaks, (batch_size,), device=device)
    z0 = torch.randn(batch_size, 2, device=device, dtype=dtype)

    Lb = L[comp]         # (B, 2, 2)
    mb = means[comp]     # (B, 2)
    xy = mb + (Lb @ z0.unsqueeze(-1)).squeeze(-1)

    x, y = xy[:, 0], xy[:, 1]

    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    return x, y
        
def sample_joint_gaussian_latents(batch_size: int, latent_cfg: dict, device = "cuda", dtype=torch.float32):
    """
    Generates joint gaussian data
    """

    mi_bits = latent_cfg["mi_bits"] ## total mutual information
    latent_dim = latent_cfg["latent_dim"] ## equally spread corrlation in latent_dims

    mi_nats = mi_bits * math.log(2) # mi in nats
    rho = math.sqrt(1.0 - math.exp(-2.0 * mi_nats/latent_dim)) ## corresponding rho -- equal rhos

    cov = np.eye(2 * latent_dim)
    cov[latent_dim:, :latent_dim] = np.eye(latent_dim) * rho
    cov[:latent_dim, latent_dim:] = np.eye(latent_dim) * rho
    latents = np.random.multivariate_normal(np.zeros(2 * latent_dim), cov, batch_size)
    z_x, z_y = latents[:, :latent_dim], latents[:, latent_dim:]
    z_x_tensor = torch.tensor(z_x, dtype=dtype, device=device)
    z_y_tensor = torch.tensor(z_y, dtype=dtype, device=device)

    return z_x_tensor, z_y_tensor

def sample_ring_latents(batch_size: int, latent_cfg: dict, device = "cuda", dtype=torch.float32):
    """
    Sample points in R^2 that lie approximately on a circle of radius `mu`,
    with finite radial spread `radial_std`.

    Construction:
      φ ~ Uniform(0, 2π)
      r = mu + radial_std * ε,  ε ~ N(0, 1)
      x = r cos φ
      y = r sin φ

    Returns:
        x, y
        
    Shapes:
        x:   (batch_size, 1)
        y:   (batch_size, 1)
        phi: (batch_size,)
    """
    expected_dim = latent_cfg["latent_dim"]
    if expected_dim != 1:
        raise ValueError(
            f"ring latents requires latent_dim=1, got latent_dim={expected_dim}"
        )
    # sample angles uniformly around the circle

    mu = latent_cfg["mu"]       # mean radius of the ring
    radial_std = latent_cfg["radial_std"]  # spread around the ring radius

    phi = torch.rand(batch_size, device=device, dtype=dtype) * 2 * torch.pi  # (B,)

    # sample radius with Gaussian spread around mu
    if radial_std > 0.0:
        r = mu + radial_std * torch.randn(batch_size, device=device, dtype=dtype)
    else:
        r = torch.full((batch_size,), mu, device=device, dtype=dtype)

    # convert to Cartesian
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)

    # keep the (B, 1) shape like your old code
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    
    return x, y

def sample_shell_latents(batch_size: int, latent_cfg: dict, device = "cuda", dtype=torch.float32):
    """
    Sample a shared latent Z on a d-dimensional hyperspherical shell,

    Latent construction in R^d:
      u ~ N(0, I_d), then normalize => u / ||u||
      if radial_std == 0:
          r = radius
      else:
          r = radius + radial_std * eps, eps ~ N(0,1)
      Z = r * u

    Returns:
        z, z  (both shape: (batch_size, d), (batch_size, d))
    """

    latent_dim= latent_cfg["latent_dim"]
    radius = latent_cfg["mu"]
    radial_std = latent_cfg["radial_std"]

    # ---- 1. sample directions ~ uniform on the unit sphere in R^d: draw standard Gaussian vectors and normalize to unit length
    z_raw = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)  # (B, d)
    z_raw_norm = z_raw.norm(dim=1, keepdim=True)                             # (B, 1)
    # avoid divide-by-zero, though probability is basically zero
    z_unit = z_raw / (z_raw_norm + 1e-8)                                     # (B, d)

    # ---- 2. choose radius: exact shell or thick shell ----
    if radial_std > 0.0:
        # r ~ N(radius, radial_std^2)
        r = radius + radial_std * torch.randn(batch_size, 1, device=device, dtype=dtype)  # (B, 1)
    else:
        r = torch.full((batch_size, 1), radius, device=device, dtype=dtype)               # (B, 1)

    # ---- 3. latent on hyperspherical shell ----
    z_latent = r * z_unit   # (B, d)

    zx = z_latent
    zy = z_latent.clone()

    return zx, zy

def sample_swiss_roll_latents(batch_size: int, latent_cfg: dict, device = "cuda", dtype=torch.float32):
    """
    Sample points on a Swiss roll surface in R^3, then optionally embed
    them with two MLPs and inject noise in embedding space.

    Standard Swiss roll parameterization:

        t ~ Uniform[t_min, t_max]
        h ~ Uniform[height_min, height_max]

        X3D = ( x, y, z ) with
            x = t * cos(t)
            y = h
            z = t * sin(t)

    Views:
        - Both X and Y see the same underlying 3D point (shared 2D manifold),
          but can be mapped differently by mlp_x, mlp_y and then noised.

    Returns:
        swiss_3d, swiss_3d: (batch_size, 3), (batch_size, 3)
    """

    expected_dim = latent_cfg["latent_dim"]
    if expected_dim != 3:
        raise ValueError(
            f"swiss_roll requires latent_dim=3, got latent_dim={expected_dim}"
        )

    t_min_pi_units = latent_cfg["t_min_pi_units"]
    t_max_pi_units = latent_cfg["t_max_pi_units"]
    height_min = latent_cfg["height_min"]
    height_max = latent_cfg["height_max"]


    t_min = t_min_pi_units * math.pi
    t_max = t_max_pi_units * math.pi

    # ---- 1. Sample latent parameters (t, h) ----
    t = torch.rand(batch_size, device=device, dtype=dtype)
    t = t_min + (t_max - t_min) * t                # (B,)

    h = torch.rand(batch_size, device=device, dtype=dtype)
    h = height_min + (height_max - height_min) * h # (B,)

    # ---- 2. Map to 3D Swiss roll coordinates ----
    x = t * torch.cos(t)   # (B,)
    y = h                  # (B,)
    z = t * torch.sin(t)   # (B,)

    swiss_3d = torch.stack([x, y, z], dim=1)       # (B, 3)

    zx = swiss_3d
    zy =  swiss_3d.clone()

    return zx, zy


LATENT_SAMPLERS = {
    "gaussian_mixture": sample_gaussian_mixture_latents,
    "joint_gaussian": sample_joint_gaussian_latents,
    "ring_with_spread": sample_ring_latents,
    "hyperspherical_shell": sample_shell_latents,
    "swiss_roll": sample_swiss_roll_latents,
}
