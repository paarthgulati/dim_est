import numpy as np
import torch
import math

_ALLOWED_DATASET_KEYS = {
    "gaussian_mixture": {"latent_dim", "n_peaks", "mu", "sig", "mi_bits_peak", "sig_embed", "noise_mode", "observe_dim_x", "observe_dim_y"},
    "joint_gaussian": {"latent_dim", "mi_bits", "sig_embed", "noise_mode", "observe_dim_x", "observe_dim_y"},
    "ring_with_spread": {"latent_dim", "mu", "radial_std", "sig_embed", "noise_mode", "observe_dim_x", "observe_dim_y"},
    "hyperspherical_shell": {"latent_dim", "mu", "radial_std", "sig_embed", "noise_mode", "observe_dim_x", "observe_dim_y"},
    "swiss_roll": {"latent_dim", "t_min_pi_units", "t_max_pi_units", "height_min", "height_max", "sig_embed", "noise_mode", "observe_dim_x", "observe_dim_y"},
}

def make_data_generator(
    dataset_type: str,
    dataset_cfg: dict,
    teacher_x=None,
    teacher_y=None,
    device="cuda",
    dtype=torch.float32,
    validate_keys: bool = True,
):
    """Return a function that generates batches."""

    # ---- Validate supported dataset type ----
    if dataset_type not in _ALLOWED_DATASET_KEYS:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Supported: {list(_ALLOWED_DATASET_KEYS)}")

    # ---- Optional: catch typos in config ----
    if validate_keys:
        allowed = _ALLOWED_DATASET_KEYS[dataset_type]
        extra = set(dataset_cfg.keys()) - allowed
        if extra:
            raise ValueError(
                f"Invalid parameters {extra} for dataset '{dataset_type}'. "
                f"Allowed keys: {allowed}"
            )

    # ---- Routing ----
    if dataset_type == "gaussian_mixture":
        if dataset_cfg["latent_dim"] != 1:
            raise ValueError(
                f"{dataset_type} has only been defined for latent dim == 1. Check your parameters"
            )
        return lambda B: sample_ring_mixture_embed_teacher(
            batch_size=B,
            n_peaks=dataset_cfg["n_peaks"],
            mu=dataset_cfg["mu"],
            sig=dataset_cfg["sig"],
            mi_bits_peak=dataset_cfg["mi_bits_peak"],
            mlp_x=teacher_x,
            mlp_y=teacher_y,
            sig_embed=dataset_cfg.get("sig_embed", 0.0),
            noise_mode=dataset_cfg.get("noise_mode", "white_relative"),
            device=device,
            dtype=dtype,
        )

    elif dataset_type == "joint_gaussian":
        return lambda B: sample_joint_gaussian_embed_teacher(
            batch_size=B,
            latent_dim=dataset_cfg["latent_dim"],
            mi_bits=dataset_cfg["mi_bits"],
            mlp_x=teacher_x,
            mlp_y=teacher_y,
            sig_embed=dataset_cfg.get("sig_embed", 0.0),
            noise_mode=dataset_cfg.get("noise_mode", "white_relative"),
            device=device,
        )

    elif dataset_type == "ring_with_spread":
        if dataset_cfg["latent_dim"] != 1:
            raise ValueError(
                f"{dataset_type} has only been defined for latent dim == 1. Check your parameters"
            )
        return lambda B: sample_ring_with_spread_embed_teacher(
            batch_size=B,
            mu=dataset_cfg["mu"],
            radial_std=dataset_cfg["radial_std"],
            mlp_x=teacher_x,
            mlp_y=teacher_y,
            sig_embed=dataset_cfg.get("sig_embed", 0.0),
            noise_mode=dataset_cfg.get("noise_mode", "white_relative"),
            device=device,
            dtype=dtype,
        )
    

    elif dataset_type == "hyperspherical_shell":
        return lambda B: sample_hyperspherical_shell_embed_teacher(
            batch_size=B,
            latent_dim=dataset_cfg["latent_dim"],
            radius=dataset_cfg["mu"],
            radial_std=dataset_cfg["radial_std"],
            mlp_x=teacher_x,
            mlp_y=teacher_y,
            sig_embed=dataset_cfg.get("sig_embed", 0.0),
            noise_mode=dataset_cfg.get("noise_mode", "white_relative"),
            device=device,
            dtype=dtype,
        )

    elif dataset_type == "swiss_roll":
        if dataset_cfg["latent_dim"] != 3:
            raise ValueError(
                f"{dataset_type} has only been defined for latent dim == 3. Check your parameters"
            )
        return lambda B: sample_swiss_roll_embed_teacher(
            batch_size=B,
            t_min_pi_units= dataset_cfg["t_min_pi_units"], 
            t_max_pi_units= dataset_cfg["t_max_pi_units"], 
            height_min= dataset_cfg["height_min"], 
            height_max= dataset_cfg["height_max"], 
            mlp_x=teacher_x,
            mlp_y=teacher_y,
            sig_embed=dataset_cfg.get("sig_embed", 0.0),
            noise_mode=dataset_cfg.get("noise_mode", "white_relative"),
            device=device,
            dtype=dtype,
        )


def sample_ring_mixture_embed_teacher(
    batch_size: int,
    n_peaks: int,
    mu: float = 4.0,
    sig=1.0,                 # scalar or length-n_peaks: σ_k (isotropic base scale)
    mi_bits_peak=0.0,                 # scalar or length-n_peaks: correlation ρ_k (in base frame)
    theta=None,              # if None: set along diagonal, if 'tangential' set tangent orientation θ_k = φ_k + π/2, else pass a list of len = n_peaks
    device="cuda",
    dtype=torch.float32,
    mlp_x = None, 
    mlp_y = None, 
    sig_embed: float= 0.0, # NEW: noise level (0 => no noise)
    noise_mode: str = "white_relative", # {"white_relative", "white_absolute"}
):
    """
    n-peak mixture in R^2 with equal weights.
    Peak k is centered on a ring at angle φ_k = 2πk/n with mean m_k = mu*[cos φ_k, sin φ_k].
    By default, each component is oriented *tangentially* to the ring: θ_k = φ_k + π/2.

    Component k covariance:
        base C_k = [[σ_k^2, ρ_k σ_k^2],
                    [ρ_k σ_k^2, σ_k^2]]
        Σ_k = R(θ_k) · C_k · R(θ_k)^T

    Returns:
        x_embed, y_embed  (possibly projected by mlp_x/mlp_y and noisy)
    """

    #####
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
    if theta is None:
        theta = torch.zeros_like(phi)   # or torch.tensor(0.0, device=phi.device)
    elif isinstance(theta, str) and theta.lower() == "tangential":
        theta = phi + torch.pi / 4
    else:
        theta = to_vec_n(theta)

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
    
    if (mlp_x is not None) and (mlp_y is not None):
        with torch.no_grad():
            x_embed = mlp_x(x)
            y_embed = mlp_y(y)
    else:
        x_embed = x
        y_embed = y

    # --- noise injection in embedding space ---
    x_embed, y_embed = embedding_noise_injector(x_embed, y_embed, sig_embed=sig_embed, noise_mode=noise_mode)

    return x_embed, y_embed
        
    # return x_embed, y_embed, comp
    
def sample_joint_gaussian_embed_teacher(
    batch_size: int, 
    latent_dim: int = 2, 
    mi_bits=0.0,
    mlp_x=None, 
    mlp_y=None, 
    dtype=torch.float32, 
    device="cuda", 
    sig_embed: float= 0.0, 
    noise_mode: str = "white_relative",
):
    """
    Generates joint gaussian data
    - dev put the batches on gpu if passed any argument
    """

    mi_nats = mi_bits * math.log(2) # --> individual cluster mi in nats
    rho = math.sqrt(1.0 - math.exp(-2.0 * mi_nats/latent_dim)) ## corresponding rho

    cov = np.eye(2 * latent_dim)
    cov[latent_dim:, :latent_dim] = np.eye(latent_dim) * rho
    cov[:latent_dim, latent_dim:] = np.eye(latent_dim) * rho
    latents = np.random.multivariate_normal(np.zeros(2 * latent_dim), cov, batch_size)
    z_x, z_y = latents[:, :latent_dim], latents[:, latent_dim:]
    z_x_tensor = torch.tensor(z_x, dtype=dtype, device=device)
    z_y_tensor = torch.tensor(z_y, dtype=dtype, device=device)

    if (mlp_x is not None) and (mlp_y is not None):
        with torch.no_grad():
            x_embed = mlp_x(z_x_tensor)
            y_embed = mlp_y(z_y_tensor)
    else:
        x_embed = z_x_tensor
        y_embed = z_y_tensor


    # --- noise injection in embedding space ---
    x_embed, y_embed = embedding_noise_injector(x_embed, y_embed, sig_embed=sig_embed, noise_mode=noise_mode)

    return x_embed, y_embed


def sample_ring_with_spread_embed_teacher(
    batch_size: int,
    mu: float = 4.0,          # mean radius of the ring
    radial_std: float = 0.1,  # spread around the ring radius
    mlp_x=None, 
    mlp_y=None,
    device="cuda",
    sig_embed: float= 0.0, 
    noise_mode: str = "white_relative",
    dtype: torch.dtype = torch.float32,
):
    """
    Sample points in R^2 that lie approximately on a circle of radius `mu`,
    with finite radial spread `radial_std`.

    Construction:
      φ ~ Uniform(0, 2π)
      r = mu + radial_std * ε,  ε ~ N(0, 1)
      x = r cos φ
      y = r sin φ

    Returns:
        x_embed, y_embed
        
    Shapes:
        x:   (batch_size, 1)
        y:   (batch_size, 1)
        phi: (batch_size,)
    """
    # sample angles uniformly around the circle
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
    

    if (mlp_x is not None) and (mlp_y is not None):
        with torch.no_grad():
            x_embed = mlp_x(x)
            y_embed = mlp_y(y)
    else:
        x_embed = x
        y_embed = y

     # --- noise injection in embedding space ---

    x_embed, y_embed = embedding_noise_injector(x_embed, y_embed, sig_embed=sig_embed, noise_mode=noise_mode)

    return x_embed, y_embed


def sample_hyperspherical_shell_embed_teacher(
    batch_size: int,
    latent_dim: int,
    radius: float = 4.0,        # mean radius of the shell
    radial_std: float = 0.1,    # 0.0 => exact shell, >0 => thick shell
    mlp_x=None,
    mlp_y=None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    sig_embed: float = 0.0,
    noise_mode: str = "white_relative",
):
    """
    Sample a shared latent Z on a d-dimensional hyperspherical shell,
    then pass it through two (optional) MLPs to generate x_embed, y_embed.

    Latent construction in R^d:
      u ~ N(0, I_d), then normalize => u / ||u||
      if radial_std == 0:
          r = radius
      else:
          r = radius + radial_std * eps, eps ~ N(0,1)
      Z = r * u

    Returns:
        x_embed, y_embed  (both shape: (batch_size, D_x), (batch_size, D_y))
    """

    # ---- 1. sample directions ~ uniform on the unit sphere in R^d ----
    # draw standard Gaussian vectors and normalize to unit length
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

    # ---- 4. pass through teacher MLPs (shared latent => two views) ----
    if (mlp_x is not None) and (mlp_y is not None):
        with torch.no_grad():
            x_embed = mlp_x(z_latent)   # (B, D_x)
            y_embed = mlp_y(z_latent)   # (B, D_y)
    else:
        # if you want raw latent as both "views"
        x_embed = z_latent
        y_embed = z_latent

    # ---- 5. optional noise injection in embedding space ----
    x_embed, y_embed = embedding_noise_injector(
        x_embed,
        y_embed,
        sig_embed=sig_embed,
        noise_mode=noise_mode,
    )

    return x_embed, y_embed


def sample_swiss_roll_embed_teacher(
    batch_size: int,
    t_min_pi_units: float = 1.5 ,   # start angle (in units of pi)
    t_max_pi_units: float = 4.5,   # end angle (in units of pi)
    height_min: float = 0.0,        # vertical extent
    height_max: float = 15.0,
    mlp_x=None,
    mlp_y=None,
    device: str = "cuda",
    sig_embed: float = 0.0,
    noise_mode: str = "white_relative",
    dtype: torch.dtype = torch.float32,
):
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
        x_embed, y_embed

    Shapes:
        swiss_3d: (batch_size, 3)
        x_embed:  (batch_size, D_x)  # D_x = output dim of mlp_x or 3 if None
        y_embed:  (batch_size, D_y)  # D_y = output dim of mlp_y or 3 if None
    """

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

    # ---- 3. Pass through teacher MLPs (shared latent -> two views) ----
    if (mlp_x is not None) and (mlp_y is not None):
        with torch.no_grad():
            x_embed = mlp_x(swiss_3d)  # (B, D_x)
            y_embed = mlp_y(swiss_3d)  # (B, D_y)
    else:
        # If no MLPs provided, raw Swiss roll coordinates are the "embeddings"
        x_embed = swiss_3d
        y_embed = swiss_3d

    # ---- 4. Inject noise in embedding space (observation noise) ----
    x_embed, y_embed = embedding_noise_injector(
        x_embed,
        y_embed,
        sig_embed=sig_embed,
        noise_mode=noise_mode,
    )

    return x_embed, y_embed


def embedding_noise_injector(x: torch.Tensor,
                   y: torch.Tensor,
                   sig_embed: float = 0.0,
                   noise_mode: str = "white_relative") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to (x, y) embeddings.

    Parameters
    ----------
    x, y : torch.Tensor
        Embedding tensors of shape [B, d] (or broadcastable).
    sig_embed : float
        Noise scale. If 0.0, no noise is added.
    noise_mode : str
        "white_relative" → noise std = sig_embed * std(x)
        "white_absolute" → noise std = sig_embed

    Returns
    -------
    (x_noisy, y_noisy) : tuple of torch.Tensor
        Noisy versions of input embeddings.
    """
    
    # Nothing to do
    if sig_embed <= 0:
        return x, y

    if noise_mode == "white_relative":
        # Scale relative to overall statistical scale of the embeddings
        scale_x = x.std().clamp_min(1e-8)  # scalar
        scale_y = y.std().clamp_min(1e-8)
        noise_x = sig_embed * scale_x * torch.randn_like(x)
        noise_y = sig_embed * scale_y * torch.randn_like(y)

    elif noise_mode == "white_absolute":
        # Fixed-scale Gaussian
        noise_x = sig_embed * torch.randn_like(x)
        noise_y = sig_embed * torch.randn_like(y)

    else:
        raise ValueError(f"Unknown noise_mode '{noise_mode}'. "
                         "Choose from {'white_relative', 'white_absolute'}.")

    return x + noise_x, y + noise_y


