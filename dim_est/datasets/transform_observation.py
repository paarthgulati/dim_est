import torch
from torch import nn
from ..utils.networks import teacher


def embedding_noise_injector(x: torch.Tensor,
                             sig_embed: float = 0.0,
                             noise_mode: str = "white_relative") -> torch.Tensor:
    if sig_embed <= 0:
        return x

    if noise_mode == "white_relative":
        scale_x = x.std().clamp_min(1e-8)
        noise_x = sig_embed * scale_x * torch.randn_like(x)
    elif noise_mode == "white_absolute":
        noise_x = sig_embed * torch.randn_like(x)
    else:
        raise ValueError(
            f"Unknown noise_mode '{noise_mode}'. "
            "Choose from {'white_relative', 'white_absolute'}."
        )
    return x + noise_x


def build_observation_transform(latent_dim_x: int, latent_dim_y: int, transform_cfg: dict, device=None):
    """
    Construct a fixed transform (x, y) -> (x_obs, y_obs) for this experiment.
    Returns a callable that can be used for every batch.
    """

    mode = transform_cfg["mode"]

    if mode == "teacher":
        Nx = transform_cfg["observe_dim_x"]
        Ny = transform_cfg["observe_dim_y"]

        teacher_model_x = teacher(dz=latent_dim_x, output_dim=Nx)
        teacher_model_y = teacher(dz=latent_dim_y, output_dim=Ny)

        if device is not None:
            teacher_model_x = teacher_model_x.to(device)
            teacher_model_y = teacher_model_y.to(device)

        for p in teacher_model_x.parameters():
            p.requires_grad_(False)
        for p in teacher_model_y.parameters():
            p.requires_grad_(False)

        return TeacherTransform(teacher_model_x, teacher_model_y, transform_cfg)

    elif mode == "identity":
        obs_x = transform_cfg.get("observe_dim_x")
        obs_y = transform_cfg.get("observe_dim_y")
        
        if obs_x is not None or obs_y is not None:
             raise ValueError(
                f"Invalid transform parameters; For identity transform.observe_dim_x and "
                f"transform.observe_dim_y should be None (or omitted). "
                f"Instead got {obs_x} and {obs_y}"
            )

        return IdentityTransform(transform_cfg)

    elif mode == "linear":

        Nx = transform_cfg["observe_dim_x"]
        Ny = transform_cfg["observe_dim_y"]

        if Nx is None or Ny is None:
            raise ValueError("Linear mode requires observe_dim_x and observe_dim_y.")

        # Random linear projections
        A_x = torch.nn.Linear(latent_dim_x, Nx, bias=False)
        A_y = torch.nn.Linear(latent_dim_y, Ny, bias=False)

        if device is not None:
            A_x = A_x.to(device)
            A_y = A_y.to(device)

        # Freeze them (match teacher behavior)
        for p in A_x.parameters(): p.requires_grad_(False)
        for p in A_y.parameters(): p.requires_grad_(False)

        return LinearTransform(A_x, A_y, transform_cfg)


    else:
        raise ValueError(f"Unknown transform mode {mode!r}")



class IdentityTransform(nn.Module):
    def __init__(self, transform_cfg: dict):
        super().__init__()
        self.sig_x = float(transform_cfg.get("sig_embed_x", 0.0) or 0.0)
        self.sig_y = float(transform_cfg.get("sig_embed_y", 0.0) or 0.0)
        self.noise_mode = transform_cfg.get("noise_mode", "white_relative")

    def forward(self, x, y):
        x_noisy = embedding_noise_injector(x, sig_embed=self.sig_x, noise_mode=self.noise_mode)
        y_noisy = embedding_noise_injector(y, sig_embed=self.sig_y, noise_mode=self.noise_mode)
        return x_noisy, y_noisy


class TeacherTransform(nn.Module):
    def __init__(self, teacher_model_x: nn.Module, teacher_model_y: nn.Module, transform_cfg: dict):
        super().__init__()
        self.teacher_model_x = teacher_model_x
        self.teacher_model_y = teacher_model_y
        self.sig_x = float(transform_cfg.get("sig_embed_x", 0.0) or 0.0)
        self.sig_y = float(transform_cfg.get("sig_embed_y", 0.0) or 0.0)
        self.noise_mode = transform_cfg.get("noise_mode", "white_relative")

    def forward(self, x, y):
        # assumes x,y already on correct device
        with torch.no_grad():
            x_embed = self.teacher_model_x(x)
            y_embed = self.teacher_model_y(y)

        x_embed_noisy = embedding_noise_injector(x_embed, sig_embed=self.sig_x, noise_mode=self.noise_mode)
        y_embed_noisy = embedding_noise_injector(y_embed, sig_embed=self.sig_y, noise_mode=self.noise_mode)
        return x_embed_noisy, y_embed_noisy


class LinearTransform(nn.Module):
    def __init__(self, A_x: nn.Module, A_y: nn.Module, transform_cfg: dict):
        super().__init__()
        self.A_x = A_x
        self.A_y = A_y
        self.sig_x = float(transform_cfg.get("sig_embed_x", 0.0) or 0.0)
        self.sig_y = float(transform_cfg.get("sig_embed_y", 0.0) or 0.0)
        self.noise_mode = transform_cfg.get("noise_mode", "white_relative")

    def forward(self, x, y):
        with torch.no_grad():
            x_lin = self.A_x(x)
            y_lin = self.A_y(y)

        x_noisy = embedding_noise_injector(x_lin, sig_embed=self.sig_x, noise_mode=self.noise_mode)
        y_noisy = embedding_noise_injector(y_lin, sig_embed=self.sig_y, noise_mode=self.noise_mode)
        return x_noisy, y_noisy
