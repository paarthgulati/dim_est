import torch
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

        def transform(x, y):
            # assume x, y are already on correct device
            with torch.no_grad():
                x_embed = teacher_model_x(x)
                y_embed = teacher_model_y(y)

            x_embed_noisy = embedding_noise_injector(x_embed, sig_embed=transform_cfg.get("sig_embed_x", 0.0), noise_mode=transform_cfg.get("noise_mode", "white_relative"))
            y_embed_noisy = embedding_noise_injector(y_embed, sig_embed=transform_cfg.get("sig_embed_y", 0.0), noise_mode=transform_cfg.get("noise_mode", "white_relative"))
            return x_embed_noisy, y_embed_noisy

        return transform

    elif mode == "identity":

        def transform(x, y):
            x_noisy = embedding_noise_injector(x, sig_embed=transform_cfg.get("sig_embed_x", 0.0), noise_mode=transform_cfg.get("noise_mode", "white_relative"))
            y_noisy = embedding_noise_injector(y, sig_embed=transform_cfg.get("sig_embed_y", 0.0), noise_mode=transform_cfg.get("noise_mode", "white_relative"))
            return x_noisy, y_noisy

        return transform

    else:
        raise ValueError(f"Unknown transform mode {mode!r}")
