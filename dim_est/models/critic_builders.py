# critic_builders.py
from typing import Dict, Any, Tuple
import torch.nn as nn
import copy

from ..models.critics import (
    SeparableCritic,
    SeparableAugmentedCritic,
    HybridCritic,
    ConcatCritic,
    BiCritic,
)
from .encoders import make_encoder


def _prepare_encoder_kwargs(cfg: dict, prefix: str, activation: str) -> dict:
    """
    Helper to extract legacy MLP args (e.g. x_hidden_dim) and merge with new encoder_kwargs.
    """
    # Base kwargs from config
    kwargs = copy.deepcopy(cfg.get("encoder_kwargs", {}))
    
    # 1. Inject activation if not present
    if "activation" not in kwargs:
        kwargs["activation"] = activation

    # 2. Map legacy keys "x_hidden_dim" -> "hidden_dim"
    legacy_hidden = cfg.get(f"{prefix}_hidden_dim")
    if legacy_hidden is not None and "hidden_dim" not in kwargs:
        kwargs["hidden_dim"] = legacy_hidden

    legacy_layers = cfg.get(f"{prefix}_layers")
    if legacy_layers is not None and "layers" not in kwargs:
        kwargs["layers"] = legacy_layers

    return kwargs


def build_separable_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    activation: str = "leaky_relu",
    encoder_type: str = "mlp",
    share_encoder: bool = False,
    encoder_kwargs: dict = None,
    **kwargs # Catch legacy args not explicitly used
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:

    # Prepare merged kwargs
    # We reconstruct a 'cfg' dict to pass to helper for legacy extraction
    cfg_proxy = {**kwargs, "encoder_kwargs": encoder_kwargs or {}}
    
    x_kwargs = _prepare_encoder_kwargs(cfg_proxy, "x", activation)
    y_kwargs = _prepare_encoder_kwargs(cfg_proxy, "y", activation)

    # Build X Encoder
    encoder_x = make_encoder(encoder_type, input_dim=Nx, embed_dim=embed_dim, **x_kwargs)
    
    # Build Y Encoder
    if share_encoder:
        encoder_y = encoder_x
    else:
        encoder_y = make_encoder(encoder_type, input_dim=Ny, embed_dim=embed_dim, **y_kwargs)

    critic = SeparableCritic(encoder_x, encoder_y)

    params = {
        "Nx": Nx, "Ny": Ny, "embed_dim": embed_dim,
        "activation": activation,
        "encoder_type": encoder_type,
        "share_encoder": share_encoder,
        "encoder_kwargs": encoder_kwargs,
        **kwargs
    }
    tags = {"critic_type": "separable", "embed_dim": embed_dim, "encoder": encoder_type}
    return critic, params, tags


def build_bilinear_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    activation: str = "leaky_relu",
    encoder_type: str = "mlp",
    share_encoder: bool = False,
    encoder_kwargs: dict = None,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:

    cfg_proxy = {**kwargs, "encoder_kwargs": encoder_kwargs or {}}
    x_kwargs = _prepare_encoder_kwargs(cfg_proxy, "x", activation)
    y_kwargs = _prepare_encoder_kwargs(cfg_proxy, "y", activation)

    encoder_x = make_encoder(encoder_type, input_dim=Nx, embed_dim=embed_dim, **x_kwargs)
    if share_encoder:
        encoder_y = encoder_x
    else:
        encoder_y = make_encoder(encoder_type, input_dim=Ny, embed_dim=embed_dim, **y_kwargs)

    critic = BiCritic(encoder_x, encoder_y, embed_dim=embed_dim)

    params = {
        "Nx": Nx, "Ny": Ny, "embed_dim": embed_dim,
        "activation": activation,
        "encoder_type": encoder_type,
        "share_encoder": share_encoder,
        "encoder_kwargs": encoder_kwargs,
        **kwargs
    }
    tags = {"critic_type": "bi", "embed_dim": embed_dim, "encoder": encoder_type}
    return critic, params, tags


def build_separable_augmented_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    activation: str = "leaky_relu",
    quad_kind: str = "full",
    encoder_type: str = "mlp",
    share_encoder: bool = False,
    encoder_kwargs: dict = None,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:

    cfg_proxy = {**kwargs, "encoder_kwargs": encoder_kwargs or {}}
    x_kwargs = _prepare_encoder_kwargs(cfg_proxy, "x", activation)
    y_kwargs = _prepare_encoder_kwargs(cfg_proxy, "y", activation)

    encoder_x = make_encoder(encoder_type, input_dim=Nx, embed_dim=embed_dim, **x_kwargs)
    if share_encoder:
        encoder_y = encoder_x
    else:
        encoder_y = make_encoder(encoder_type, input_dim=Ny, embed_dim=embed_dim, **y_kwargs)

    critic = SeparableAugmentedCritic(
        encoder_x=encoder_x,
        encoder_y=encoder_y,
        embed_dim=embed_dim,
        quad_kind=quad_kind,
    )

    params = {
        "Nx": Nx, "Ny": Ny, "embed_dim": embed_dim,
        "activation": activation, "quad_kind": quad_kind,
        "encoder_type": encoder_type,
        "share_encoder": share_encoder,
        "encoder_kwargs": encoder_kwargs,
        **kwargs
    }
    tags = {"critic_type": "separable_augmented", "embed_dim": embed_dim, "quad_kind": quad_kind, "encoder": encoder_type}
    return critic, params, tags


def build_hybrid_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    activation: str = "leaky_relu",
    pair_hidden_dim: int = 128,
    pair_layers: int = 2,
    encoder_type: str = "mlp",
    share_encoder: bool = False,
    encoder_kwargs: dict = None,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:

    assert embed_dim is None or embed_dim > 0, \
        f"Hybrid critic requires embed_dim > 0; got {embed_dim}"
    
    # 1. Encoders
    cfg_proxy = {**kwargs, "encoder_kwargs": encoder_kwargs or {}}
    x_kwargs = _prepare_encoder_kwargs(cfg_proxy, "x", activation)
    y_kwargs = _prepare_encoder_kwargs(cfg_proxy, "y", activation)

    encoder_x = make_encoder(encoder_type, input_dim=Nx, embed_dim=embed_dim, **x_kwargs)
    if share_encoder:
        encoder_y = encoder_x
    else:
        encoder_y = make_encoder(encoder_type, input_dim=Ny, embed_dim=embed_dim, **y_kwargs)

    # 2. Pair MLP (Always MLP for Hybrid Head)
    from ..utils.networks import mlp
    pair_mlp = mlp(
        dim=embed_dim + embed_dim,
        hidden_dim=pair_hidden_dim,
        output_dim=1,
        layers=pair_layers,
        activation=activation,
    )
    critic = HybridCritic(encoder_x=encoder_x, encoder_y=encoder_y, pair_mlp=pair_mlp)

    params = {
        "Nx": Nx, "Ny": Ny, "embed_dim": embed_dim,
        "activation": activation,
        "pair_hidden_dim": pair_hidden_dim, "pair_layers": pair_layers,
        "encoder_type": encoder_type,
        "share_encoder": share_encoder,
        "encoder_kwargs": encoder_kwargs,
        **kwargs
    }
    tags = {"critic_type": "hybrid", "embed_dim": embed_dim, "encoder": encoder_type}
    return critic, params, tags


def build_concat_critic(
    *,
    Nx: int,
    Ny: int,
    pair_hidden_dim: int = 128,
    pair_layers: int = 2,
    activation: str = "leaky_relu",
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Concat critic typically doesn't use separate encoders, so we ignore encoder_type here
    unless we want to add feature extraction before concatenation later.
    """
    from ..utils.networks import mlp
    pair_mlp = mlp(
        dim=Nx + Ny,
        hidden_dim=pair_hidden_dim,
        output_dim=1,
        layers=pair_layers,
        activation=activation,
    )
    critic = ConcatCritic(Nx=Nx, Ny=Ny, pair_mlp=pair_mlp)

    params = {
        "Nx": Nx, "Ny": Ny,
        "pair_hidden_dim": pair_hidden_dim, "pair_layers": pair_layers,
        "activation": activation,
        **kwargs
    }
    tags = {"critic_type": "concat"}
    return critic, params, tags


_CRITIC_BUILDERS = {
    "separable": build_separable_critic,
    "bi": build_bilinear_critic,
    "separable_augmented": build_separable_augmented_critic,
    "hybrid": build_hybrid_critic,
    "concat": build_concat_critic,
}


def make_critic(critic_type: str, critic_cfg: dict):
    if critic_type not in _CRITIC_BUILDERS:
        raise ValueError(f"Unknown critic_type: {critic_type}")

    builder = _CRITIC_BUILDERS[critic_type]
    return builder(**critic_cfg)