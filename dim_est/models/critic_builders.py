# critic_builders.py
from typing import Dict, Any, Tuple
import torch.nn as nn

from ..utils.networks import mlp

from ..models.critics import (
    SeparableCritic,
    SeparableAugmentedCritic,
    HybridCritic,
    ConcatCritic,
    BiCritic,
)


def build_separable_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    x_hidden_dim: int = 64,
    x_layers: int = 2,
    y_hidden_dim: int = 64,
    y_layers: int = 2,
    activation: str = "leaky_relu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Build a SeparableCritic with potentially different x/y encoder architectures.
    """
    encoder_x = mlp(
        dim=Nx,
        hidden_dim=x_hidden_dim,
        output_dim=embed_dim,
        layers=x_layers,
        activation=activation,
    )
    encoder_y = mlp(
        dim=Ny,
        hidden_dim=y_hidden_dim,
        output_dim=embed_dim,
        layers=y_layers,
        activation=activation,
    )
    critic = SeparableCritic(encoder_x, encoder_y)

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "embed_dim": embed_dim,
        "x_hidden_dim": x_hidden_dim,
        "x_layers": x_layers,
        "y_hidden_dim": y_hidden_dim,
        "y_layers": y_layers,
        "activation": activation,
    }
    tags = {"critic_type": "separable", "embed_dim": embed_dim}
    return critic, params, tags


def build_bilinear_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    x_hidden_dim: int = 64,
    x_layers: int = 2,
    y_hidden_dim: int = 64,
    y_layers: int = 2,
    activation: str = "leaky_relu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    encoder_x = mlp(
        dim=Nx,
        hidden_dim=x_hidden_dim,
        output_dim=embed_dim,
        layers=x_layers,
        activation=activation,
    )
    encoder_y = mlp(
        dim=Ny,
        hidden_dim=y_hidden_dim,
        output_dim=embed_dim,
        layers=y_layers,
        activation=activation,
    )
    critic = BiCritic(encoder_x, encoder_y, embed_dim=embed_dim)

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "embed_dim": embed_dim,
        "x_hidden_dim": x_hidden_dim,
        "x_layers": x_layers,
        "y_hidden_dim": y_hidden_dim,
        "y_layers": y_layers,
        "activation": activation,
    }
    tags = {"critic_type": "bi", "embed_dim": embed_dim}
    return critic, params, tags


def build_separable_augmented_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    x_hidden_dim: int = 64,
    x_layers: int = 2,
    y_hidden_dim: int = 64,
    y_layers: int = 2,
    activation: str = "leaky_relu",
    quad_kind: str = "full",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    encoder_x = mlp(
        dim=Nx,
        hidden_dim=x_hidden_dim,
        output_dim=embed_dim,
        layers=x_layers,
        activation=activation,
    )
    encoder_y = mlp(
        dim=Ny,
        hidden_dim=y_hidden_dim,
        output_dim=embed_dim,
        layers=y_layers,
        activation=activation,
    )
    critic = SeparableAugmentedCritic(
        encoder_x=encoder_x,
        encoder_y=encoder_y,
        embed_dim=embed_dim,
        quad_kind=quad_kind,
    )

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "embed_dim": embed_dim,
        "x_hidden_dim": x_hidden_dim,
        "x_layers": x_layers,
        "y_hidden_dim": y_hidden_dim,
        "y_layers": y_layers,
        "activation": activation,
        "quad_kind": quad_kind,
    }
    tags = {"critic_type": "separable_augmented", "embed_dim": embed_dim, "quad_kind": quad_kind}
    return critic, params, tags


def build_hybrid_critic(
    *,
    Nx: int,
    Ny: int,
    embed_dim: int,
    x_hidden_dim: int = 64,
    x_layers: int = 2,
    y_hidden_dim: int = 64,
    y_layers: int = 2,
    pair_hidden_dim: int = 128,
    pair_layers: int = 2,
    activation: str = "leaky_relu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Separate encoders for x and y, plus a separate pair MLP for concat(zX, zY).
    """

    assert embed_dim is None or embed_dim > 0, \
        f"Hybrid critic requires embed_dim > 0; got {embed_dim}"
        
    encoder_x = mlp(
        dim=Nx,
        hidden_dim=x_hidden_dim,
        output_dim=embed_dim,
        layers=x_layers,
        activation=activation,
    )
    encoder_y = mlp(
        dim=Ny,
        hidden_dim=y_hidden_dim,
        output_dim=embed_dim,
        layers=y_layers,
        activation=activation,
    )
    pair_mlp = mlp(
        dim=embed_dim + embed_dim,
        hidden_dim=pair_hidden_dim,
        output_dim=1,
        layers=pair_layers,
        activation=activation,
    )
    critic = HybridCritic(encoder_x=encoder_x, encoder_y=encoder_y, pair_mlp=pair_mlp)

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "embed_dim": embed_dim,
        "x_hidden_dim": x_hidden_dim,
        "x_layers": x_layers,
        "y_hidden_dim": y_hidden_dim,
        "y_layers": y_layers,
        "pair_hidden_dim": pair_hidden_dim,
        "pair_layers": pair_layers,
        "activation": activation,
    }
    tags = {"critic_type": "hybrid", "embed_dim": embed_dim}
    return critic, params, tags


def build_concat_critic(
    *,
    Nx: int,
    Ny: int,
    pair_hidden_dim: int = 128,
    pair_layers: int = 2,
    activation: str = "leaky_relu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    No encoders. Single MLP on concat(x, y).
    """
    pair_mlp = mlp(
        dim=Nx + Ny,
        hidden_dim=pair_hidden_dim,
        output_dim=1,
        layers=pair_layers,
        activation=activation,
    )
    critic = ConcatCritic(Nx=Nx, Ny=Ny, pair_mlp=pair_mlp)

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "pair_hidden_dim": pair_hidden_dim,
        "pair_layers": pair_layers,
        "activation": activation,
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


def make_critic(critic_type: str, **kwargs):
    """
    Main entry point.

    Example:
        critic, params, tags = make_critic(
            "hybrid",
            Nx=500, Ny=500,
            embed_dim=16,
            x_hidden_dim=64, x_layers=2,
            y_hidden_dim=128, y_layers=3,
            pair_hidden_dim=256, pair_layers=2,
            activation="leaky_relu",
        )
    """
    if critic_type not in _CRITIC_BUILDERS:
        raise ValueError(f"Unknown critic_type: {critic_type}")
    return _CRITIC_BUILDERS[critic_type](**kwargs)
