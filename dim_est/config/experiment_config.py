"""
experiment_config.py

Lightweight dataclasses describing the configuration of a single experiment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DatasetConfig:
    """
    Configuration for the data-generating process.
    `cfg` is expected to be the FULL dataset config dict
    """
    dataset_type: str        # e.g. "joint_gaussian", "gaussian_mixture", "swiss_roll"
    cfg: Dict[str, Any]      # full config dict for dataset_type see defaults for the entires
    source: str = "synthetic"  # "synthetic" or "external"
    data_path: Optional[str] = None # Path to .pt or .npy file if source="external"
    split_strategy: str = "none" # "none", "random_feature", "temporal", "spatial", "augment"
    split_params: Dict[str, Any] = field(default_factory=dict) # e.g. {"lag": 1, "axis": 2}

@dataclass
class ModelConfig:
    """
    Configuration for the high-level Information Bottleneck model (DSIB vs DVSIB).
    """
    model_type: str = "dsib"  # "dsib" or "dvsib"
    
    # Store model-specific params like beta here
    # e.g. {"beta": 256.0} for DVSIB
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticConfig:
    """
    Configuration for the critic / encoder architecture.
    `cfg` is expected to be the FULL critic config dict
    """
    critic_type: str         # e.g. "separable", "hybrid", "concat", ...
    cfg: Dict[str, Any]      # full config dict for critic_type see defaults for the entires
    encoder_type: str = "mlp" # "mlp", "resnet18", "cnn", "gru", "custom"
    share_encoder: bool = False # If True, use same encoder instance for X and Y
    variational: bool = False  # If True, use variational encoders
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra args like 'pretrained', 'layers'

@dataclass
class TrainingConfig:
    """
    Configuration for how the model is trained.
    `cfg` is expected to be the FULL training config dict
    """

    setup: str # infinite_data_iter or finite_data_epoch
    cfg: Dict[str, Any]   # e.g. {"batch_size": 128, "n_iter": 20000, ...


@dataclass
class ExperimentConfig:
    """
    Top-level configuration for a single experiment run.

    Contains:
    - dataset: fully-prepared DatasetConfig (defaults+overrides already merged)
    - critic: fully-prepared CriticConfig  (defaults+overrides already merged)
    - model: fully-prepared ModelConfig  (defaults+overrides already merged)
    - training: fully-prepared TrainingConfig  (defaults+overrides already merged)
    - outfile: HDF5 path where results will be written
    """
    dataset: DatasetConfig
    critic: CriticConfig
    model: ModelConfig
    training: TrainingConfig
    outfile: str = "h5_results/test.h5"
    seed: Optional[int] = None
    estimator: str = "lclip"
    # optional free-form description / extra tags
    description: Optional[str] = None
    extra_tags: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DatasetConfig",
    "CriticConfig",
    "TrainingConfig",
    "ExperimentConfig",
]
