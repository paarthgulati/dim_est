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


@dataclass
class CriticConfig:
    """
    Configuration for the critic / encoder architecture.
    `cfg` is expected to be the FULL critic config dict
    """
    critic_type: str         # e.g. "separable", "hybrid", "concat", ...
    cfg: Dict[str, Any]      # full config dict for critic_type see defaults for the entires


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
    - training: fully-prepared TrainingConfig  (defaults+overrides already merged)
    - outfile: HDF5 path where results will be written
    """
    dataset: DatasetConfig
    critic: CriticConfig
    training: TrainingConfig
    outfile: str
    seed: Optional[int] = None
    # optional free-form description / extra tags
    description: Optional[str] = None
    extra_tags: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DatasetConfig",
    "CriticConfig",
    "TrainingConfig",
    "ExperimentConfig",
]
