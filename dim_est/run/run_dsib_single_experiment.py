from xml.parsers.expat import model
import torch
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import Any, Dict, Optional, Mapping
from pathlib import Path

from ..models.critic_builders import make_critic
from ..models.models import DSIB
from ..training import train_model_infinite_data, train_model_finite_data
from ..datasets.data_generation import make_data_generator, get_finite_dataloaders
from ..utils.networks import teacher, Dataset
from ..config.critic_defaults import CRITIC_DEFAULTS
from ..config.dataset_defaults import DATASET_DEFAULTS 
from ..config.training_defaults import TRAINING_DEFAULTS 
from ..config.experiment_config import (
    DatasetConfig, CriticConfig, TrainingConfig, ExperimentConfig,
) 

from ..utils.h5_result_store import H5ResultStore
from ..utils.version_logs import get_git_commit_hash, is_dirty, get_git_diff

def _get_auto_device(device):
    """Helper to auto-detect device if None is passed."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def _infer_data_dimensions(dataset_cfg: dict) -> tuple:
    """
    Inspects dataset config to determine output shapes Nx and Ny.
    This bridges the gap between 'None' in identity transform and explicit ints needed for Critics.
    """
    latent_dim = dataset_cfg["latent"]["latent_dim"]
    trans_cfg = dataset_cfg.get("transform", {})
    
    # Check if external data defines dimensions
    # If source is external, we might not know dims until load. 
    # But if it's synthetic, we know.
    
    # Logic: Use observe_dim if present and not None, else fallback to latent_dim
    if trans_cfg.get("observe_dim_x") is not None:
        Nx = trans_cfg["observe_dim_x"]
    else:
        Nx = latent_dim

    if trans_cfg.get("observe_dim_y") is not None:
        Ny = trans_cfg["observe_dim_y"]
    else:
        Ny = latent_dim
        
    return Nx, Ny

def run_dsib_infinite(
    dataset_type: str = "gaussian_mixture",
    critic_type: str = "hybrid",
    setup: str ="infinite_data_iter",
    outfile: str = "h5_results/test_output.h5",
    dataset_overrides=None,  #override options 
    critic_overrides= None,  #override options
    training_overrides= None,  #override options
    seed: Optional[int] = None,
    estimator = "lclip",
    optimizer_cls=torch.optim.Adam, 
    device: Optional[str] = None, 
    save_trained_model_data_transform: bool = False,
):
    # 0. Auto-detect device
    device = _get_auto_device(device)
    print(f"Using device: {device}")

    # 1. initialize seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1) 

    set_seed(seed)

    # 2. Build experiment config
    exp_cfg = make_experiment_config(setup = setup, dataset_type=dataset_type, critic_type=critic_type, dataset_overrides=dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides, estimator=estimator, seed=seed, outfile=outfile)

    dataset_cfg = exp_cfg.dataset.cfg
    critic_cfg = exp_cfg.critic.cfg
    training_cfg = exp_cfg.training.cfg

    _validate_mode(exp_cfg, mode = "infinite_data_iter")

    # 3. Data Generation
    # Automatically set Nx/Ny in critic based on dataset config
    auto_nx, auto_ny = _infer_data_dimensions(dataset_cfg)
    # We update the cfg dict directly. 
    # Note: If user explicitly provided Nx in overrides, it's already in critic_cfg.
    # However, to be safe, we usually trust the DATA logic for input sizes over defaults.
    # If we want to allow manual override, we should check if it was in defaults or overrides.
    # For now, forcing consistency with data is usually the desired behavior.
    critic_cfg["Nx"] = auto_nx
    critic_cfg["Ny"] = auto_ny   
    data_generator = make_data_generator(dataset_type, dataset_cfg, device = device)
    
    # 4. Build Network
    critic, *_ = make_critic(critic_type, critic_cfg) 
    model = DSIB(estimator=estimator, critic=critic)
    
    # 5. Training
    estimates_mi,  trace_cov_results = train_model_infinite_data(model, data_generator, training_cfg, optimizer_cls=optimizer_cls, device=device)
    mis_dsib_bits = np.array(estimates_mi)*np.log2(np.e)            

    # 6. Saving
    ## quick fields to help navigate the output instead of nested dictionaries. Modify build function to change tags fields; all the information about the run is saved under params

    tags = _build_run_tags(method = 'dsib', dataset_type = dataset_type, critic_type = critic_type, setup = setup, critic_cfg = critic_cfg, training_cfg = training_cfg, estimator = estimator )
    params = _build_run_params(exp_cfg)
    rid = save_run(outfile=outfile, tags=tags, params=params, mi_bits=mis_dsib_bits)

    if save_trained_model_data_transform:
        model.eval()
        model_path = _save_trained_model(model, outfile, rid, params, tags, transform = data_generator.transform)

    return mis_dsib_bits, trace_cov_results, exp_cfg

def run_dsib_finite(
    dataset_type: str = "gaussian_mixture",
    critic_type: str = "hybrid",
    setup: str ="finite_data_epoch",
    outfile: str = "h5_results/test_output.h5",
    dataset_overrides=None,  #override options 
    critic_overrides= None,  #override options
    training_overrides= None,  #override options
    seed: Optional[int] = None,
    estimator = "lclip",
    optimizer_cls=torch.optim.Adam, 
    device: Optional[str] = None,
    save_trained_model_data_transform: bool = False,
):
    # 0. Auto-detect device
    device = _get_auto_device(device)
    print(f"Using device: {device}")

    # 1. initialize seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)

    set_seed(seed)

    # 2. Build experiment config
    exp_cfg = make_experiment_config(setup = setup, dataset_type=dataset_type, critic_type=critic_type, dataset_overrides=dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides, estimator=estimator, seed=seed, outfile=outfile)

    # Note: dataset_cfg here comes from merger, which now includes the dynamic keys
    dataset_cfg = exp_cfg.dataset.cfg
    critic_cfg = exp_cfg.critic.cfg
    training_cfg = exp_cfg.training.cfg

    _validate_mode(exp_cfg, "finite_data_epoch")

# 3. Data Generation
    train_loader, test_loader, train_subset_loader = get_finite_dataloaders(
        dataset_type=dataset_type, 
        dataset_cfg=dataset_cfg, 
        training_cfg=training_cfg, 
        device=device,
        train_dataset_override=dataset_overrides.get('train_dataset') if dataset_overrides else None,
        test_dataset_override=dataset_overrides.get('test_dataset') if dataset_overrides else None
    )
    try:
        # train_loader.dataset is likely a Subset or TensorDataset
        # We can just peek at the first batch from the loader
        sample_x, sample_y = next(iter(train_loader))
        # sample_x shape is [Batch, Dim] or [Batch, C, H, W]
        # We want the feature dim (everything after Batch)
        # If flat: [Batch, D] -> D
        # If image: [Batch, C, H, W] -> We assume Encoder handles this based on encoder_type?
        # Critic Nx expects an INT usually. 
        # If using MLP, Nx should be D. 
        # If using CNN, Nx can be ignored or treated as channels. 
        
        if dataset_cfg.get("source") == "external":
             # For external, we MUST rely on data shape
             if sample_x.dim() > 1:
                 critic_cfg["Nx"] = int(np.prod(sample_x.shape[1:]))
                 critic_cfg["Ny"] = int(np.prod(sample_y.shape[1:]))
        else:
            # Synthetic: Use the config inference
            auto_nx, auto_ny = _infer_data_dimensions(dataset_cfg)
            critic_cfg["Nx"] = auto_nx
            critic_cfg["Ny"] = auto_ny
            data_generator = make_data_generator(dataset_type, dataset_cfg, device = device)

    except Exception as e:
        print(f"Warning: Could not auto-infer input dimensions from data: {e}")
        # Fallback to defaults or user overrides
        pass
    
    # 4. Build Network
    critic, *_ = make_critic(critic_type, critic_cfg) 
    model = DSIB(estimator=estimator, critic=critic)
    
    # 5. Training
    mi_train_trace, mi_test_trace, final_train_mi, final_test_mi, final_pr_train, final_pr_test, trace_cov_results = train_model_finite_data(
        model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        train_subset_loader=train_subset_loader, # Pass the subset loader
        training_cfg=training_cfg, 
        optimizer_cls=optimizer_cls, 
        device=device
    )

    # Save the traces
    mis_dsib_bits_train = np.array(mi_train_trace) * np.log2(np.e)            
    mis_dsib_bits_test = np.array(mi_test_trace) * np.log2(np.e)
    final_results = {
        "final_train_mi_bits": np.array([final_train_mi * np.log2(np.e)]),
        "final_test_mi_bits":  np.array([final_test_mi * np.log2(np.e)])
    }

    # 6. Saving
    tags = _build_run_tags(method = 'dsib', dataset_type = dataset_type, critic_type = critic_type, setup = setup, critic_cfg = critic_cfg, training_cfg = training_cfg, estimator = estimator )
    params = _build_run_params(exp_cfg)
    
    rid = save_run(
        outfile=outfile, 
        tags=tags, 
        params=params, 
        mi_bits_train=mis_dsib_bits_train, 
        mi_bits_test=mis_dsib_bits_test,
        **final_results
    )


    if save_trained_model_data_transform:
        model.eval()
        model_path = _save_trained_model(model, outfile, rid, params, tags, transform = data_generator.transform)

    return [mis_dsib_bits_train, mis_dsib_bits_test], [np.array([final_train_mi * np.log2(np.e)]), np.array([final_test_mi * np.log2(np.e)])], [final_pr_train, final_pr_test], trace_cov_results, exp_cfg

def _save_trained_model(model, outfile, rid, params, tags, transform = None):
    # strip ".h5" and make the correct outdir from the outfile location:
    out = Path(outfile)
    model_dir = out.parent / f"{out.stem}.models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # append rid for model path
    model_path = model_dir / f"{rid}.pt"

    payload = {
        "state_dict": model.state_dict(),
        "params": params,     
        "rid": rid,
        "pytorch_version": torch.__version__,
    }

    # Save transform state if it is an nn.Module (TeacherTransform / LinearTransform / IdentityTransform)
    if transform is not None and isinstance(transform, torch.nn.Module):
        payload["transform_state_dict"] = transform.state_dict()  # {} for identity, weights for linear/teacher
        payload["transform_class"] = transform.__class__.__name__
    else:
        payload["transform_state_dict"] = None
        payload["transform_class"] = None

    torch.save(payload, model_path)
    print(f"Saved trained model checkpoint to {model_path}")

    return model_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def merge_with_validation(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
    error_prefix: str = "",
    _path: str = "",
) -> dict:
    # Always work on a deep copy so we never mutate the defaults in-place
    merged = copy.deepcopy(defaults)

    for k, v in overrides.items():
        if k not in defaults:
            prefix = (error_prefix + ": ") if error_prefix else ""
            full_path = f"{_path}{k}"
            # Allow Phase 1 & 3 keys (source, data_path, split_strategy, split_params)
            # These are dynamic keys that won't be in the defaults dictionary
            if k in ["source", "data_path", "split_strategy", "split_params"]:
                merged[k] = v
                continue
            raise KeyError(
                f"{prefix}Invalid override key '{full_path}'. "
                f"Allowed keys at this level: {list(defaults.keys())}"
            )

        default_val = defaults[k]

        # If the default is a dict, we expect a dict and recurse
        if isinstance(default_val, dict):
            if not isinstance(v, Mapping):
                prefix = (error_prefix + ": ") if error_prefix else ""
                full_path = f"{_path}{k}"
                raise TypeError(
                    f"{prefix}Override for '{full_path}' must be a mapping, "
                    f"got {type(v).__name__}"
                )

            merged[k] = merge_with_validation(
                default_val, v, error_prefix=error_prefix, _path=f"{_path}{k}."
            )

        else:
            # Leaf value: just override
            merged[k] = v

    return merged


def make_experiment_config(
    *,
    dataset_type: str = "joint_gaussian",
    critic_type: str = "hybrid",
    setup: str = "infinite_data_iter",
    outfile: str,
    estimator: str,
    seed=int, 
    dataset_overrides=None,
    critic_overrides=None,
    training_overrides=None,
    optimizer_cls=torch.optim.Adam,
) -> ExperimentConfig:

    dataset_overrides = dataset_overrides or {}
    critic_overrides = critic_overrides or {}
    training_overrides = training_overrides or {}

    # ---- merge dataset ----
    ds_defaults = copy.deepcopy(DATASET_DEFAULTS[dataset_type])
    ds_cfg = merge_with_validation(ds_defaults, dataset_overrides, "dataset overrides")
    
    # Phase 1 & 3: Handle dynamic keys extraction for Dataclass
    source = dataset_overrides.get("source", "synthetic")
    data_path = dataset_overrides.get("data_path", None)
    split_strategy = dataset_overrides.get("split_strategy", "none")
    split_params = dataset_overrides.get("split_params", {})

    # ---- merge critic ----
    cr_defaults = copy.deepcopy(CRITIC_DEFAULTS[critic_type])
    cr_cfg = merge_with_validation(cr_defaults, critic_overrides, "critic overrides")

    # ---- merge training ----
    tr_defaults = copy.deepcopy(TRAINING_DEFAULTS[setup])
    tr_cfg = merge_with_validation(tr_defaults, training_overrides, "training overrides")

    tr_cfg["optimizer_cls_name"] = optimizer_cls.__name__

    return ExperimentConfig(
        dataset=DatasetConfig(
            dataset_type=dataset_type, 
            cfg=ds_cfg, 
            source=source, 
            data_path=data_path,
            split_strategy=split_strategy,
            split_params=split_params
        ),
        critic=CriticConfig(critic_type=critic_type, cfg=cr_cfg),
        training=TrainingConfig(setup=setup, cfg=tr_cfg),
        outfile=outfile,
        seed=seed
    )
    
def save_run(outfile, tags, params, **arrays):
    with H5ResultStore(outfile) as rs:
        rid = rs.new_run(params=params, tags=tags, dedupe_on_fingerprint=False)
        for name, arr in arrays.items():
            rs.save_array(rid, name, arr)
    
    print(f'Run completed; saved to {outfile}')
    return rid


def _build_code_metadata() -> dict:
    code_meta = {
        "git_commit": get_git_commit_hash(),
        "dirty": is_dirty(),
    }
    return code_meta


def _build_run_tags(dataset_type: str, critic_type: str, setup: str, critic_cfg: dict, training_cfg: dict, method: str, estimator: str) -> dict:
    if setup == "infinite_data_iter":
        length_key = "n_iter"
        length_val = training_cfg.get("n_iter", None)
    elif setup == "finite_data_epoch":
        length_key = "n_epoch"
        length_val = training_cfg.get("n_epoch", None)
    else:
        raise ValueError(f"Unknown training setup: {setup}")

        
    return {
        "method": method, 
        "estimator": estimator,
        "dataset_type": dataset_type,
        "critic_type": critic_type,
        "setup": setup,
        "kz": critic_cfg["embed_dim"],
        length_key: length_val,
    }


def _build_run_params(exp_cfg) -> dict:
    return {
        "experiment_cfg": asdict(exp_cfg),
        "code": _build_code_metadata(),
    }

def _validate_mode(exp_cfg, mode: str = "infinite_data_iter"):
    if exp_cfg.training.setup != mode:
        raise ValueError(
            f"mode expected {mode}, "
            f"got {exp_cfg.training.setup!r}"
        )
