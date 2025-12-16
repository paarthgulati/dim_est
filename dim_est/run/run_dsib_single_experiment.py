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
from ..datasets.data_generation import make_data_generator
from ..utils.networks import teacher, Dataset
from ..config.critic_defaults import CRITIC_DEFAULTS
from ..config.dataset_defaults import DATASET_DEFAULTS 
from ..config.training_defaults import TRAINING_DEFAULTS 
from ..config.experiment_config import (
    DatasetConfig, CriticConfig, TrainingConfig, ExperimentConfig,
) 

from ..utils.h5_result_store import H5ResultStore
from ..utils.version_logs import get_git_commit_hash, is_dirty, get_git_diff


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
    device = 'cuda',
    save_trained_model_data_transform: bool = False,
):
    # 1. initialize seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1) # log this seed in the h5 output and other trianing parameters

    set_seed(seed)

    # 2. Build experiment config
    exp_cfg = make_experiment_config(setup = setup, dataset_type=dataset_type, critic_type=critic_type, dataset_overrides=dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides, estimator=estimator, seed=seed, outfile=outfile)

    dataset_cfg = exp_cfg.dataset.cfg
    critic_cfg = exp_cfg.critic.cfg
    training_cfg = exp_cfg.training.cfg

    # make sure we have the infinite data config set up:
    _validate_mode(exp_cfg, mode = "infinite_data_iter")

    # ## make device specification and use consistent across elements 
    # device = training_cfg.get("device", device) ## need to be synced up across training, model and datasets

    # 3. Data Generation -- INFINITE DATA, pass data generator through to training for fresh samples every iteration
    data_generator = make_data_generator(dataset_type, dataset_cfg, device = device)

    # 4. Build Network
    critic, *_ = make_critic(critic_type, critic_cfg) 
    model = DSIB(estimator=estimator, critic=critic)
    
    # 5. Training
    estimates_mi = train_model_infinite_data(model, data_generator, training_cfg, optimizer_cls=optimizer_cls, device=device)  # returns mi in nats
    mis_dsib_bits = np.array(estimates_mi)*np.log2(np.e)            

    # 6. Saving
    ## quick fields to help navigate the output instead of nested dictionaries. Modify build function to change tags fields; all the information about the run is saved under params

    tags = _build_run_tags(method = 'dsib', dataset_type = dataset_type, critic_type = critic_type, setup = setup, critic_cfg = critic_cfg, training_cfg = training_cfg, estimator = estimator )
    params = _build_run_params(exp_cfg)
    rid = save_run(outfile=outfile, tags=tags, params=params, mi_bits=mis_dsib_bits)

    if save_trained_model_data_transform:
        model.eval()
        model_path = _save_trained_model(model, outfile, rid, params, tags, transform = data_generator.transform)

    return mis_dsib_bits, exp_cfg

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
    device = 'cuda',
    save_trained_model_data_transform: bool = False,
):
    # 1. initialize seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1) # log this seed in the h5 output and other trianing parameters

    set_seed(seed)

    # 2. Build experiment config
    exp_cfg = make_experiment_config(setup = setup, dataset_type=dataset_type, critic_type=critic_type, dataset_overrides=dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides, estimator=estimator, seed=seed, outfile=outfile)

    dataset_cfg = exp_cfg.dataset.cfg
    critic_cfg = exp_cfg.critic.cfg
    training_cfg = exp_cfg.training.cfg

    _validate_mode(exp_cfg, "finite_data_epoch")

    # ## make device specification and use consistent across elements 
    # device = training_cfg.get("device", device) ## need to be synced up across training, model and datasets

    # 3. Data Generation --- FINITE DATA, create datasets to pass through to training
    data_generator = make_data_generator(dataset_type, dataset_cfg, device = device)
    trainSet_X,trainSet_Y = data_generator(training_cfg['n_samples'])  ## training dataset
    evalSet_X, evalSet_Y = trainSet_X[:min(training_cfg['batch_size'], training_cfg['n_samples'])], trainSet_Y[:min(training_cfg['batch_size'], training_cfg['n_samples'])] # eval set: fixed subset of train dataset to report the training mi
    testSet_X, testSet_Y = data_generator(training_cfg['batch_size']) # test dataset
    trainData_ = Dataset(trainSet_X, trainSet_Y)
    train_data_loader = torch.utils.data.DataLoader(trainData_, batch_size=training_cfg['batch_size'],shuffle=True)

    # 4. Build Network
    critic, *_ = make_critic(critic_type, critic_cfg) 
    model = DSIB(estimator=estimator, critic=critic)
    
    # 5. Training
    estimates_mi_train, estimates_mi_test = train_model_finite_data(model, train_data_loader, evalSet_X, evalSet_Y, testSet_X, testSet_Y,  training_cfg = training_cfg, optimizer_cls=optimizer_cls, device = device)  # returns mi_test and mi_train in nats

    mis_dsib_bits_train = np.array(estimates_mi_train)*np.log2(np.e)            
    mis_dsib_bits_test = np.array(estimates_mi_test)*np.log2(np.e)            

    # 6. Saving
    ## quick fields to help navigate the output instead of nested dictionaries. Modify build function to change tags fields; all the information about the run is saved under params
    tags = _build_run_tags(method = 'dsib', dataset_type = dataset_type, critic_type = critic_type, setup = setup, critic_cfg = critic_cfg, training_cfg = training_cfg, estimator = estimator )
    params = _build_run_params(exp_cfg)
    rid = save_run(outfile=outfile, tags=tags, params=params, mi_bits_train=mis_dsib_bits_train, mi_bits_test=mis_dsib_bits_test)

    if save_trained_model_data_transform:
        model.eval()
        model_path = _save_trained_model(model, outfile, rid, params, tags, transform = data_generator.transform)

    return [mis_dsib_bits_train, mis_dsib_bits_test], exp_cfg

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
    """
    Recursively merge `overrides` into `defaults`, validating keys.

    - Any key in `overrides` that does not exist in `defaults` raises KeyError.
    - If the default value is a dict, the override must also be a dict, and we recurse.
    - Returns a *new* merged dict, does not mutate `defaults`.
    """
    # Always work on a deep copy so we never mutate the defaults in-place
    merged = copy.deepcopy(defaults)

    for k, v in overrides.items():
        if k not in defaults:
            prefix = (error_prefix + ": ") if error_prefix else ""
            full_path = f"{_path}{k}"
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

    # ---- merge critic ----
    cr_defaults = copy.deepcopy(CRITIC_DEFAULTS[critic_type])
    cr_cfg = merge_with_validation(cr_defaults, critic_overrides, "critic overrides")

    # ---- merge training ----
    tr_defaults = copy.deepcopy(TRAINING_DEFAULTS[setup])
    tr_cfg = merge_with_validation(tr_defaults, training_overrides, "training overrides")

    # attach optimizer name --- not in the default training dict. To cahnge optimizer pass it through the run function
    tr_cfg["optimizer_cls_name"] = optimizer_cls.__name__

    # ---- assemble experiment config ----
    return ExperimentConfig(
        dataset=DatasetConfig(dataset_type=dataset_type, cfg=ds_cfg),
        critic=CriticConfig(critic_type=critic_type, cfg=cr_cfg),
        training=TrainingConfig(setup=setup, cfg=tr_cfg),
        outfile=outfile,
        seed=seed
    )
    
def save_run(outfile, tags, params, **arrays):
    """
    Small convenience wrapper around H5ResultStore.
    - outfile: path to the .h5 file
    - tags:    dict, used for querying/grouping
    - params:  dict, full config/meta (e.g. asdict(exp_cfg), code info, etc.)
    - arrays:  any named numpy arrays to save, e.g. mi_bits=..., loss=...
    """
    with H5ResultStore(outfile) as rs:
        rid = rs.new_run(params=params, tags=tags, dedupe_on_fingerprint=False)
        for name, arr in arrays.items():
            rs.save_array(rid, name, arr)
    
    print(f'Run completed; saved to {outfile}')
    return rid


def _build_code_metadata() -> dict:
    """Capture code-state metadata for reproducibility."""
    code_meta = {
        "git_commit": get_git_commit_hash(),
        "dirty": is_dirty(),
    }
    # if code_meta["dirty"]:
    #     code_meta["dirty_diff"] = get_git_diff()
    return code_meta


def _build_run_tags(dataset_type: str, critic_type: str, setup: str, critic_cfg: dict, training_cfg: dict, method: str, estimator: str) -> dict:
    """Lightweight, query-friendly tags for H5ResultStore."""
    # Decide what the "training length" means
    if setup == "infinite_data_iter":
        length_key = "n_iter"
        length_val = training_cfg.get("n_iter", None)
    elif setup == "finite_data_epoch":
        length_key = "n_epoch"
        length_val = training_cfg.get("n_epoch", None)
    else:
        raise ValueError(f"Unknown training setup: {setup}")

        
    return {
        "method": method, #dsib or whatever else
        "estimator": estimator,
        "dataset_type": dataset_type,
        "critic_type": critic_type,
        "setup": setup,
        "kz": critic_cfg["embed_dim"],
        length_key: length_val,
    }


def _build_run_params(exp_cfg) -> dict:
    """Heavier structured metadata; full experiment config + code state."""
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

@torch.no_grad()
def participation_ratio_dim(Z: torch.Tensor, center: bool = True) -> float:
    """
    Z: [N, d] encoder outputs, float32/float64, on any device.
    Returns: scalar effective (intrinsic) dimension estimate.
    """
    N, d = Z.shape

    # Optionally mean-center
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)

    # Covariance in feature space: [d, d]
    # (You can also use SVD on Z directly; effect is the same.)
    C = (Z.T @ Z) / (N - 1)  # [d, d]

    # Symmetric, so use eigvalsh for numerical stability
    evals = torch.linalg.eigvalsh(C)
    evals = torch.clamp(evals.real, min=0.0)  # remove tiny negative noise

    s1 = evals.sum()
    s2 = (evals ** 2).sum().clamp(min=1e-12)

    D_pr = (s1 ** 2 / s2).item()
    return D_pr