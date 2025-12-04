from critic_config import make_critic
from models import *
from utils_training import *
from utils_data_generator import *
from utils import *
import math
import matplotlib.pyplot as plt
from config_defaults import CRITIC_DEFAULTS, DATASET_DEFAULTS
from h5_result_store import H5ResultStore
import torch
import numpy as np
import random
import copy
import itertools

from run_dsib_infinite_data import *


def run_hp_sweep_joint_gaussian(latent_dim: int = 8, embed_dim: int = 10):
    outfile = "h5_results/hp_sweep_joint_gaussian.h5"
    
    dataset_type = "joint_gaussian"
    ##override defaults
    cfg_user_dataset = dict(mi_bits=2.0, latent_dim=latent_dim)

    num_trials = 10
    embed_dim = embed_dim
    critic_type = "hybrid"

    # Hyperparameter grid
    pair_layers = [1]
    pair_sizes = [16]
    encode_layers = [2, 4]
    hidden_sizes = [128, 256, 512]
    lr_s = [1e-4, 1e-3, 5e-4]
    n_iter = 20_000

    # Build all hyperparameter combinations
    grid = list(itertools.product(
        lr_s,
        pair_layers,
        encode_layers,
        hidden_sizes,
        pair_sizes,
    ))

    print(f"Total configs: {len(grid)}")  # sanity check


    for trial_num in range(num_trials):
        print(f'Trial Number : {trial_num}')
        for lr, pair_layer, encode_layer, hidden_size, pair_size in grid:
            if critic_type == "hybrid" and embed_dim == 0:
                continue

            cfg_user_critic = dict(embed_dim=embed_dim, x_hidden_dim=hidden_size, x_layers=encode_layer, y_hidden_dim=hidden_size, y_layers=encode_layer, pair_hidden_dim=pair_size, pair_layers=pair_layer)
    
            mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter, lr=lr, show_progress= False)
    
            print(
                f"Run completed; saved to {outfile}; "
                f"trial={trial_num}"
                f"lr={lr}, cfg_user_critic={cfg_user_critic}, cfg_user_dataset={cfg_user_dataset}"
            )


if __name__=="__main__":
    run_hp_sweep_joint_gaussian(latent_dim = 8, embed_dim = 10)
    run_hp_sweep_joint_gaussian(latent_dim = 8, embed_dim = 8)
    run_hp_sweep_joint_gaussian(latent_dim = 8, embed_dim = 6)
    run_hp_sweep_joint_gaussian(latent_dim = 4, embed_dim = 4)
