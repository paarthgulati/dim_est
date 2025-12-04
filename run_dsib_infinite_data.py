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

## TO-DO: Add argparse to do this systematically when calling the run file from a job
## TO-DO: Make dataclasses to keep track of all parameters for an experiment, maybe called experimentConfig, basically group all the loose parameters and separate dictionaries into one


def create_teacher_models_symmetric(input_dim: int, output_dim: int, device='cuda'):
    teacher_model_x = teacher(dz=input_dim, output_dim=output_dim)
    teacher_model_y = teacher(dz=input_dim, output_dim=output_dim)

    teacher_model_x = teacher_model_x.to(device)
    teacher_model_y = teacher_model_y.to(device)

    for param_x in teacher_model_x.parameters():
        param_x.requires_grad_(False)  # Freeze
    for param_y in teacher_model_y.parameters():
        param_y.requires_grad_(False)

    return teacher_model_x, teacher_model_y



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def run_parameter_sweep():
    outfile = "h5_results/testing_a.h5"
    
    dataset_type = "gaussian_mixture"
    ##override defaults
    cfg_user_dataset = dict(n_peaks=8, mu=2.0, sig=1.0, mi_bits_peak=2.0, sig_embed=0.0, noise_mode="white_relative", latent_dim=1, observe_dim_x=500, observe_dim_y=500)

    n_iter=2000
    num_trials = 1
    kz_list = range(20)
    
    for trial_num in range(num_trials):
        for kz in kz_list:
            for critic_type in ["hybrid", "separable"]:

                if critic_type == "hybrid" and kz == 0:
                    continue  

                cfg_user_critic = {"embed_dim": kz}
    
                mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')



def run_single_experiment_dsib_infinite(
    estimator: str = "lclip",
    critic_type = "hybrid",
    batch_size=128,
    n_iter=2000,
    dataset_type = "gaussian_mixture",
    seed = None,
    outfile = "h5_results/test_output.h5",
    optimizer_cls=torch.optim.Adam, 
    lr=5e-4, 
    optimizer_kwargs=None, 
    cfg_user_dataset=None,  #override options 
    cfg_user_critic= None,  #override options
    show_progress = True, 
    device = 'cuda'
):
    ## initialize seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1) # log this seed in the h5 output and other trianing parameters

    set_seed(seed)

    # create dataset and critic configurations: override defaults with user inputs
    dataset_cfg = copy.deepcopy(DATASET_DEFAULTS[dataset_type])
    if cfg_user_dataset:
        dataset_cfg.update(cfg_user_dataset)

    critic_cfg = copy.deepcopy(CRITIC_DEFAULTS[critic_type])
    if cfg_user_critic:
        critic_cfg.update(cfg_user_critic)

    dataset_latent_dim = dataset_cfg["latent_dim"]
    if dataset_cfg["observe_dim_x"] == dataset_cfg["observe_dim_y"]:
        dataset_observe_dim = dataset_cfg["observe_dim_x"]
    else:
        raise ValueError(f"Output dim_x != Ouput dim_y, currently set up for a symmetric case only")    
    
    teacher_model_x, teacher_model_y = create_teacher_models_symmetric(input_dim = dataset_latent_dim, output_dim = dataset_observe_dim, device=device)

    ##################################    

    critic, critic_params, critic_tags = make_critic(critic_type, **critic_cfg) 
    data_generator = make_data_generator(dataset_type, dataset_cfg, teacher_model_x, teacher_model_y)

    model = DSIB(estimator=estimator, critic=critic)
    
    ################TRAINING###########################
    
    estimates_mi = train_model_infinite_data(model, data_generator, batch_size, n_iter, show_progress=show_progress, optimizer_cls=optimizer_cls, lr=lr, optimizer_kwargs=optimizer_kwargs, device=device)  
    # returns -mi in nats
    mis_dsib_bits = -np.array(estimates_mi)*np.log2(np.e)            

    ##############SAVE_OUTPUTS#############################

    tags={
        "method": "dsib",
        "critic_type": critic_type,
        "dataset_type": dataset_type,
        "estimator": estimator,

        "kz": critic_cfg.get("embed_dim", None),
        "batch_size": batch_size,
        "n_iter": n_iter,           
    }

    params={
        "method": "dsib",
        "critic_type": critic_type,
        "critic_cfg": critic_cfg,
        "critic_params": critic_params,

        "dataset_type": dataset_type,
        "dataset_cfg": dataset_cfg,
        "dataset_embed":"teacher",

        "training_cfg": {
            "setup": "infinite_data_iter",
            "estimator": estimator, 
            "batch_size": batch_size,
            "n_iter": n_iter,
            "optimizer": optimizer_cls.__name__,
            "lr": lr,
            "seed": seed,
        },
    }

    with H5ResultStore(outfile) as rs:        
        rid = rs.new_run(params=params, tags=tags, dedupe_on_fingerprint=False)
        rs.save_array(rid, "mi_bits", mis_dsib_bits)

    return mis_dsib_bits


if __name__=="__main__":
    run_parameter_sweep()

