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
import argparse

from run_dsib_infinite_data import *

## TO-DO: Make dataclasses to keep track of all parameters for an experiment, maybe called experimentConfig, basically group all the loose parameters and separate dictionaries into one


def run_parameter_sweep_swiss_roll():
    outfile = "h5_results/testing_swiss_roll.h5"
    
    dataset_type = "swiss_roll"

    num_trials = 10
    n_iter=20000
    kz_list = range(6)
    
    for trial_num in range(num_trials):  
        for sig_embed in [2.0, 5.0]:
            
            cfg_user_dataset = {"sig_embed":sig_embed} # override parameters
        
            for kz in kz_list:
                for critic_type in ["hybrid", "separable"]:
        
                    if critic_type == "hybrid" and kz == 0:
                        continue  
        
                    cfg_user_critic = {"embed_dim": kz}
        
                    print(f'Override parameters: {cfg_user_dataset}')
        
                    mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                    print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')
                    
def run_parameter_sweep_swiss_roll_2():
    outfile = "h5_results/testing_swiss_roll.h5"
    
    dataset_type = "swiss_roll"

    num_trials = 10
    n_iter=20000
    kz_list = range(6)
    
    for trial_num in range(num_trials):  
        for t_max  in [2.5, 3.5]:
            for sig_embed in [2.0, 5.0]:
                cfg_user_dataset = {"sig_embed":sig_embed, "t_min_pi_units":1.5, "t_max_pi_units":t_max} # override parameters
            
                for kz in kz_list:
                    for critic_type in ["hybrid", "separable"]:
            
                        if critic_type == "hybrid" and kz == 0:
                            continue  
            
                        cfg_user_critic = {"embed_dim": kz}
            
                        print(f'Override parameters: {cfg_user_dataset}')
            
                        mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                        print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')
                    

def run_parameter_sweep_hyperspherical_shell():
    outfile = "h5_results/testing_hyperspherical_shell.h5"
    
    dataset_type = "hyperspherical_shell"
    ##override defaults

    num_trials = 10
    n_iter=20000
    kz_list = range(6)
    
    for trial_num in range(num_trials):        
        for latent_dim in [1, 2, 3, 4, 6]:        
            for sig_embed in [2.0, 5.0]: 
                cfg_user_dataset = {"sig_embed":sig_embed, "latent_dim":latent_dim}
            
                for kz in kz_list:
                    for critic_type in ["hybrid", "separable"]:
        
                        if critic_type == "hybrid" and kz == 0:
                            continue  
        
                        cfg_user_critic = {"embed_dim": kz}
        
                        print(f'Override parameters: {cfg_user_dataset}')
            
                        mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                        print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')



def run_parameter_sweep_joint_gaussian():
    outfile = "h5_results/testing_joint_gaussian.h5"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 10
    kz_list = range(12)
    
    for trial_num in range(num_trials):
        for latent_dim in [2, 4, 8]:
            for critic_type in ["hybrid", "separable"]:
                for kz in kz_list:

                    cfg_user_dataset = dict(mi_bits=2.0, latent_dim=latent_dim)

                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    cfg_user_critic = {"embed_dim": kz}

                    print(cfg_user_dataset)
        
                    mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                    print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')



def run_parameter_sweep_gaussian_mixture():
    outfile = "h5_results/testing_gaussian_mixture.h5"
    
    dataset_type = "gaussian_mixture"
    ##override defaults
    cfg_user_dataset = dict(n_peaks=8, mu=2.0, sig=1.0, mi_bits_peak=2.0, sig_embed=0.0, noise_mode="white_relative", latent_dim=1, observe_dim_x=500, observe_dim_y=500)

    n_iter=20000
    num_trials = 10
    kz_list = range(15)
    
    for trial_num in range(num_trials):
        for kz in kz_list:
            for critic_type in ["hybrid", "separable"]:

                if critic_type == "hybrid" and kz == 0:
                    continue  

                cfg_user_critic = {"embed_dim": kz}
    
                mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True,
                        help="Which sweep to run")
    args = parser.parse_args()

    if args.job == "joint_gaussians":
        run_parameter_sweep_joint_gaussian()
    elif args.job == "gaussian_mixtures":
        run_parameter_sweep_gaussian_mixture()
    elif args.job == "swissroll":
        run_parameter_sweep_swiss_roll()
    elif args.job == "swissroll_2":
        run_parameter_sweep_swiss_roll_2()
    elif args.job == "hypershell":
        run_parameter_sweep_hyperspherical_shell()
    else:
        raise ValueError(f"Unknown job: {args.job}")

if __name__ == "__main__":
    main()
