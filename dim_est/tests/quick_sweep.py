from ..run.run_dsib_infinite_data import run_single_experiment_dsib_infinite


def run_parameter_sweep(outfile = "h5_results/testing_code_version_history_2.h5"):
    
    dataset_type = "swiss_roll"

    num_trials = 1
    n_iter=2000
    kz_list = range(3)
    
    for trial_num in range(num_trials):  
        for sig_embed in [2.0, 5.0]:
            
            cfg_user_dataset = {"sig_embed":sig_embed, "t_min_pi_units":1.5, "t_max_pi_units":3.5} # override parameters
        
            for kz in kz_list:
                for critic_type in ["hybrid", "separable"]:
        
                    if critic_type == "hybrid" and kz == 0:
                        continue  
        
                    cfg_user_critic = {"embed_dim": kz}
        
                    print(f'Override parameters: {cfg_user_dataset}')
        
                    mis_dsib_bits = run_single_experiment_dsib_infinite(critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, cfg_user_dataset = cfg_user_dataset, cfg_user_critic=cfg_user_critic, n_iter=n_iter)
                    print(f'Run with kz = {kz}, critic_type: {critic_type};  dataset_type: {dataset_type}; Saved to {outfile}')
                    