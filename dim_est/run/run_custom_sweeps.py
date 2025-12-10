import argparse
import itertools
from .run_dsib_single_experiment import run_dsib_infinite, run_dsib_finite



                   
def infinite_swiss_roll():
    outfile = "h5_results/infinite_data_swiss_roll.h5"
    setup ="infinite_data_iter"

    dataset_type = "swiss_roll"

    num_trials = 10
    n_iter=20000
    kz_list = range(7)
    
    for trial_num in range(num_trials):  
        for t_max  in [2.5, 3.5, 4.5]:
            for sig_embed in [2.0, 5.0]:
                dataset_overrides = dict(latent=dict(t_min_pi_units=1.5, t_max_pi_units=t_max), transform = dict(sig_embed_x=sig_embed, sig_embed_y=sig_embed))
            
                for kz in kz_list:
                    for critic_type in ["hybrid", "separable"]:
            
                        if critic_type == "hybrid" and kz == 0:
                            continue  
            
                        critic_overrides = {"embed_dim": kz}
            
            
                        training_overrides = dict(n_iter=n_iter)

                        print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                        print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                        print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                        mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)

def infinite_hyperspherical_shell():
    outfile = "h5_results/infinite_data_hyperspherical_shell.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "hyperspherical_shell"

    num_trials = 10
    n_iter=20000
    kz_list = range(10)
    
    for trial_num in range(num_trials):        
        for latent_dim in [1, 2, 3, 4, 6]:        
            for sig_embed in [2.0, 5.0]: 
                dataset_overrides = dict(latent=dict(latent_dim=latent_dim), transform = dict(sig_embed_x=sig_embed, sig_embed_y=sig_embed))
                
                for kz in kz_list:
                    for critic_type in ["hybrid", "separable"]:
        
                        if critic_type == "hybrid" and kz == 0:
                            continue  
        
                        critic_overrides = {"embed_dim": kz}
                        training_overrides = dict(n_iter=n_iter)
        
                        print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                        print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                        print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                        mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_hyperspherical_shell_2():
    outfile = "h5_results/infinite_data_hyperspherical_shell.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "hyperspherical_shell"

    num_trials = 10
    n_iter=20000
    kz_list = range(10)
    radial_std=0.5
    
    for trial_num in range(num_trials):        
        for latent_dim in [1, 2, 3, 4, 6]:        
            for sig_embed in [2.0, 5.0]: 
                dataset_overrides = dict(latent=dict(latent_dim=latent_dim, radial_std=radial_std), transform = dict(sig_embed_x=sig_embed, sig_embed_y=sig_embed))
                
                for kz in kz_list:
                    for critic_type in ["hybrid", "separable"]:
        
                        if critic_type == "hybrid" and kz == 0:
                            continue  
        
                        critic_overrides = {"embed_dim": kz}
                        training_overrides = dict(n_iter=n_iter)
        
                        print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                        print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                        print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                        mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_joint_gaussian():
    outfile = "h5_results/infinite_data_joint_gaussian.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 10
    kz_list = range(12)
    
    for trial_num in range(num_trials):
        for latent_dim in [2, 4, 8]:
            for critic_type in ["hybrid", "separable"]:
                for kz in kz_list:
                    dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0))


                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    critic_overrides = {"embed_dim": kz}
                    training_overrides = dict(n_iter=n_iter)

                    print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                    print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                    print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                    mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)

def infinite_joint_gaussian_linear():
    outfile = "h5_results/infinite_data_joint_gaussian_linear_transform.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 10
    kz_list = range(12)
    
    for trial_num in range(num_trials):
        for latent_dim in [4]:
            for critic_type in ["hybrid", "separable", "separable_augmented"]:
                for kz in kz_list:
                    dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode='linear'))


                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    critic_overrides = {"embed_dim": kz}
                    training_overrides = dict(n_iter=n_iter)

                    print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                    print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                    print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                    mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_joint_gaussian_teacher():
    outfile = "h5_results/infinite_data_joint_gaussian_teacher_transform.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 10
    kz_list = range(12)
    
    for trial_num in range(num_trials):
        for latent_dim in [4]:
            for critic_type in ["hybrid", "separable", "separable_augmented"]:
                for kz in kz_list:
                    dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode='teacher'))


                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    critic_overrides = {"embed_dim": kz}
                    training_overrides = dict(n_iter=n_iter)

                    print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                    print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                    print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                    mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_joint_gaussian_overkill():
    outfile = "h5_results/infinite_data_joint_gaussian.h5"
    setup ="infinite_data_iter"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 10
    kz_list = range(12)
    encode_layer = 2
    hidden_size = 512
    lr = 1e-3
    pair_size=128
    pair_layer=1

    for trial_num in range(num_trials):
        for latent_dim in [8]:
            for critic_type in ["hybrid", "separable"]:
                for kz in kz_list:
                    dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0))


                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    critic_overrides = dict(embed_dim=kz, x_hidden_dim=hidden_size, x_layers=encode_layer, y_hidden_dim=hidden_size, y_layers=encode_layer, pair_hidden_dim=pair_size, pair_layers=pair_layer)
                    training_overrides = dict(lr=lr, n_iter=n_iter)

                    print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                    print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                    print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                    mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_gaussian_mixture():
    outfile = "h5_results/infinite_data_gaussian_mixture.h5"
    setup ="infinite_data_iter"

    dataset_type = "gaussian_mixture"
    dataset_overrides = dict(latent=dict(n_peaks=8, mi_bits_peak=2.0, mu=2.0, sig=1.0))

    n_iter=20000
    num_trials = 10
    kz_list = range(15)
    
    for trial_num in range(num_trials):
        for kz in kz_list:
            for critic_type in ["hybrid", "separable"]:

                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_iter=n_iter)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_gaussian_mixture_linear():
    outfile = "h5_results/infinite_data_gaussian_mixture_linear_transform.h5"
    setup ="infinite_data_iter"

    dataset_type = "gaussian_mixture"
    dataset_overrides = dict(latent=dict(n_peaks=8, mi_bits_peak=2.0, mu=2.0, sig=1.0), transform=dict(mode = 'linear'))

    n_iter=20000
    num_trials = 10
    kz_list = range(15)
    
    for trial_num in range(num_trials):
        for kz in kz_list:
            for critic_type in ["hybrid", "separable", "separable_augmented"]:

                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_iter=n_iter)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def infinite_gaussian_mixture_teacher():
    outfile = "h5_results/infinite_data_gaussian_mixture_teacher_transform.h5"
    setup ="infinite_data_iter"

    dataset_type = "gaussian_mixture"
    dataset_overrides = dict(latent=dict(n_peaks=8, mi_bits_peak=2.0, mu=2.0, sig=1.0), transform=dict(mode = 'teacher'))

    n_iter=20000
    num_trials = 10
    kz_list = range(15)
    
    for trial_num in range(num_trials):
        for kz in kz_list:
            for critic_type in ["hybrid", "separable", "separable_augmented"]:

                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_iter=n_iter)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)




def infinite_gaussian_mixture_2():
    outfile = "h5_results/infinite_data_gaussian_mixture.h5"
    setup ="infinite_data_iter"

    dataset_type = "gaussian_mixture"

    for n_peaks, mu in zip([16], [2.0]):
        dataset_overrides = dict(latent=dict(n_peaks=n_peaks, mi_bits_peak=2.0, mu=mu, sig=1.0))

        n_iter=20000
        num_trials = 10
        kz_list = range(25)
        
        for trial_num in range(num_trials):
            for kz in kz_list:
                for critic_type in ["hybrid", "separable"]:

                    if critic_type == "hybrid" and kz == 0:
                        continue  

                    critic_overrides = {"embed_dim": kz}
                    training_overrides = dict(n_iter=n_iter)

                    print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                    print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                    print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                    mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def hp_sweep_hybrid_joint_gaussian(latent_dim: int = 8, embed_dim: int = 8):
    outfile = "h5_results/hp_sweep_infinite_joint_gaussian.h5"
    setup ="infinite_data_iter"

    dataset_type = "joint_gaussian"
    dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0))

    num_trials = 3
    embed_dim = embed_dim
    critic_type = "hybrid"

    # Hyperparameter grid
    pair_layers = [1,2]
    pair_sizes = [16, 64]
    encode_layers = [2, 4]
    hidden_sizes = [256, 512]
    lr_s = [1e-4, 1e-5]
    n_iters = [20_000, 50_000]

    # Build all hyperparameter combinations
    grid = list(itertools.product(
        lr_s,
        pair_layers,
        encode_layers,
        hidden_sizes,
        pair_sizes,
        n_iters
    ))

    print(f"Total configs: {len(grid)}")  # sanity check


    for trial_num in range(num_trials):
        print(f'Trial Number : {trial_num}')
        for n_iter in n_iters:
            for lr, pair_layer, encode_layer, hidden_size, pair_size, n_iter in grid:
                if critic_type == "hybrid" and embed_dim == 0:
                    continue

                critic_overrides = dict(embed_dim=embed_dim, x_hidden_dim=hidden_size, x_layers=encode_layer, y_hidden_dim=hidden_size, y_layers=encode_layer, pair_hidden_dim=pair_size, pair_layers=pair_layer)
                training_overrides = dict(lr=lr, n_iter=n_iter)


                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_infinite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True,
                        help="Which sweep to run")
    args = parser.parse_args()

    if args.job == "joint_gaussians":
        infinite_joint_gaussian()
    elif args.job == "joint_gaussians_linear":
        infinite_joint_gaussian_linear()
    elif args.job == "joint_gaussians_teacher":
        infinite_joint_gaussian_teacher()
    elif args.job == "joint_gaussians_overkill":
        infinite_joint_gaussian_overkill()
    elif args.job == "gaussian_mixtures":
        infinite_gaussian_mixture()
    elif args.job == "gaussian_mixtures_linear":
        infinite_gaussian_mixture_linear()
    elif args.job == "gaussian_mixtures_teacher":
        infinite_gaussian_mixture_teacher()
    elif args.job == "gaussian_mixtures_2":
        infinite_gaussian_mixture_2()
    elif args.job == "swissroll":
        infinite_swiss_roll()
    elif args.job == "hypershell":
        infinite_hyperspherical_shell()
    elif args.job == "hypershell_2":
        infinite_hyperspherical_shell_2()
    elif args.job == "hp_optim":
        hp_sweep_hybrid_joint_gaussian()
    else:
        raise ValueError(f"Unknown job: {args.job}")

if __name__ == "__main__":
    main()
