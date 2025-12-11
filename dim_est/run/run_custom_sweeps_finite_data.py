import argparse
import itertools
from .run_dsib_single_experiment import run_dsib_infinite, run_dsib_finite


def finite_gaussian_mixture():
    outfile = "h5_results/finite_data_gaussian_mixture.h5"
    setup ="finite_data_epoch"

    dataset_type = "gaussian_mixture"
    n_peaks = 8 ## pass an int here
    mu = 2.0
    sig = 0.0

    dataset_overrides = dict(latent=dict(n_peaks=n_peaks, mi_bits_peak=2.0, mu=mu, sig=1.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))

    n_epoch=100
    num_trials = 10
    kz_list = range(15)
    critic_type = "hybrid"
        
    for trial_num in range(num_trials):
        for n_samples in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            for kz in kz_list:
                if critic_type == "hybrid" and kz == 0:
                    continue

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


    
def finite_gaussian_mixture_2():
    outfile = "h5_results/finite_data_gaussian_mixture.h5"
    setup ="finite_data_epoch"

    dataset_type = "gaussian_mixture"
    n_peaks = 8 ## pass an int here
    mu = 2.0
    sig = 0.0

    dataset_overrides = dict(latent=dict(n_peaks=n_peaks, mi_bits_peak=2.0, mu=mu, sig=1.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))

    n_epoch=1000
    num_trials = 10
    kz_list = range(15)
    critic_type = "hybrid"
        
    for trial_num in range(num_trials):
        for n_samples in [64, 1024, 65536]:
            for kz in kz_list:
                if critic_type == "hybrid" and kz == 0:
                    continue

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def finite_gaussian_mixture_3():
    outfile = "h5_results/finite_data_gaussian_mixture.h5"
    setup ="finite_data_epoch"

    dataset_type = "gaussian_mixture"
    n_peaks = 8 ## pass an int here
    mu = 2.0
    sig = 0.0

    dataset_overrides = dict(latent=dict(n_peaks=n_peaks, mi_bits_peak=2.0, mu=mu, sig=1.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))

    n_epoch=100
    num_trials = 10
    kz_list = range(15)
    critic_type = "separable"
        
    for trial_num in range(num_trials):
        for n_samples in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            for kz in kz_list:
                if critic_type == "hybrid" and kz == 0:
                    continue

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)

    

def finite_joint_gaussian():
    outfile = "h5_results/finite_data_joint_gaussian.h5"
    setup ="finite_data_epoch"
    
    dataset_type = "joint_gaussian"

    num_trials = 10
    kz_list = range(12)
    latent_dim = 4
    critic_type = "hybrid"

    n_epoch = 1000
    sig = 0.0
    
    for trial_num in range(num_trials):
        for n_samples in [64, 1024, 65536]: 
            for kz in kz_list:
                dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))


                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def finite_joint_gaussian_3():
    outfile = "h5_results/finite_data_joint_gaussian.h5"
    setup ="finite_data_epoch"
    
    dataset_type = "joint_gaussian"

    num_trials = 10
    kz_list = range(12)
    latent_dim = 4
    critic_type = "hybrid"

    n_epoch = 100
    sig = 0.0
    
    for trial_num in range(num_trials):
        for n_samples in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            for kz in kz_list:
                dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))


                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def finite_joint_gaussian_4():
    outfile = "h5_results/finite_data_joint_gaussian.h5"
    setup ="finite_data_epoch"
    
    dataset_type = "joint_gaussian"

    num_trials = 10
    kz_list = range(12)

    n_epoch = 100
    sig = 0.0
    
    for trial_num in range(num_trials):
        for n_samples in [64, 1024, 65536]:
            for latent_dim in [2, 3]:
                for critic_type in ["hybrid", "separable"]:
                    for kz in kz_list:
                        dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))


                        if critic_type == "hybrid" and kz == 0:
                            continue  

                        critic_overrides = {"embed_dim": kz}
                        training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                        print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                        print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                        print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                        mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)


def finite_joint_gaussian_2():
    outfile = "h5_results/finite_data_joint_gaussian.h5"
    setup ="finite_data_epoch"
    
    dataset_type = "joint_gaussian"

    num_trials = 10
    kz_list = range(12)
    latent_dim = 4
    critic_type = "separable"

    n_epoch = 100
    sig = 0.0
    
    for trial_num in range(num_trials):
        for n_samples in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            for kz in kz_list:
                dataset_overrides = dict(latent=dict(latent_dim=latent_dim, mi_bits=2.0), transform=dict(mode = 'teacher', sig_embed_x = sig, sig_embed_y = sig))


                if critic_type == "hybrid" and kz == 0:
                    continue  

                critic_overrides = {"embed_dim": kz}
                training_overrides = dict(n_epoch=n_epoch, n_samples = n_samples)

                print(f'Setup: {setup}, Training override parameters: {training_overrides}')
                print(f'Dataset Type: {dataset_type}; Dataset override parameters: {dataset_overrides}')
                print(f'Critic Type: {critic_type}; Critic override parameters: {critic_overrides}')

                mis_dsib_bits = run_dsib_finite(setup = setup, critic_type = critic_type, outfile=outfile, dataset_type=dataset_type, dataset_overrides = dataset_overrides, critic_overrides=critic_overrides, training_overrides=training_overrides)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True,
                        help="Which sweep to run")
    args = parser.parse_args()

    if args.job == "gaussian_mixture":
        finite_gaussian_mixture()
    elif args.job == "gaussian_mixture_2":
        finite_gaussian_mixture_2()
    elif args.job == "gaussian_mixture_3":
        finite_gaussian_mixture_3()
    elif args.job == "joint_gaussian":
        finite_joint_gaussian()
    elif args.job == "joint_gaussian_2":
        finite_joint_gaussian_2()
    elif args.job == "joint_gaussian_3":
        finite_joint_gaussian_3()
    elif args.job == "joint_gaussian_4":
        finite_joint_gaussian_4()
    else:
        raise ValueError(f"Unknown job: {args.job}")

if __name__ == "__main__":
    main()
