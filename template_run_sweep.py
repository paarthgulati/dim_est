import os, datetime
import argparse
import itertools
from dim_est.run.run_dsib_single_experiment import run_dsib_infinite, run_dsib_finite
from dim_est.tests import quick_test

## To prevent writing from multiple heads to the same h5 file (which will crash the writer) -- make smaller outfiles with job ID and time which can be merged or later used together for plotting
## Default output filenames correspond to h5_results/ -- initialize that directory externally or change the output locations

def _default_output_dir():
    """
    Default directory for HDF5 result files:
    <directory_of_this_script>/h5_results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "h5_results")


def _make_outfile(output_dir: str, stem: str) -> str:
    """
    Create an outfile path of the form:
        <output_dir>/<stem>_<JOBID>_<TIMESTAMP>.h5
    and ensure the directory exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{stem}_{job_id}_{timestamp}.h5"
    return os.path.join(output_dir, fname)
    
# Sample functions for finite and infinite data regime using the joint gaussian latent distribution are included
def main():

    ## Quick test to make sure all imports work correctly
    quick_test.run_tests()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True,
                        help="Which sweep to run")
    parser.add_argument("--output-dir", type=str, default=None,
            help="Directory to store HDF5 results "
             "(default: <script_dir>/h5_results)")

    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir()

    if args.job == "infinite_joint_gaussian":
        infinite_joint_gaussian(output_dir=output_dir)
    elif args.job == "finite_joint_gaussian":
        finite_joint_gaussian(output_dir=output_dir)
    else:
        raise ValueError(f"Unknown job: {args.job}")



def finite_joint_gaussian(output_dir: str):
    outfile = _make_outfile(output_dir, stem="finite_data_joint_gaussian")

    setup ="finite_data_epoch"
    dataset_type = "joint_gaussian"
    num_trials = 2
    kz_list = range(12)
    latent_dim = 4
    critic_type = "hybrid"

    n_epoch = 100
    sig = 0.0
    
    for trial_num in range(num_trials):
        for n_samples in [128]: 
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



def infinite_joint_gaussian(output_dir: str):
    outfile = _make_outfile(output_dir, stem="infinite_data_joint_gaussian")

    setup ="infinite_data_iter"
    
    dataset_type = "joint_gaussian"

    n_iter=20_000
    num_trials = 1
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


if __name__ == "__main__":
    main()
