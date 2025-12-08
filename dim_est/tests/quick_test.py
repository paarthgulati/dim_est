from dim_est.run.run_dsib_single_experiment import run_dsib_infinite, run_dsib_finite


def single_finite_run(outfile = "h5_results/test_run.h5"):
    mi_trial, exp_cfg = run_dsib_finite(outfile=outfile)
    print(f"Experimental config: {exp_cfg}")
    return exp_cfg

def single_infinite_run(outfile = "h5_results/test_run.h5"):
    mis, exp_cfg = run_dsib_infinite(outfile=outfile)
    print(f"Experimental config: {exp_cfg}")

    return exp_cfg

def parameter_sweep(outfile = "h5_results/test_run.h5"):

    for embed_dim in range(1, 3, 1):
        mis, exp_cfg = run_dsib_infinite(outfile=outfile, critic_overrides={'embed_dim':embed_dim})
        print(f"Experimental config: {exp_cfg}")


    return exp_cfg

def main():
    exp_cfg = single_infinite_run()
    print(f"\n INFINITE DATA TRIAL PASSED... \n")

    exp_cfg = single_finite_run()
    print("\n FINITE DATA TRIAL PASSED... \n")

    exp_cfg = parameter_sweep()
    print(f"\n PARAM SWEEP TRIAL PASSED...\n")

    
if __name__=="__main__":
    main()