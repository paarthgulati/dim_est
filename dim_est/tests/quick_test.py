from dim_est.run.run_dsib_single_experiment import run_dsib_infinite, run_dsib_finite


def single_finite_run(outfile = "trial_run.h5"):
    training_overrides = dict(n_epoch=5, n_samples = 20)
    mi_trial, exp_cfg = run_dsib_finite(outfile=outfile, training_overrides = training_overrides)
    print(f"Experimental config: {exp_cfg}")
    return exp_cfg

def single_infinite_run(outfile = "trial_run.h5"):
    training_overrides = dict(n_iter=10)
    mis, exp_cfg = run_dsib_infinite(outfile=outfile, training_overrides = training_overrides)
    print(f"Experimental config: {exp_cfg}")

    return exp_cfg

def run_tests():
    exp_cfg = single_infinite_run()
    print(f"\n INFINITE DATA TRIAL PASSED... \n")

    exp_cfg = single_finite_run()
    print("\n FINITE DATA TRIAL PASSED... \n")

    # exp_cfg = parameter_sweep()
    # print(f"\n PARAM SWEEP TRIAL PASSED...\n")

    
if __name__=="__main__":
    run_testa()