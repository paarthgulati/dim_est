TRAINING_DEFAULTS = {
    "infinite_data_iter": {
        "batch_size": 128,
        "n_iter": 2_000,
        "lr": 5e-4,
        "device": "cuda",
        "optimizer_kwargs": {},
        "show_progress": True,
        "track_cov_trace": False,     # Set to True in overrides to enable
        "cov_check_freq": 100,      # Frequency (in iterations) to track covariance trace
        "cov_sample_size": 1024,  # Number of samples to use when estimating covariance trace
        
    },
    "finite_data_epoch": {
        "batch_size": 128,
        "n_epoch": 100,
        "n_samples":4096,
        "n_test_samples": 128,
        "lr": 5e-4,
        "device": "cuda",
        "optimizer_kwargs": {},
        "show_progress": True,
        "patience": None,          # Options: None (disable) or int (patience epochs)
        "eval_train_mode": False,
        "eval_train_mode_final": True, # Find the last MI value if needed
        "device_final": 'cpu',        # Device to use for final train eval to avoid OOM
        "smooth_sigma": 1,
        "smooth_win": 5,
        "track_cov_trace": False,     # Set to True in overrides to enable
    },
}