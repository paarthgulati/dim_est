TRAINING_DEFAULTS = {
    "infinite_data_iter": {
        "batch_size": 128,
        "n_iter": 2_000,
        "lr": 5e-4,
        "device": "cuda",
        "optimizer_kwargs": {},
        "show_progress": True,
    },
    "finite_data_epoch": {
        "batch_size": 128,
        "n_epoch": 100,
        "n_samples":4096,
        "lr": 5e-4,
        "device": "cuda",
        "optimizer_kwargs": {},
        "show_progress": True,
    },
}