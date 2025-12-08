_TRANSFORM_PRESETS = {
    "teacher": dict(
        mode="teacher",
        observe_dim_x=500,
        observe_dim_y=500,
        sig_embed_x=0.0,
        sig_embed_y=0.0,
        noise_mode="white_relative",
    ),
    "identity": dict(
        mode="identity",
        # obs dims inferred from latent_dim in this mode
        observe_dim_x=None,
        observe_dim_y=None,
        sig_embed_x=0.0,
        sig_embed_y=0.0,
        noise_mode="none",
    ),
    "linear": dict(
        mode="linear",
        observe_dim_x=500,
        observe_dim_y=500,
        sig_embed_x=0.0,
        sig_embed_y=0.0,
        noise_mode="white_relative",
    ),
}

DATASET_DEFAULTS = {
    "gaussian_mixture": dict(
        latent=dict(
            n_peaks=8,
            mu=2.0,
            sig=1.0,
            mi_bits_peak=2.0,
            latent_dim=1,
        ),
        # transform is part of *dataset*, not a separate top-level block
        transform=_TRANSFORM_PRESETS["teacher"],
    ),

    "joint_gaussian": dict(
        latent=dict(
            latent_dim=2,
            mi_bits=0.0,
        ),
        transform=_TRANSFORM_PRESETS["teacher"],
    ),

    "ring_with_spread": dict(
        latent=dict(
            mu=4.0,
            radial_std=0.1,
            latent_dim=1,
        ),
        transform=_TRANSFORM_PRESETS["teacher"],
    ),

    "hyperspherical_shell": dict(
        latent=dict(
            mu=4.0,
            radial_std=2.0,
            latent_dim=4,
        ),
        transform=_TRANSFORM_PRESETS["teacher"],
    ),

    "swiss_roll": dict(
        latent=dict(
            t_min_pi_units=1.5,
            t_max_pi_units=4.5,
            height_min=0.0,
            height_max=15.0,
            latent_dim=3,
        ),
        transform=_TRANSFORM_PRESETS["teacher"],
    ),
}
