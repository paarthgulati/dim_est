CRITIC_DEFAULTS = {
    "separable": dict(
        embed_dim=2,
        activation="leaky_relu",
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        Nx =500, Ny=500,
    ),

    "bi": dict(
        embed_dim=2,
        activation="leaky_relu",
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        Nx =500, Ny=500,
    ),

    "separable_augmented": dict(
        embed_dim=2,
        activation="leaky_relu",
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        quad_kind="full",
        Nx =500, Ny=500,
    ),

    "hybrid": dict(
        embed_dim=2,
        activation="leaky_relu",
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        pair_hidden_dim=64, pair_layers=1,
        Nx =500, Ny=500,
    ),

    "concat": dict(
        activation="leaky_relu",
        pair_hidden_dim=256, pair_layers=2,
        Nx =500, Ny=500,
    ),
}

DATASET_DEFAULTS = {
    "gaussian_mixture": dict(
        n_peaks=8,
        mu=2.0,
        sig=1.0,
        mi_bits_peak=0.0,
        sig_embed=0.0,
        noise_mode="white_relative",
        latent_dim=1,
        observe_dim_x=500,
        observe_dim_y=500,
    ),
    "joint_gaussian": dict(
        latent_dim=2,
        mi_bits=0.0,
        sig_embed=0.0,
        noise_mode="white_relative",
        observe_dim_x=500,
        observe_dim_y=500,
    ),
    "ring_with_spread": dict(
        mu=4.0,
        radial_std=0.1,
        sig_embed=0.0,
        noise_mode="white_relative",
        latent_dim=1,
        observe_dim_x=500,
        observe_dim_y=500,
    ),
    
    "hyperspherical_shell": dict(
        mu=4.0,
        radial_std=2.0,
        sig_embed=1.0,
        noise_mode="white_relative",
        latent_dim=4,
        observe_dim_x=500,
        observe_dim_y=500,
    ),
    
    "swiss_roll": dict(
        t_min_pi_units=1.5,
        t_max_pi_units=4.5,
        height_min = 0.0,
        height_max = 15.0,
        sig_embed=1.0,
        noise_mode="white_relative",
        latent_dim=3,
        observe_dim_x=500,
        observe_dim_y=500,
    ),

}
