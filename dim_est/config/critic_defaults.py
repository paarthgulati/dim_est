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
