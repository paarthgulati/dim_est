CRITIC_DEFAULTS = {
    "separable": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        encoder_type="mlp",
        share_encoder=False,
        use_norm=True,
        dropout=0.0,
        encoder_kwargs={}
    ),

    "bi": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        encoder_type="mlp",
        share_encoder=False,
        use_norm=True,
        dropout=0.0,
        encoder_kwargs={}
    ),

    "separable_augmented": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        quad_kind="full",
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        encoder_type="mlp",
        share_encoder=False,
        use_norm=True,
        dropout=0.0,
        encoder_kwargs={}
    ),

    "hybrid": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        pair_hidden_dim=64, pair_layers=1, # Slightly larger default for robustness
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        encoder_type="mlp",
        share_encoder=False,
        use_norm=True,
        dropout=0.0,
        encoder_kwargs={}
    ),

    "concat": dict(
        pair_hidden_dim=128, pair_layers=2,
        Nx=500, Ny=500,
        activation="leaky_relu",
        use_norm=True,
        dropout=0.0,
    )
}