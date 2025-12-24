CRITIC_DEFAULTS = {
    "separable": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        # New Phase 2 defaults
        encoder_type="mlp",
        share_encoder=False,
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
        encoder_kwargs={}
    ),

    "hybrid": dict(
        embed_dim=2,
        x_hidden_dim=128, x_layers=2,
        y_hidden_dim=128, y_layers=2,
        pair_hidden_dim=64, pair_layers=1,
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        encoder_type="mlp",
        share_encoder=False,
        encoder_kwargs={}
    ),

    "concat": dict(
        pair_hidden_dim=256, pair_layers=2,
        Nx=500, Ny=500,
        activation="leaky_relu",
        
        # Concat usually doesn't have separate encoders, but if we add feature extractors later:
        encoder_type="none", 
        share_encoder=False,
        encoder_kwargs={}
    ),
}