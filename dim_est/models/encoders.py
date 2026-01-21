import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import importlib
from typing import List, Optional, Union, Tuple

# --- Import the robust mlp classes ---
from ..utils.networks import mlp

class MLPEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        layers: int, 
        activation: str = "leaky_relu",
        use_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Use the robust mlp class from networks.py
        self.net = mlp(
            dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=layers,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout
        )

    def forward(self, x):
        # Flatten if input is > 2D (e.g. image provided to MLP)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)        

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim: int, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        # Load torchvision model
        if not hasattr(models, variant):
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # weights="DEFAULT" corresponds to best available pretrained weights
        weights = "DEFAULT" if pretrained else None
        self.net = getattr(models, variant)(weights=weights)
        
        # Replace fc layer
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        return self.net(x)

class CNNEncoder(nn.Module):
    """
    Base CNN: Sequence of Conv2d -> ReLU -> MaxPool.
    Followed by global average pooling and a linear projection.
    """
    def __init__(self, input_channels: int, output_dim: int, channels: List[int] = [32, 64, 128]):
        super().__init__()
        layers = []
        in_c = input_channels
        
        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_c = out_c
            
        self.features = nn.Sequential(*layers)
        self.projector = nn.Linear(in_c, output_dim)

    def forward(self, x):
        x = self.features(x)
        # Global Average Pooling: (B, C, H, W) -> (B, C)
        x = x.mean(dim=[2, 3]) 
        x = self.projector(x)
        return x

class RNNEncoder(nn.Module):
    """
    GRU or LSTM encoder. Returns the last hidden state mapped to output_dim.
    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int, rnn_type="gru", num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
            
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x expected: (Batch, Seq, Feature)
        if self.rnn_type == "gru":
            _, hn = self.rnn(x) # hn: (num_layers, B, H)
        else:
            _, (hn, _) = self.rnn(x)
            
        # Take last layer's hidden state
        last_hidden = hn[-1] 
        return self.fc(last_hidden)

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder. 
    Uses a learnable CLS token or Mean Pooling (default: Mean Pooling for simplicity).
    """
    def __init__(self, input_size: int, output_dim: int, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        # Simple projection to internal model dim if needed, here we assume input_size IS model dim or projected
        self.d_model = input_size 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        out = self.transformer(x)
        # Mean pooling over sequence
        pooled = out.mean(dim=1)
        return self.fc(pooled)

def _load_custom_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class VariationalAdapter(nn.Module):
    def __init__(self, base_encoder: nn.Module, feature_dim: int, embed_dim: int):
        """
        Wraps any deterministic encoder to make it variational.
        Input -> [Base Encoder] -> Feature (h) -> [Heads] -> (z, kl_loss)
        """
        super().__init__()
        self.base_encoder = base_encoder
        
        # Variational heads
        self.fc_mu = nn.Linear(feature_dim, embed_dim)
        self.fc_logvar = nn.Linear(feature_dim, embed_dim)
        
        # Init weights
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x):
        # 1. Get deterministic features from the backbone
        h = self.base_encoder(x)
        
        # 2. Project to distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            # Reparameterization Trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Analytic KL for standard normal prior N(0, I)
            # Sum over dimensions, mean over batch
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            return z, kl_loss
        else:
            # Deterministic evaluation (return mean, zero KL)
            return mu, 0.0
        
def make_encoder(
    encoder_type: str,
    input_dim: int,
    embed_dim: int,
    variational: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to build encoders.
    """
    t = encoder_type.lower()
    
    if t == "mlp":
        # Default args for MLP if not in kwargs
        hidden_dim = kwargs.get("hidden_dim", 128)
        layers = kwargs.get("layers", 2)
        activation = kwargs.get("activation", "leaky_relu")
        use_norm = kwargs.get("use_norm", False)
        dropout = kwargs.get("dropout", 0.0)
        
        enc = MLPEncoder(
            input_dim, 
            hidden_dim, 
            embed_dim, 
            layers, 
            activation, 
            use_norm=use_norm, 
            dropout=dropout
        )
    
    elif "resnet" in t:
        pretrained = kwargs.get("pretrained", True)
        enc = ResNetEncoder(embed_dim, variant=t, pretrained=pretrained)
        
    elif t == "cnn":
        input_channels = kwargs.get("input_channels", 3) 
        channels = kwargs.get("channels", [32, 64, 128])
        enc = CNNEncoder(input_channels, embed_dim, channels)
        
    elif t in ["gru", "lstm"]:
        hidden_size = kwargs.get("hidden_size", 128)
        num_layers = kwargs.get("num_layers", 1)
        enc = RNNEncoder(input_dim, hidden_size, embed_dim, rnn_type=t, num_layers=num_layers)
        
    elif t == "transformer":
        nhead = kwargs.get("nhead", 4)
        num_layers = kwargs.get("num_layers", 2)
        enc = TransformerEncoder(input_dim, embed_dim, nhead=nhead, num_layers=num_layers)
        
    elif t == "custom":
        class_path = kwargs.get("class_path")
        if not class_path:
            raise ValueError("For 'custom' encoder, 'class_path' must be provided in encoder_kwargs.")
        Cls = _load_custom_class(class_path)
        enc = Cls(input_dim, embed_dim, **kwargs)

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    if variational:
        # Wrap the deterministic encoder. 
        # Assumes base encoder outputs 'embed_dim' features.
        enc = VariationalAdapter(base_encoder=enc, feature_dim=embed_dim, embed_dim=embed_dim)

    return enc