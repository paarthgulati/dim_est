import torch
import torch.nn as nn
import torchvision.models as models
import importlib
from typing import List, Optional, Union, Tuple

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int, activation: str = "leaky_relu"):
        super().__init__()
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
        
        seq = []
        # Input layer
        seq.append(nn.Linear(input_dim, hidden_dim))
        seq.append(activation_fn())
        # Hidden layers
        for _ in range(layers):
            seq.append(nn.Linear(hidden_dim, hidden_dim))
            seq.append(activation_fn())
        # Output layer (linear projection to embedding)
        seq.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*seq)

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

def make_encoder(
    encoder_type: str,
    input_dim: int,
    embed_dim: int,
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
        return MLPEncoder(input_dim, hidden_dim, embed_dim, layers, activation)
    
    elif "resnet" in t:
        # kwargs: pretrained
        pretrained = kwargs.get("pretrained", True)
        return ResNetEncoder(embed_dim, variant=t, pretrained=pretrained)
    
    elif t == "cnn":
        # kwargs: input_channels, channels
        # Note: input_dim is treated as input_channels here if provided, 
        # but often input_dim from config is flat dimension. 
        # For CNNs, explicit 'input_channels' in kwargs is safer.
        input_channels = kwargs.get("input_channels", 3) 
        channels = kwargs.get("channels", [32, 64, 128])
        return CNNEncoder(input_channels, embed_dim, channels)
    
    elif t in ["gru", "lstm"]:
        hidden_size = kwargs.get("hidden_size", 128)
        num_layers = kwargs.get("num_layers", 1)
        return RNNEncoder(input_dim, hidden_size, embed_dim, rnn_type=t, num_layers=num_layers)
    
    elif t == "transformer":
        nhead = kwargs.get("nhead", 4)
        num_layers = kwargs.get("num_layers", 2)
        return TransformerEncoder(input_dim, embed_dim, nhead=nhead, num_layers=num_layers)
        
    elif t == "custom":
        # kwargs must contain 'class_path'
        class_path = kwargs.get("class_path")
        if not class_path:
            raise ValueError("For 'custom' encoder, 'class_path' must be provided in encoder_kwargs.")
        Cls = _load_custom_class(class_path)
        # We assume the custom class takes (input_dim, embed_dim, **kwargs)
        return Cls(input_dim, embed_dim, **kwargs)

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")