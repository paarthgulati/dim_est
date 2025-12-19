import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RandomFeatureDataset(Dataset):
    """
    Splits a flat vector X into two sub-vectors X_view and Y_view based on a random mask.
    Useful for 'independent dimensions' assumption.
    """
    def __init__(self, data_tensor, fraction=0.5, seed=42):
        super().__init__()
        self.data = data_tensor
        self.fraction = fraction
        
        # Determine split mask once (fixed per run)
        # We assume data is (N, D) or (N, C, H, W) flattened? 
        # Usually this applies to flat feature vectors (N, D).
        D = data_tensor.shape[1]
        n_x = int(D * fraction)
        
        # Create permutation
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(D, generator=g)
        
        self.indices_x = perm[:n_x]
        self.indices_y = perm[n_x:]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        return row[self.indices_x], row[self.indices_y]


class TemporalDataset(Dataset):
    """
    Pairs X_t with Y_{t+lag}.
    Effective length is N - lag.
    """
    def __init__(self, data_tensor, lag=1):
        super().__init__()
        self.data = data_tensor
        self.lag = lag
        if self.lag < 1:
            raise ValueError(f"Lag must be >= 1, got {lag}")
        if self.lag >= len(self.data):
             raise ValueError(f"Lag {lag} is too large for dataset of size {len(self.data)}")

    def __len__(self):
        return len(self.data) - self.lag

    def __getitem__(self, idx):
        # x = frame t, y = frame t+lag
        return self.data[idx], self.data[idx + self.lag]


class SpatialSplitDataset(Dataset):
    """
    Splits an image tensor (C, H, W) into two spatial halves.
    axis=1 -> Split height (Top/Bottom)
    axis=2 -> Split width (Left/Right)
    """
    def __init__(self, data_tensor, axis=2):
        super().__init__()
        self.data = data_tensor
        self.axis = axis # Dimension to split along (0 is channel, 1 is height, 2 is width)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        # img shape: (C, H, W)
        dim_size = img.shape[self.axis]
        mid = dim_size // 2
        
        # Slicing logic
        if self.axis == 1: # Height
            x_view = img[:, :mid, :]
            y_view = img[:, mid:, :]
        elif self.axis == 2: # Width
            x_view = img[:, :, :mid]
            y_view = img[:, :, mid:]
        else:
            # Fallback or Channel split
            # Using torch.split might be cleaner but manual slice is explicit
            x_view, y_view = torch.split(img, mid, dim=self.axis)
            
        return x_view, y_view


class AugmentationDataset(Dataset):
    """
    Applies two independent random augmentations to the same input.
    SimCLR / Self-Supervised style.
    """
    def __init__(self, data_tensor, transform_spec="crop_flip"):
        super().__init__()
        self.data = data_tensor
        
        # Build transform pipeline based on spec string
        # In a real app, we might pass a full Compose object, but for config serialization we use strings.
        transforms_list = []
        
        # Assume input is (C, H, W) scaled 0-1 or normalized
        if "crop" in transform_spec:
            # We need to know image size. Assuming data is uniform size.
            # If tensor is (N, C, H, W)
            _, h, w = data_tensor[0].shape
            transforms_list.append(T.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0)))
            
        if "flip" in transform_spec:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            
        if "color" in transform_spec:
             transforms_list.append(T.ColorJitter(0.4, 0.4, 0.4, 0.1))
             
        if not transforms_list:
            # Default identity if spec is empty or unknown
            transforms_list.append(T.Lambda(lambda x: x))
            
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # Apply transform twice independently
        # Note: input img is Tensor. T.RandomResizedCrop works on Tensor.
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2