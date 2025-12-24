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
        # Handle (N, C, H, W) by flattening first if needed, 
        # but usually we expect the user to flatten before or handle inside
        if data_tensor.dim() > 2:
            # Flatten everything after batch dim for indexing
            self.flat_dim = data_tensor[0].numel()
            self.flatten = True
        else:
            self.flat_dim = data_tensor.shape[1]
            self.flatten = False

        n_x = int(self.flat_dim * fraction)
        
        # Create permutation
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(self.flat_dim, generator=g)
        
        self.indices_x = perm[:n_x]
        self.indices_y = perm[n_x:]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        if self.flatten:
            row = row.view(-1)
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
    Splits an image tensor (C, H, W) into two spatial views.
    
    Modes:
      - 'axis': Splits along a cardinal axis (returns rectangular tensors).
           axis=1 -> Split height (Top/Bottom)
           axis=2 -> Split width (Left/Right)
      - 'diagonal': Splits along the diagonal (returns FLATTENED vectors).
           diagonal_dir=1 -> Main diagonal (Top-Left / Bottom-Right)
           diagonal_dir=-1 -> Anti-diagonal (Top-Right / Bottom-Left)
    """
    def __init__(self, data_tensor, mode="axis", axis=2, diagonal_dir=1):
        super().__init__()
        self.data = data_tensor
        self.mode = mode
        self.axis = axis 
        self.diagonal_dir = diagonal_dir
        
        # Precompute diagonal indices if needed
        if self.mode == "diagonal":
            # Assume data is (N, C, H, W) or (N, 1, H, W)
            # We treat C as part of the features per pixel or flatten it?
            # For simplicity, we flatten C, H, W entirely, but compute mask based on H, W.
            if data_tensor.dim() != 4:
                raise ValueError("Diagonal split requires (N, C, H, W) input.")
            
            _, C, H, W = data_tensor.shape
            
            # Create grid
            row_idx = torch.arange(H).unsqueeze(1).repeat(1, W)
            col_idx = torch.arange(W).unsqueeze(0).repeat(H, 1)
            
            if diagonal_dir == 1:
                # Main diagonal: row < col vs row >= col
                mask_x = (row_idx < col_idx) # Upper Triangle
            else:
                # Anti-diagonal: row + col < W
                mask_x = ((row_idx + col_idx) < W) # Top-Left Triangle
            
            # Expand mask to C channels: (C, H, W)
            # We want to keep all channels for a selected pixel
            mask_x = mask_x.unsqueeze(0).repeat(C, 1, 1) # (C, H, W)
            mask_y = ~mask_x
            
            # Convert to flat indices
            self.indices_x = torch.where(mask_x.view(-1))[0]
            self.indices_y = torch.where(mask_y.view(-1))[0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx] # (C, H, W)
        
        if self.mode == "diagonal":
            flat_img = img.view(-1)
            return flat_img[self.indices_x], flat_img[self.indices_y]
        
        else: # mode == "axis"
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
            transforms_list.append(T.Lambda(lambda x: x))
            
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2