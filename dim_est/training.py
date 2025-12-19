import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


#for infinite data: fresh dataset of batch_size is generated every iteration, using the accompanying data_generator function
def train_model_infinite_data(model, data_generator, training_cfg: dict, optimizer_cls=torch.optim.Adam, device = 'cuda'):

    batch_size = training_cfg["batch_size"]
    n_iter = training_cfg["n_iter"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    show_progress = training_cfg["show_progress"]

    model.to(device)  
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi = []

    iterator = tqdm(range(n_iter)) if show_progress else range(n_iter)

    for i in iterator:
        x, y = data_generator(batch_size)
        
        opt.zero_grad()
        
        # Compute loss based on model type
        loss, mi, extras = model(x, y) 
        loss.backward()

        opt.step()

        estimator_tr = mi.to('cpu').detach().numpy()
        estimates_mi.append(estimator_tr)

    return np.array(estimates_mi)


def train_model_finite_data(
    model, 
    train_loader, 
    test_loader, 
    train_eval_loader, 
    training_cfg: dict, 
    optimizer_cls=torch.optim.Adam, 
    device='cuda'
):
    """
    Finite data training loop with early stopping and distinct train/test evaluation.
    
    Args:
        model: DSIB model
        train_loader: DataLoader for the full training set (shuffled)
        test_loader: DataLoader for the hold-out test set
        train_eval_loader: DataLoader for a subset of the training set (for tracking train MI)
        training_cfg: Config dict
        optimizer_cls: Optimizer class
        device: 'cuda' or 'cpu'
    """

    n_epoch = training_cfg["n_epoch"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {}) or {}
    show_progress = training_cfg.get("show_progress", True)
    
    # Early stopping parameters
    patience = training_cfg.get("patience", 10)
    best_test_mi = -float('inf')
    epochs_no_improve = 0

    model.to(device)  
    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi_train = []
    estimates_mi_test = []

    iterator = tqdm(range(n_epoch), desc="Epochs") if show_progress else range(n_epoch)

    for epoch in iterator:
        # --- 1. TRAIN LOOP ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, mi, extras = model(x, y) 
            loss.backward()
            opt.step()

        # --- 2. EVAL LOOP (End of Epoch) ---
        model.eval()
        
        # Helper to compute average MI over a loader
        # This computes the MI average as multiple batches, but the loader should be one batch ideally
        def evaluate_loader(loader):
            total_mi = 0.0
            steps = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    _, mi, _ = model(x, y)
                    total_mi += mi.item()
                    steps += 1
            return total_mi / max(1, steps)

        # Eval on Train Subset
        mi_train_avg = evaluate_loader(train_eval_loader)
        estimates_mi_train.append(mi_train_avg)

        # Eval on Test Set
        mi_test_avg = evaluate_loader(test_loader)
        estimates_mi_test.append(mi_test_avg)
        
        # --- 3. EARLY STOPPING CHECK ---
        if mi_test_avg > best_test_mi:
            best_test_mi = mi_test_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            if show_progress:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Test MI: {best_test_mi:.4f}")
            break

    return np.array(estimates_mi_train), np.array(estimates_mi_test)