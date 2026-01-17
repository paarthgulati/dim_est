import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import copy
from scipy.ndimage import gaussian_filter1d, median_filter

def smooth(arr, sigma=1, med_win=5):
    """
    Smooths a noisy trace using median filtering (outliers) then gaussian filtering.
    """
    hist = np.array(arr)
    if len(hist) < 2: return hist
    nan_mask = np.isnan(hist)
    valid_hist = hist[~nan_mask]
    if len(valid_hist) == 0: return hist
    
    # Fill NaNs with last valid
    hist[nan_mask] = valid_hist[-1]
    
    # 1. Median Filter (Robust to spikes)
    if med_win > 1 and len(hist) >= med_win:
        hist = median_filter(hist, size=med_win, mode='nearest')
        
    # 2. Gaussian Filter (Smoothing)
    if sigma > 0: 
        hist = gaussian_filter1d(hist, sigma=sigma, mode='nearest')
        
    hist[nan_mask] = np.nan
    return hist

def calculate_participation_ratio(spectrum, eps=1e-12):
    """
    Computes participation ratio from eigenvalues.
    PR = (sum(lambda))^2 / sum(lambda^2)
    
    Args:
        spectrum (np.ndarray): Array of singular values.
        eps (float): Small value to prevent division by zero.
    """
    # Convert singular values to eigenvalues of covariance (lambda = sigma^2)
    lam = spectrum ** 2
    
    numerator = np.sum(lam) ** 2
    denominator = np.sum(lam ** 2).clip(min=eps)

    return numerator / denominator

def evaluate_full_dataset(model, loader, device, max_samples_gpu=5000):
    """
    Computes MI on the entire dataset in the loader.
    Auto-offloads to CPU if dataset > max_samples_gpu to avoid OOM.
    """
    xs, ys = [], []
    total_samples = 0
    
    with torch.no_grad():
        for x_b, y_b in loader:
            xs.append(x_b) 
            ys.append(y_b)
            total_samples += x_b.size(0)

    use_cpu = (total_samples > max_samples_gpu) or (device == 'cpu')
    eval_device = 'cpu' if use_cpu else device
    
    previous_device = next(model.parameters()).device
    if use_cpu: model.cpu()
    else: model.to(eval_device)
    
    model.eval()
    
    mi_val = np.nan
    try:
        x_full = torch.cat(xs, dim=0).to(eval_device)
        y_full = torch.cat(ys, dim=0).to(eval_device)
        
        with torch.no_grad():
            _, mi, _ = model(x_full, y_full)
            mi_val = mi.item()
    except RuntimeError as e:
        if "out of memory" in str(e) and not use_cpu:
            print("OOM during full-batch eval on GPU. Falling back to CPU...")
            torch.cuda.empty_cache()
            return evaluate_full_dataset(model, loader, device, max_samples_gpu=0)
        else:
            raise e
    finally:
        if use_cpu: model.to(previous_device)
            
    return mi_val

def compute_covariance_matrices(model, loader_or_data, device, max_samples_gpu=5000):
    """
    Computes covariance matrices on Device, then decomposes on CPU to avoid MPS issues.
    Accepts either a DataLoader OR a tuple of (x_full, y_full) tensors.
    """
    
    # 1. Prepare Data
    if isinstance(loader_or_data, tuple):
        # Case: Infinite data passed as tensors
        x_full, y_full = loader_or_data
        total_samples = x_full.size(0)
        use_cpu = (total_samples > max_samples_gpu) or (device == 'cpu')
    else:
        # Case: DataLoader
        xs, ys = [], []
        total_samples = 0
        with torch.no_grad():
            for x_b, y_b in loader_or_data:
                xs.append(x_b)
                ys.append(y_b)
                total_samples += x_b.size(0)
        
        use_cpu = (total_samples > max_samples_gpu) or (device == 'cpu')
        
        # We will concat later after model move logic
    
    eval_device = 'cpu' if use_cpu else device
    previous_device = next(model.parameters()).device
    
    if use_cpu: model.cpu()
    else: model.to(eval_device)
    
    model.eval()
    
    results = {}
    try:
        if not isinstance(loader_or_data, tuple):
            x_full = torch.cat(xs, dim=0).to(eval_device)
            y_full = torch.cat(ys, dim=0).to(eval_device)
        else:
            x_full = x_full.to(eval_device)
            y_full = y_full.to(eval_device)

        with torch.no_grad():
            if hasattr(model, 'critic'):
                zx = model.critic.encoder_x(x_full)
                zy = model.critic.encoder_y(y_full)
            else:
                raise AttributeError("Model does not have a 'critic' attribute.")

            # 1. Center Embeddings
            zx = zx - zx.mean(dim=0, keepdim=True)
            zy = zy - zy.mean(dim=0, keepdim=True)
            
            N = zx.size(0)
            
            # 2. Compute Matrices on GPU/MPS (Fast MatMul)
            cov_xy = torch.matmul(zx.T, zy) / (N - 1)
            
            # 3. Move to CPU for Decomposition (Avoids MPS crashes)
            cov_xy_cpu = cov_xy.cpu()
            
            # 4. Decompose
            _, s_xy, _ = torch.linalg.svd(cov_xy_cpu)
            
            results = {
                "spectrum_xy": s_xy.numpy(),
                "cov_xy": cov_xy_cpu.numpy(),
            }

    except RuntimeError as e:
        if "out of memory" in str(e) and not use_cpu:
            torch.cuda.empty_cache()
            return compute_covariance_matrices(model, loader_or_data, device, max_samples_gpu=0)
        raise e
    finally:
        if use_cpu: model.to(previous_device)
            
    return results

def train_model_infinite_data(
    model, 
    data_generator, 
    training_cfg: dict, 
    optimizer_cls=torch.optim.Adam, 
    device='cuda'
):
    """
    Infinite data training with periodic Spectral Analysis support.
    """
    batch_size = training_cfg["batch_size"]
    n_iter = training_cfg["n_iter"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {}) or {}
    show_progress = training_cfg.get("show_progress", True)
    
    # Spectrum Tracking Params
    track_cov_trace = training_cfg.get("track_cov_trace", False)
    # Default: Check every 1% of iterations or at least every 100 steps
    cov_check_freq = training_cfg.get("cov_check_freq", max(100, int(n_iter * 0.01)))
    cov_sample_size = training_cfg.get("cov_sample_size", 2048) # Samples for spectral est

    model.to(device)  
    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi = []
    
    # Trace containers
    cov_trace_xy = []
    pr_trace_xy = []
    steps_trace = []

    iterator = tqdm(range(n_iter)) if show_progress else range(n_iter)

    for i in iterator:
        x, y = data_generator(batch_size)
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        loss, mi, _ = model(x, y) 
        loss.backward()
        opt.step()

        estimator_tr = mi.item()
        estimates_mi.append(estimator_tr)
        
        # --- Periodic Spectral Analysis ---
        if track_cov_trace and (i % cov_check_freq == 0 or i == n_iter - 1):
            # Generate a dedicated large batch for stable spectral estimation
            with torch.no_grad():
                x_val, y_val = data_generator(cov_sample_size)
                # Pass tuple directly to helper
                cov_data = compute_covariance_matrices(model, (x_val, y_val), device)
                
                s_xy = cov_data["spectrum_xy"]
                cov_trace_xy.append(s_xy)
                pr_trace_xy.append(calculate_participation_ratio(s_xy))
                steps_trace.append(i)
                
                # Switch back to training mode
                model.train()

        if show_progress:
            postfix = {"mi": f"{estimator_tr:.4f}"}
            if track_cov_trace and pr_trace_xy:
                 postfix["PR"] = f"{pr_trace_xy[-1]:.2f}"
            iterator.set_postfix(**postfix)

    # Package results
    final_cov_results = {
        "mi_trace": np.array(estimates_mi),
        "cov_results": {
            "trace_spectrum_xy": np.array(cov_trace_xy) if track_cov_trace else None,
            "trace_pr_xy": np.array(pr_trace_xy) if track_cov_trace else None,
            "steps": np.array(steps_trace) if track_cov_trace else None
        }
    }
    return np.array(estimates_mi), final_cov_results

def train_model_finite_data(
    model, 
    train_loader, 
    test_loader, 
    training_cfg: dict, 
    train_subset_loader=None, 
    optimizer_cls=torch.optim.Adam, 
    device='cuda'
):
    """
    Finite data training loop with robust Max-Test logic, TTUR, Cyclic Scheduling,
    and Hybrid Covariance Spectrum Analysis.
    """
    # 1. Config unpacking
    n_epoch = training_cfg["n_epoch"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {}) or {}
    show_progress = training_cfg.get("show_progress", True)
    patience = training_cfg.get("patience", None) 

    eval_train_mode = training_cfg.get("eval_train_mode", False) 
    eval_train_mode_final = training_cfg.get("eval_train_mode_final", True) 
    device_final = training_cfg.get("device_final", 'cpu')
    smooth_sigma = training_cfg.get("smooth_sigma", 1)
    smooth_win = training_cfg.get("smooth_win", 5)
    
    track_cov_trace = training_cfg.get("track_cov_trace", False)

    model.to(device)  
    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)


    best_smoothed_test_mi = -float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    estimates_mi_train = []
    estimates_mi_test = [] 
    
    # Traces for Spectrum & PR
    cov_trace_train = [] 
    cov_trace_test = []
    pr_trace_train = []
    pr_trace_test = []

    iterator = tqdm(range(n_epoch), desc="Epochs") if show_progress else range(n_epoch)

    for epoch in iterator:
        # --- A. TRAIN LOOP ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, _, _ = model(x, y) 
            loss.backward()
            opt.step()
        

        # --- B. EVAL LOOP ---
        mi_test_curr = evaluate_full_dataset(model, test_loader, device)
        estimates_mi_test.append(mi_test_curr)
        
        mi_train_curr = np.nan
        should_eval_full = (eval_train_mode is True) or (eval_train_mode == 'full')
        should_eval_subset = isinstance(eval_train_mode, int) and (train_subset_loader is not None)
        
        if should_eval_full:
            mi_train_curr = evaluate_full_dataset(model, train_loader, device)
        elif should_eval_subset:
            mi_train_curr = evaluate_full_dataset(model, train_subset_loader, device)
        estimates_mi_train.append(mi_train_curr)
        
        # --- Covariance & PR Tracking ---
        if track_cov_trace:
            # Test Spectrum
            cov_data = compute_covariance_matrices(model, test_loader, device)
            s_xy_test = cov_data["spectrum_xy"]
            cov_trace_test.append(s_xy_test)
            pr_trace_test.append(calculate_participation_ratio(s_xy_test))
            
            # Train Spectrum (Optional)
            cov_data_tr = compute_covariance_matrices(model, train_loader, device)
            s_xy_train = cov_data_tr["spectrum_xy"]
            cov_trace_train.append(s_xy_train)
            pr_trace_train.append(calculate_participation_ratio(s_xy_train))

        if show_progress: 
            postfix = {"test_mi": f"{mi_test_curr:.4f}"}
            if track_cov_trace and pr_trace_test:
                postfix["PR"] = f"{pr_trace_test[-1]:.2f}"
            iterator.set_postfix(**postfix)

        # --- C. CHECKPOINTING ---
        smoothed_history = smooth(estimates_mi_test, sigma=smooth_sigma, med_win=smooth_win)
        current_smoothed_score = smoothed_history[-1]
        
        if current_smoothed_score > best_smoothed_test_mi:
            best_smoothed_test_mi = current_smoothed_score
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if patience is not None and epochs_no_improve >= patience:
            if show_progress:
                print(f"Early stopping at epoch {epoch}. Best Smoothed Test MI: {best_smoothed_test_mi:.4f}")
            break

    # --- 4. FINALIZE ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Metrics
    final_test_mi = evaluate_full_dataset(model, test_loader, device)

    if eval_train_mode_final:
        final_train_mi = evaluate_full_dataset(model, train_loader, device=device_final)
    else:
        final_train_mi = np.nan
    
    # Full Covariance Dump (Best Model)
    final_cov_data_train = compute_covariance_matrices(model, train_loader, device)
    final_cov_data_test = compute_covariance_matrices(model, test_loader, device)

    # Final PR Results (Best Model)
    final_pr_test = calculate_participation_ratio(final_cov_data_test["spectrum_xy"])
    final_pr_train = calculate_participation_ratio(final_cov_data_train["spectrum_xy"])

    final_cov_results = {
        "train": final_cov_data_train,
        "test": final_cov_data_test,
        "trace_spectrum_test": np.array(cov_trace_test) if track_cov_trace else None,
        "trace_spectrum_train": np.array(cov_trace_train) if track_cov_trace else None,
        "trace_pr_test": np.array(pr_trace_test) if track_cov_trace else None,
        "trace_pr_train": np.array(pr_trace_train) if track_cov_trace else None,
    }

    return np.array(estimates_mi_train), np.array(estimates_mi_test), final_train_mi, final_test_mi, final_pr_train, final_pr_test, final_cov_results