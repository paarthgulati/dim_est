import json
import hashlib
import h5py

from dim_est.datasets.data_generation import make_data_generator
from cca_zoo.linear import CCA
import copy
from tqdm import tqdm
import numpy as np

def cfg_hash(ds_cfg: dict, n_samples: int) -> str:
    payload = {
        "dataset_cfg": ds_cfg,
        "n_samples": int(n_samples),
    }
    cfg_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()


def get_cached_cca_value(cache_path: str, ds_cfg: dict, n_samples: int, kz: int):
    """
    Look up cached CCA MI value for (ds_cfg, n_samples, kz).

    Returns:
        float value if present, or None if not cached.
    """
    key = cfg_hash(ds_cfg, n_samples)
    dset_name = f"kz_{int(kz)}"

    with h5py.File(cache_path, "a") as f:
        if key not in f:
            return None

        grp = f[key]
        if dset_name not in grp:
            return None

        return float(grp[dset_name][()])


def set_cached_cca_value(
    cache_path: str,
    ds_cfg: dict,
    n_samples: int,
    kz: int,
    value: float,
):
    """
    Store a single CCA MI value for (ds_cfg, n_samples, kz).
    Overwrites if it already exists.
    """
    key = cfg_hash(ds_cfg, n_samples)
    dset_name = f"kz_{int(kz)}"

    with h5py.File(cache_path, "a") as f:
        if key in f:
            grp = f[key]
        else:
            grp = f.create_group(key)
            grp.attrs["ds_cfg_json"] = json.dumps(ds_cfg, sort_keys=True)
            grp.attrs["n_samples"] = int(n_samples)

        if dset_name in grp:
            del grp[dset_name]

        grp.create_dataset(dset_name, data=float(value))


def mut_info_optimized(x, y, threshold=1e-10):
    """
    Calculates -1/2 log det(rho) and removes the near zero directions
    """
    try:
        # Combine x and y column-wise (variables are columns)
        xy = np.hstack((x, y))
        
        # Compute joint covariance matrix once
        c_tot = np.cov(xy, rowvar=False)
        n_x = x.shape[1]  # Number of features in X
        n_y = y.shape[1]  # Number of features in Y
        
        # Extract C_x and C_y from the joint covariance matrix
        c_x = c_tot[:n_x, :n_x]
        c_y = c_tot[n_x:, n_x:]
        
        # Compute eigenvalues using eigh (faster for symmetric matrices)
        eig_tot = np.linalg.eigh(c_tot)[0]  # Returns sorted eigenvalues (ascending)
        eig_x = np.linalg.eigh(c_x)[0]
        eig_y = np.linalg.eigh(c_y)[0]
        
        # Threshold eigenvalues (avoid log(0))
        eig_tot_thr = np.maximum(eig_tot, threshold)
        eig_x_thr = np.maximum(eig_x, threshold)
        eig_y_thr = np.maximum(eig_y, threshold)
        
        # Compute log determinants
        logdet_tot = np.sum(np.log2(eig_tot_thr))
        logdet_x = np.sum(np.log2(eig_x_thr))
        logdet_y = np.sum(np.log2(eig_y_thr))
        
        # Mutual information
        info = 0.5 * (logdet_x + logdet_y - logdet_tot)
        return info if not np.isinf(info) else np.nan
    except np.linalg.LinAlgError:
        return np.nan


def generate_CCA_mi_estimate(data_generator, ds_cfg, n_samples, kz, use_cache= True, cache_path = "cca_mi_estimates.h5", write_to_cache=True):

    if use_cache:
        cached = get_cached_cca_value(cache_path, ds_cfg, n_samples, kz)
        if cached is not None:
            return cached

    # data_generator = make_data_generator(dataset_type, ds_cfg, device="cpu")

    ## if not using cache or no existing cache:          
    x_data, y_data = data_generator(n_samples)
    x_data, y_data = x_data.numpy(), y_data.numpy()

    cca_mdl = CCA(latent_dimensions=kz);
    cca_mdl.fit((x_data,y_data));    
    x_data_transform, y_data_transform = cca_mdl.transform((x_data,y_data))
    mi_cca_opt = mut_info_optimized(x_data_transform, y_data_transform)

    if write_to_cache:
        set_cached_cca_value(cache_path, ds_cfg, n_samples, kz, mi_cca_opt)    
    return mi_cca_opt
