import numpy as np
import torch
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter1d, median_filter
from matplotlib.ticker import MaxNLocator
import math
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from typing import Optional, Dict
import itertools
from ..utils.h5_result_store import H5ResultStore


## summarize based on the test-train heuristic

import tempfile
import shutil
import os
from contextlib import contextmanager
## DON'T LOAD H5 files DIRECTLY. Make a temporary copy first to avoid interrupting the writer

@contextmanager
def temporary_h5_snapshot(src_path, suffix=".h5"):
    """
    Creates a temporary copy of an HDF5 file and yields the path.
    Deletes the copy on exit.
    """
    # Create temp file in /tmp or TMPDIR
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # close low-level file descriptor
    
    # Copy live file → temp file
    shutil.copy2(src_path, temp_path)
    
    try:
        yield temp_path
    finally:
        # Always delete the temp file
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def _deep_get(d, dotted_key):
    """
    Safe lookup for nested metadata:
        dotted_key = 'params.base_critic_params.Nx'
    Returns None if missing anywhere.
    """
    keys = dotted_key.split(".")
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


## get plotting metrics from MI trace over iterations
def summarize_mi_traces_test_train(
    mi_bits_test: np.ndarray, 
    mi_bits_train: np.ndarray,
    ema_span: int = 20,
    smooth_sigma: float = 1.0,
    tail_fraction: float = 0.5,
):
    """
    """

    # Ensure integer kernel size (robust to float inputs)
    ema_span = max(1, int(round(ema_span)))
    # Replace NaNs just in case
    mi_test = np.nan_to_num(mi_bits_test)
    mi_train = np.nan_to_num(mi_bits_train)

    # ---- Smooth full trace ----
    smoothed_full_test = gaussian_filter1d(
        median_filter(mi_test, size=ema_span),
        sigma=smooth_sigma,
    )
    smoothed_full_train = gaussian_filter1d(
        median_filter(mi_train, size=ema_span),
        sigma=smooth_sigma,
    )

    max_test_idx = np.argmax(smoothed_full_test)
    train_est = smoothed_full_train[max_test_idx]


    return train_est, max_test_idx

## Data aggregation functions: given axes to sweep over collectively return a dataframe with rows composed of these parameters and the summary metrics I asked for

def iter_sweep_queries(sweep_tags, extra_filters=None):
    """
    Given:
        sweep_tags = {
            'tags.dataset_type': [...],
            'tags.critic_type': [...],
            ...
        }
    Yields (where_dict, context_dict) for each combination.

    where_dict  -> used directly in rs.query(where=...)
    context_dict -> simplified keys, e.g. 'dataset_type' instead of 'tags.dataset_type'
    """
    if extra_filters is None:
        extra_filters = {}

    keys = list(sweep_tags.keys())
    value_lists = [list(v) for v in sweep_tags.values()]

    for combo in itertools.product(*value_lists):
        where = dict(zip(keys, combo))
        where.update(extra_filters)

        # Build a more human-friendly context
        context = {}
        for k, v in where.items():
            # 'tags.estimator' -> 'estimator', 'params.foo.bar' -> 'foo.bar'
            parts = k.split(".", 1)
            context_key = parts[-1]  # drop leading 'tags' or 'params' if present
            context[context_key] = v

        yield where, context

def default_row_builder(
    rid,
    context,
    mi_bits_test,
    mi_bits_train,
    metric_key_test,
    metric_key_train,
    meta_dict=None,
    include_meta_keys=None,
):
    """
    Build a result row (dict) from a single run.

    include_meta_keys: list of metadata field names to save.
      - Keys may reference nested paths, e.g. 'params.base_critic_params.Nx'
      - If key not found, value recorded as None.
    """
    info_est, max_test_idx = summarize_mi_traces_test_train(mi_bits_test, mi_bits_train)

    row = {
        "run_id": rid,
        "info_est": info_est,
        "max_test_idx": max_test_idx,
        "metric_key_test": metric_key_test,
        "metric_key_train": metric_key_train,
    }

    # Include sweep + filter context
    row.update(context)

    # Record requested metadata keys
    if include_meta_keys and meta_dict:
        for key in include_meta_keys:
            value = _deep_get(meta_dict, key)
            short_key = key.split(".", 1)[-1]  # strip leading params/tags/etc.
            row[short_key] = value

    return row

def load_mi_summary_raw(
    outfile,
    sweep_tags,
    extra_filters=None,
    metric_key_test="mi_bits_test",
    metric_key_train="mi_bits_train",
    row_builder=default_row_builder,
    include_meta_keys=None,
):
    """
    Same as before, but now supports extracting extra metadata fields
    into the output DataFrame.
    """
    if extra_filters is None:
        extra_filters = {}
    if include_meta_keys is None:
        include_meta_keys = []


    results = []

    with H5ResultStore(outfile, "r") as rs:
        for where, context in iter_sweep_queries(sweep_tags, extra_filters):
            rids = rs.query(where=where)

            for rid in rids:
                mi_bits_test = rs.load_array(rid, metric_key_test)
                mi_bits_train = rs.load_array(rid, metric_key_train)
                meta = rs.get_meta(rid)

                row = row_builder(
                    rid=rid,
                    context=context,
                    mi_bits_test=mi_bits_test,
                    mi_bits_train=mi_bits_train,
                    metric_key_test=metric_key_test,
                    metric_key_train=metric_key_train,
                    meta_dict=meta,
                    include_meta_keys=include_meta_keys,
                )
                results.append(row)

    return pd.DataFrame(results)


def load_mi_summary(
    outfile,
    sweep_tags,
    extra_filters=None,
    metric_key_test="mi_bits_test",
    metric_key_train="mi_bits_train",
    row_builder=default_row_builder,
    include_meta_keys=None,
    use_temp_snapshot: bool =True,
):
    """
    Public API: by default, reads from a temporary snapshot of `outfile`
    to avoid clashing with a writer on the same HDF5 file.
    """
    if not use_temp_snapshot:
        # Direct read (unsafe if writer is active, but available if needed)
        return load_mi_summary_raw(
            outfile,
            sweep_tags=sweep_tags,
            extra_filters=extra_filters,
            metric_key_test=metric_key_test,
            metric_key_train=metric_key_train,
            row_builder=row_builder,
            include_meta_keys=include_meta_keys,
        )

    # Safe: make a snapshot and read that
    with temporary_h5_snapshot(outfile, suffix=".h5") as tmp:
        return load_mi_summary_raw(
            tmp,
            sweep_tags=sweep_tags,
            extra_filters=extra_filters,
            metric_key_test=metric_key_test,
            metric_key_train=metric_key_train,
            row_builder= row_builder,
            include_meta_keys=include_meta_keys,
        )