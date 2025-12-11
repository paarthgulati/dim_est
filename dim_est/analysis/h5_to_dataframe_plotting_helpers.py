import numpy as np
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter1d, median_filter
import os, sys
import pandas as pd
from typing import Optional, Dict, Sequence, Union
import itertools
from ..utils.h5_result_store import H5ResultStore
import tempfile
import shutil
import os
from contextlib import contextmanager

## this notebook builds the functions necessary to recover the summarized scalar(s) for different runs based on the stored metrics (trace over training of various MI estimates)

## CURRENTLY WORKS WITH TWO TRAINING MODES, SAME AS THE EXPERIMENTS:
# "setup: ["infinite_data_iter", "finite_data_epoch"]"


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
):
    """
    """

    # Ensure integer kernel size (robust to float inputs)
    ema_span = max(1, int(round(ema_span)))
    assert mi_bits_test.shape == mi_bits_train.shape
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

## get plotting metrics from MI trace over iterations
def summarize_mi_trace_infinite_resampling(
    mi_bits: np.ndarray,
    ema_span: int = 20,
    smooth_sigma: float = 1.0,
    tail_fraction: float = 0.5,
):
    """
    Summarize an MI trace by smoothing and extracting:
      - max over smoothed entire trace
      - std over smoothed tail portion

    Parameters
    ----------
    mi_bits : np.ndarray
        MI values over training iterations.
    ema_span : int, default=20
        Window size for median filtering before Gaussian smoothing.
    smooth_sigma : float, default=1.0
        Standard deviation for Gaussian smoothing.
    tail_fraction : float, default=0.5
        Fraction of the trace tail used for stability estimation.

    Returns
    -------
    tuple (max_smoothed_info, std_smoothed_tail):
        max of smoothed trace and std of smoothed tail.
    """

    # Ensure integer kernel size (robust to float inputs)
    ema_span = max(1, int(round(ema_span)))
    # Replace NaNs just in case
    mi = np.nan_to_num(mi_bits)

    # ---- Smooth full trace ----
    smoothed_full = gaussian_filter1d(
        median_filter(mi, size=ema_span),
        sigma=smooth_sigma,
    )
    max_smoothed_info = np.max(smoothed_full)

    # ---- Tail smoothing ----
    n = len(mi)
    tail_start = int((1 - tail_fraction) * n)
    tail = mi[tail_start:]

    smoothed_tail = gaussian_filter1d(
        median_filter(tail, size=ema_span),
        sigma=smooth_sigma,
    )
    std_smoothed_tail = np.std(smoothed_tail)

    return max_smoothed_info, std_smoothed_tail



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
    setup: str,
    metrics: Dict[str, np.ndarray],
    metric_keys: Sequence[str],
    meta_dict=None,
    include_meta_keys=None,
    **kwargs
):
    """
    Build a result row (dict) from a single run.

    metrics: dict of metric_key -> trace array
    metric_keys: ordered metric names relevant for this setup
    """
    include_meta_keys = include_meta_keys or []

    if setup == "infinite_data_iter":
        # Expect a single metric, e.g. "mi_bits"
        assert len(metric_keys) == 1, "infinite_data_iter expects exactly one metric key"
        key = metric_keys[0]
        max_info, std_info = summarize_mi_trace_infinite_resampling(metrics[key])
        row = {
            "run_id": rid,
            "max_smoothed_info": max_info,
            "std_smoothed_info": std_info,
            "metric_key": key,
        }

    elif setup == "finite_data_epoch":
        # Expect two metrics: test, train
        assert len(metric_keys) == 2, "finite_data_epoch expects two metric keys (test, train)"
        assert "test" in metric_keys[0].lower(), "First metric must be test trace"
        assert "train" in metric_keys[1].lower(), "Second metric must be train trace"

        key_test, key_train = metric_keys
        info_est, max_test_idx = summarize_mi_traces_test_train(
            metrics[key_test],
            metrics[key_train],
        )
        row = {
            "run_id": rid,
            "info_est": info_est,
            "max_test_idx": max_test_idx,
            "metric_key_test": key_test,
            "metric_key_train": key_train,
        }

    else:
        raise ValueError(
            "Invalid setup! Pick one out of 'infinite_data_iter' or 'finite_data_epoch'. "
            f"Instead got {setup!r}"
        )

    # Include sweep + filter context
    row.update(context)

    # Record requested metadata keys
    if meta_dict:
        for key in include_meta_keys:
            value = _deep_get(meta_dict, key)
            short_key = key.split(".", 1)[-1]
            row[short_key] = value

    return row

def load_mi_summary_raw(
    outfile,
    sweep_tags,
    extra_filters=None,
    setup: str = "finite_data_epoch",
    metric_keys: Optional[Sequence[str]] = None,
    row_builder=default_row_builder,
    include_meta_keys=None,
):
    """
    Load MI traces and summarize into a DataFrame.

    setup : {"infinite_data_iter", "finite_data_epoch"}
    metric_keys :
        - if None:
            * infinite_data_iter -> ["mi_bits"]
            * finite_data_epoch -> ["mi_bits_test", "mi_bits_train"]
        - else: user-defined list/tuple of metric names in H5
    """
    if extra_filters is None:
        extra_filters = {}
    if include_meta_keys is None:
        include_meta_keys = []

    # Choose default metric_keys based on setup
    if metric_keys is None:
        if setup == "infinite_data_iter":
            metric_keys = ("mi_bits",)
        elif setup == "finite_data_epoch":
            metric_keys = ("mi_bits_test", "mi_bits_train")
        else:
            raise ValueError(
                "Invalid setup! Pick one out of 'infinite_data_iter' or 'finite_data_epoch'. "
                f"Instead got {setup!r}"
            )
    if isinstance(metric_keys, str):
        metric_keys = (metric_keys,)

    results = []

    with H5ResultStore(outfile, "r") as rs:
        for where, context in iter_sweep_queries(sweep_tags, extra_filters):
            rids = rs.query(where=where)

            for rid in rids:
                # Load all requested metrics for this run
                metrics = {k: rs.load_array(rid, k) for k in metric_keys}
                meta = rs.get_meta(rid)

                row = row_builder(
                    rid=rid,
                    context=context,
                    setup=setup,
                    metrics=metrics,
                    metric_keys=metric_keys,
                    meta_dict=meta,
                    include_meta_keys=include_meta_keys,
                )
                results.append(row)

    return pd.DataFrame(results)



def load_mi_summary(
    outfile: Union[str, Sequence[str]],
    sweep_tags,
    extra_filters=None,
    setup: str = "finite_data_epoch",
    metric_keys: Optional[Sequence[str]] = None,
    row_builder=default_row_builder,
    include_meta_keys=None,
    use_temp_snapshot: bool =True,
):

    ## Added logic to concat the return pd dataframes if passed a list or tuple of file names
    # If outfile is a list/tuple of files, recurse and concat
    if isinstance(outfile, (list, tuple)):
        dfs = [
            load_mi_summary(
                out,
                sweep_tags=sweep_tags,
                extra_filters=extra_filters,
                setup=setup,
                metric_keys=metric_keys,
                row_builder=row_builder,
                include_meta_keys=include_meta_keys,
                use_temp_snapshot=use_temp_snapshot,
            )
            for out in outfile
        ]
        if not dfs:
            return pd.DataFrame()
        # Optional: add a column telling us which file each row came from
        for out, df in zip(outfile, dfs):
            df["source_file"] = os.path.basename(out)
        return pd.concat(dfs, ignore_index=True)


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
            setup = setup,
            metric_keys= metric_keys,
            row_builder=row_builder,
            include_meta_keys=include_meta_keys,
        )

    # Safe: make a snapshot and read that
    with temporary_h5_snapshot(outfile, suffix=".h5") as tmp:
        return load_mi_summary_raw(
            tmp,
            sweep_tags=sweep_tags,
            extra_filters=extra_filters,
            setup = setup,
            metric_keys=metric_keys,
            row_builder= row_builder,
            include_meta_keys=include_meta_keys,
        )