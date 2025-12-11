import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict


def plot_from_df_agg_errorbar_mean_stderr(
    ax,
    results_df,
    *,
    group_keys=("critic_type", "experiment_cfg.dataset.cfg.transform.sig_embed_x"),
    x_key="kz",
    y_key="info_est",
    # yerr_key="std_smoothed_info",
    label_fn=None,
    title: Optional[str] = None,
    true_dim: Optional[int] = None,
    x_label = r"Embedding dimension, $k_z$", 
    y_label = r"Estimated information, $I_{\rm est}$",
):
    """
    Plot test_train MI (info_est) vs k_z from an already-built DataFrame.
    """

    if results_df is None or results_df.empty:
        ax.text(0.5, 0.5, "No matching runs", ha="center", va="center")
        ax.set_xlabel(r"$k_z$")
        ax.set_ylabel(r"$I_{\rm est}$")
        if title:
            ax.set_title(title)
        return

    if isinstance(group_keys, str):
        group_keys = (group_keys,)          
    else:
        group_keys = tuple(group_keys)

    # Default label_fn: turn group values into a readable string
    if label_fn is None:
        def label_fn(group_vals: tuple):
            parts = []
            for k, v in zip(group_keys, group_vals):
                parts.append(f"{k}={v}")
            return ", ".join(parts)


    ## plot mean and standard deviation error bar over the trials
    for group_vals, df_sub in results_df.groupby(list(group_keys)):
        # 1. Light scatter of all individual trials in this group
        axscatter = ax.scatter(
            df_sub[x_key],
            df_sub[y_key],
            alpha=0.05,
        )
    
        # 2. Aggregate over repeated x_key (e.g. trials at the same kz)
        agg = (
            df_sub
            .groupby(x_key)[y_key]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    
        label = label_fn(group_vals)
    
        # 3. Choose color consistent with the faint scatter
        facecolors = axscatter.get_facecolors()
        color = facecolors[0] if len(facecolors) > 0 else None
    
        # 4. Errorbar plot: mean ± std
        ax.errorbar(
            agg[x_key],
            agg["mean"],
            yerr=agg["std"]/np.sqrt(agg["count"]),
            label=label,
            linewidth=1.5,
            marker="o",
            linestyle="-",
            capsize=3,
            color=color,
            alpha=1.0,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        ax.set_title(title)
    if true_dim is not None:
        ax.axvline(true_dim, ls="--", c="k", alpha=0.2)
    ax.legend()

    
def plot_mi_vs_kz_from_df(
    ax,
    results_df,
    *,
    group_keys=("critic_type", "experiment_cfg.dataset.cfg.transform.sig_embed_x"),
    x_key="kz",
    y_key="max_smoothed_info",
    yerr_key="std_smoothed_info",
    label_fn=None,
    title: Optional[str] = None,
    true_dim: Optional[int] = None,
):
    """
    Plot max-smoothed MI vs k_z from an already-built DataFrame.

    Assumes `results_df` already has columns:
      - x_key           (default 'kz')
      - y_key           (default 'max_smoothed_info')
      - yerr_key        (default 'std_smoothed_info')
      - group_keys      (e.g. 'critic_type', noise level, etc.)
    """
    if results_df is None or results_df.empty:
        ax.text(0.5, 0.5, "No matching runs", ha="center", va="center")
        ax.set_xlabel(r"$k_z$")
        ax.set_ylabel(r"$I_{\rm est}$")
        if title:
            ax.set_title(title)
        return

    if isinstance(group_keys, str):
        group_keys = (group_keys,)          
    else:
        group_keys = tuple(group_keys)

    # Default label_fn: turn group values into a readable string
    if label_fn is None:
        def label_fn(group_vals: tuple):
            parts = []
            for k, v in zip(group_keys, group_vals):
                parts.append(f"{k}={v}")
            return ", ".join(parts)

    for group_vals, df_sub in results_df.groupby(list(group_keys)):
        # scatter + error bars for all runs in this group
        axerr, _, _ = ax.errorbar(
            df_sub[x_key],
            df_sub[y_key],
            yerr=df_sub[yerr_key],
            fmt="o",
            alpha=0.2,
        )

        # aggregate over repeated kz (e.g. max over trials)
        agg = (
            df_sub.groupby(x_key)[y_key]
            .agg(["max"])
            .reset_index()
        )

        label = label_fn(group_vals)

        ax.plot(
            agg[x_key],
            agg["max"],
            label=label,
            linewidth=1.5,
            c=axerr.get_color(),
        )
        ax.scatter(
            agg[x_key],
            agg["max"],
            c=axerr.get_color(),
            alpha=1.0,
        )

    ax.set_xlabel(r"$k_z$")
    ax.set_ylabel(r"$I_{\rm est}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        ax.set_title(title)
    if true_dim is not None:
        ax.axvline(true_dim, ls="--", c="k", alpha=0.2)
    ax.legend()