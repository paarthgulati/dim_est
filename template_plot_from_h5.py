import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict
import glob
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


from dim_est.analysis.h5_to_dataframe_plotting_helpers import load_mi_summary

def plotting_template_finite_data(outdir=None, stem="finite_data_joint_gaussian"):

    ### 0. Resolve the output directory and find saved HDF5 files     ########################################################################
    if outdir is None:
        outdir = os.path.join(SCRIPT_DIR, "h5_results")
    else:
        outdir = os.path.abspath(outdir)
        
    files = glob.glob(f"{outdir}/{stem}_*.h5")
    if not files:
        print(f"No HDF5 files found matching pattern: {stem}_*.h5 in {search_dir}")
    else:
        print("Found HDF5 files:")
        for f in files:
            print("  •", os.path.basename(f))

        
    dataset_type = 'joint_gaussian'

    mi_bits = 2.0
    latent_dim = 4
    n_epoch = 100
    fig, ax = plt.subplots()

    critic_type = "hybrid"
        
    ### 1. Create a dataframe from the h5 files relevant to the plot     ########################################################################
    
    # different parameter values to sweep over for different curves on the same plot
    sweep_tags = {
        "params.experiment_cfg.training.cfg.n_samples": [128],
        "tags.dataset_type": [dataset_type],
        "tags.critic_type": [critic_type],
    }

    # filter by any other parameters to be kept fixed
    extra_filters = {
        "params.experiment_cfg.dataset.cfg.latent.latent_dim": latent_dim,
        "params.experiment_cfg.dataset.cfg.latent.mi_bits": mi_bits,
        "params.experiment_cfg.training.cfg.n_epoch": n_epoch,
    }
    
    # load a combined dataframe that has all the datapoints as individual rows, with columns based on the setup and using the saved values of metric_keys.
    # WARNING: It expects the keys to be in order (test, train)
    # Any additional column entries not sweeped or filtered, can be saved as additional columns include_meta_keys 
    results_df = load_mi_summary(
        outfile=files,
        sweep_tags=sweep_tags,
        extra_filters=extra_filters,
        setup = "finite_data_epoch",
        metric_keys=("mi_bits_test", "mi_bits_train"),
        include_meta_keys=[
            "tags.kz",       
        ]
    )
    
    ## 2. Plot from the dataframe ########################################################################

    # axis title -- nominally includes information about the filters used or other things kept fixed
    title = rf"{dataset_type}, $K_Z =$ {latent_dim}; Max {n_epoch} epochs"

    # plot labels/legends -- converts the passed group keys into entries that can be used to create legends
    def label_fn(group_vals):
            critic_type, n_samples = group_vals ## group_vals is always a tuple (read with the comma)
            return rf"{critic_type}, Samples: {n_samples}"
    
    ## convert into a plot: return generally, groups entries by group_keys, and plots y_key vs x_key
    ## for a given group and x value, it plots light scatter plots for each trial
    ## makes an errorbar plot with the mean y_val over trials, with y_err given by the standard error over the trials 
    plot_from_df_agg_errorbar_mean_stderr(
        ax,
        results_df,
        group_keys=["critic_type", "experiment_cfg.training.cfg.n_samples"],
        x_key="kz",
        y_key="info_est",
        label_fn=label_fn,
        title=title,
        true_dim=latent_dim
    )

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.show()


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


if __name__ == "__main__":
    print("Plotting the results of the template sweep saved in h5_results/....")
    print("Plotting the finite samples, joint gaussian case")
    plotting_template_finite_data()