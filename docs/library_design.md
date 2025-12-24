# Library Design & Philosophy

`dim_est` is designed to operationalize the "Sweep" as the fundamental unit of experimentation. A single run with a fixed embedding dimension is rarely informative; insight comes from the curve of MI vs. Embedding Dimension ($k_z$).

## Core Components

The library pipeline follows the flow of information:

### 1. Data Ingestion & Splitting
* **Infinite Limit:** For theoretical validation, we generate synthetic data on-the-fly (unlimited -or practically, very large- number batches. This is common in validating neural estimators in general).
* **Finite Data:** For real-world applications, we load fixed tensors.
* **Single-Source Splitting:** The library natively handles converting a single dataset $X$ into pairs $(X, Y)$ via:
    * `TemporalDataset` (Time-lagged pairs)
    * `SpatialSplitDataset` (Image patches)
    * `AugmentationDataset` (SimCLR-style views)

### 2. The Encoder ($f, g$)
* Maps high-dimensional inputs to the bottleneck $Z \in \mathbb{R}^{k_z}$.
* **Modular Architecture:**
    * **MLP:** For vector data.
    * **ResNet/CNN:** For image data.
    * **RNN/Transformer:** For sequential data.
    * **Siamese:** Option to force $f = g$ (`share_encoder=True`) for single-source datasets.

### 3. The Critic ($h$)
* Scores the relationship between $Z_X$ and $Z_Y$.
* **Hybrid Critic:** The default recommendation. It concatenates embeddings and passes them through a powerful MLP to estimate density ratios accurately, preventing "false saturation."

### 4. The Estimator
* Converts critic scores into a scalar Lower Bound of Mutual Information (in nats or bits).
* Includes `InfoNCE`, `L-Clip`, and `SMILE`.

## The "Sweep" Workflow

The library is architected to avoid the "single run" trap.

1.  **Define the Sweep:** You choose a range of `embed_dim` (e.g., `[1, 2, ..., 12]`).
2.  **Parallel Execution:** The `run_sweep_parallel` module dispatches these trials to available CPUs/GPUs.
3.  **Aggregation:** Results are merged into a single HDF5 file.
4.  **Analysis:** We look for the "knee" or plateau in the MI curve.

## Configuration System
We use strictly typed `dataclasses` (`ExperimentConfig`) backed by dictionary-based defaults.
* **Reproducibility:** Every run saves its full configuration and the git commit hash.
* **Flexibility:** Users override specific parameters (like `critic_type` or `split_strategy`) while inheriting sensible defaults.