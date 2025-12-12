# dim_est

## Installation and basic usage

### Conda environment
Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dim_est
```
Note: the current environment file only allows for a cuda implementation; will add cpu requirement in the future.

### Install the package (editable mode)
From the repository root (the directory containing `pyproject.toml`):
```bash
pip install -e .
```
Editable installation (`-e`) ensures that changes made to the source code are picked up immediately when running scripts.

Verify the install:
```
python -c "import dim_est; print(dim_est.__file__)"
```
This should print a path pointing to the local `dim_est/` directory.

## Running an experiment
The basic execution unit is a **single experiment**, implemented in:
  `dim_est/run/run_dsib_single_experiment.py`

In practice, experiments are typically launched via a sweep template or a call to the correct single experiment function.

A template for running parameter sweeps is provided in:
    `templates/template_run_sweep.py`

You can modify this file directly or run it as-is, for example:
```
    python templates/template_run_sweep.py --job finite_joint_gaussian
```
The template:
- runs a short sanity check for both finite- and infinite-data regimes
- demonstrates how to construct experiments using default dataset, critic, and training configurations from `dim_est/config/`
- allows parameters to be overridden via configuration dictionaries defined in the template

Single experiment logic:
A single experiment (`run_dsib_single_experiment.py`) performs the following steps:

0. Initializes initial state using the passed or randomly generated seed.
1. Constructs datasets and models using default and user-specified configuration parameters
2. Runs training using the specified setup:
   - infinite_data_iter
   - finite_data_epoch
3. Saves results to an HDF5 file, including:
   - full experiment configuration
   - code metadata
   - training traces of the relevant mutual information estimates

## Plotting and analysis

Results are stored in HDF5 files and can be loaded using utilities in:
`dim_est/analysis/` and `dim_est/utils/h5_result_store.py`

A plotting template is provided in: `templates/template_plot_from_h5.py` and can be run directly to plot the results of the sweep template as:
```
python templates/template_plot_from_h5.py
```

This template demonstrates how to:
- load results from HDF5 files
- construct dataframes for analysis
- generate plots from the dataframe 
The default template includes a pipeline to construct a plot of the estimated mutual information vs the encoding dimension, for the finite_data_epoch setup using the default joint Gaussian dataset constructed by the sweep template.

The template can be easily adapted to other datasets, setups, and experiment configurations.

Notes
- Python >= 3.9 is required
- Output directories (e.g. for HDF5 results) are created automatically if missing
- This repository is intended for research use and active development

## Directory structure
    dim_est/
      Core Python package:
        datasets/   – synthetic latent data generation and observation transforms
        models/     – critics, estimators, and neural models (DSIB)
        training.py – training loops for infinite and finite data regimes
        run/        – single-experiment execution logic
        analysis/   – helpers for loading HDF5 results and plotting
        config/     – default experiment, dataset, critic, and training configs
        utils/      – HDF5 storage, CCA utilities, networks, logging
        tests/      – lightweight sanity checks

    templates/
      Example Python templates:
        template_run_sweep.py     – parameter sweeps
        template_plot_from_h5.py  – plotting and analysis

    environment.yml
    requirements.txt
    pyproject.toml
