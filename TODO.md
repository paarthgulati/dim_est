# TO-DO:

## Short-term:

1. (PG) Fix device handling. Currently works with cuda. Add proper device handling based on available platforms. Fix environment/requirements file accordingly.

2. (PG) Build Dataset classes and functions to import data instead of self-generated Datasets, with corresponding dataset dictionaries. Allow for image, neural and other arbitrary inputs

3. (EA - Minor) Fix CCA to account for bits or nats in the same way as everything else.

4. (EA - Minor) Make the tqdm bar show the MI values.

5. (EA - Minor) Include smoothing in utils.

6. (EA - Major) A more flexible train and test splits.

7. (PG - Minor) Fix the observed dimensionality.

8. (EA - Minor) Fix device in run finite

9. Add DVSIB

10. (PG) Fix the defaults from the transforms and datasets.

_______________________________________
## For the next run:

### A. Documentation (Crucial)
README.md: The landing page. It needs:
- The Pitch: "Neural Estimation of Intrinsic Dimensionality."

- Installation: Instructions for conda or pip (requirements.txt).

- Quick Start: A 5-line code snippet showing a basic sweep.

- Citation: How to cite your work.

docs/ Folder (Partially Done):

- ✅ theory_background.md (Drafted).

- ✅ library_design.md (Drafted).

- [Missing] API Reference: A simple markdown file listing key arguments for run_dsib_infinite and the ExperimentConfig structure. Users shouldn't have to read source code to understand what split_strategy does.

### B. Tutorials (The "How-To")
Current Status: 01_Basic_Estimation_and_Critics.ipynb is ready.

Missing (Critical for Utility):

- 02_Real_Data_and_Splitting.ipynb: Essential for users who want to use their own data files (.pt/.npy).

- 03_Advanced_Architectures.ipynb: Essential for users with image data (CNNs) or needing Siamese networks.

- 04_Other_Dim_Estimators.ipynb: Compare performance to other estimators.

### C. Code Polish & Hygiene
Dependency Management: Ensure requirements.txt or environment.yml includes seaborn, tqdm, joblib, and scikit-learn (used in your tutorial).

License: Add an MIT or Apache 2.0 license file.

.gitignore: Ensure __pycache__, .ipynb_checkpoints, h5_results/, and test_results/ are ignored to avoid bloating the repo.

### D. Validation
Test Suite: Ensure dim_est/tests/run_all.py passes cleanly on your final codebase. This gives users confidence the code is stable.
________________________________________
## Long-term:
