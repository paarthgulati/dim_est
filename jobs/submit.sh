#!/bin/bash
##############################################################################################
#SBATCH --job-name=low_dim_test   # Name of the job
#SBATCH --nodes=1                        # Number of nodes requested
#SBATCH --partition=l4-1-gm24-c16-m64,l4-4-gm96-c48-m192,l4-8-gm192-c192-m768,l40s-8-gm384-c192-m1536,a100-8-gm320-c96-m1152    # Name of the partition
#SBATCH --output=logs/%x_%j.out               # Name of the output file
#SBATCH --error=logs/%x_%j.err                # Name of the error file
#SBATCH --gpus=1                         # Number of GPUs needed
#SBATCH --mem=20G                         # Memory needed (adjust as needed)

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate MI_estimation 

python -u "$@"
