#!/bin/bash
#SBATCH --job-name=diff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1 # GPU per nodes
##SBATCH --gres=gpu:1

# env
module load conda
conda activate oa_reactdiff

# run
python train_ts1x.py
