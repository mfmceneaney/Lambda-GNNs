#!/bin/bash
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/Optimization/July_18_study_0.out
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/Optimization/July_18_study_0.error
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Hyperparameter_optimization_double.py --Date "July_18"

