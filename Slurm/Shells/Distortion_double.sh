#!/bin/bash
distort=0.02
epochs=8
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/July_18_distortion_0_02_8_epochs.out
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/July_18_distortion_0_02_8_epochs.error
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Distortion_NF.py --Date "July_18" --num_epochs epochs --Distortion distort

