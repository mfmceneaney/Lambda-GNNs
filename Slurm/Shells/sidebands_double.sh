#!/bin/bash
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/sidebands/July_19_sidebands.out
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/sidebands/July_19_sidebands.error
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL
data_epochs=2
mc_epochs=2
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Sidebands_NF.py --Date "July_19" --num_epochs_mc $mc_epochs --num_epochs_data $data_epochs

