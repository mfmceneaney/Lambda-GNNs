#!/bin/bash
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/sidebands/July_26_sidebands_double_no_balanced_.out
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/sidebands/July_26_sidebands_double_no_balanced_.error
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
data_epochs=1
mc_epochs=1
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Sidebands_NF.py --Date "Jul_26" --num_epochs_mc $mc_epochs --num_epochs_data $data_epochs --extra_info "_sidebands_double_no_balanced_" --switch_mask --sidebands --double --no-balanced

