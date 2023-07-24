#!/bin/bash
#SBATCH -p vossenlab-gpu
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/sidebands/July_24_switch_no_sidebands_double_.out
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/sidebands/July_24_switch_no_sidebands_double_.error
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
data_epochs=6
mc_epochs=11
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Sidebands_NF.py --Date "Jul_24" --num_epochs_mc $mc_epochs --num_epochs_data $data_epochs --extra_info "_switch_no_sidebands_double_TEST_" --switch_mask True --sidebands False --double True

