#!/bin/bash
#SBATCH -J job_--no-sidebands_--no-double_--no-balanced  # Job name
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/Latent_Jul_27/output_--no-sidebands_--no-double_--no-balanced_fewer_epochs.txt   # Output file name
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/Latent_Jul_27/error_--no-sidebands_--no-double_--no-balanced_fewer_epochs.txt    # Error file name
#SBATCH -p vossenlab-gpu     # Partition (queue) name
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL

cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv

# Run the Python script with the specific command line arguments
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Sidebands_NF.py --Date Jul_27 --num_epochs_mc 11 --num_epochs_data 5 --extra_info "_ns_nd_nb_fewer_epochs_" --switch_mask "--no-sidebands" "--no-double" "--no-balanced"
