#!/bin/bash
#SBATCH -J job_latent_slurm_test  # Job name
#SBATCH --output=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/Test_July_5/output_slurm_test.txt   # Output file name
#SBATCH --error=/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/Test_July_5/error_slurm_test.txt    # Error file name
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
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 latent_test_July_5.py
