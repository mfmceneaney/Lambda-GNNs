#!/bin/bash

current_date=$(date +"%b_%d")

# Create a folder based on the current date (YYYY-MM-DD format)
current_date=$(date +"%b_%d")
working_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/"
output_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/Latent_${current_date}"
error_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/Latent_${current_date}"
shell_folder="${working_folder}Shells/latent_${current_date}"
runJobs="${working_folder}/runLatent.sh"
touch $runJobs
echo " " > $runJobs

if [ ! -d "$output_folder" ]; then
  mkdir "$output_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir "$error_folder"
fi

if [ ! -d "$shell_folder" ]; then
  mkdir "$shell_folder"
fi
mc_epochs=1
data_epochs=1

# Define arrays for each argument's options
declare -a arg1_options=("--sidebands" "--no-sidebands")
declare -a arg2_options=("--double" "--no-double")
declare -a arg3_options=("--balanced" "--no-balanced")
s_options=("s" "ns")
d_options=("d" "nd")
b_options=("b" "nb")

# Loop through all combinations and generate slurm scripts
i=0
s=0
for arg1 in "${arg1_options[@]}"; do
  d=0
  for arg2 in "${arg2_options[@]}"; do
    b=0
    for arg3 in "${arg3_options[@]}"; do
      # Generate a unique name for the slurm script based on the arguments
      script_name="$shell_folder/slurm_latent_${s_options[$s]}_${d_options[$d]}_${b_options[$b]}.sh"
      if (( i <= 3 )); then
        partition="vossenlab-gpu"
      else
        partition="scavenger-gpu"
      fi
      # Create/open the slurm script
      cat > "$script_name" <<EOL
#!/bin/bash
#SBATCH -J job_${arg1}_${arg2}_${arg3}  # Job name
#SBATCH --output=${output_folder}/output_${s_options[$s]}_${d_options[$d]}_${b_options[$b]}.txt   # Output file name
#SBATCH --error=${error_folder}/error_${s_options[$s]}_${d_options[$d]}_${b_options[$b]}.txt    # Error file name
#SBATCH -p $partition     # Partition (queue) name
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
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Sidebands_NF.py --Date $current_date --num_epochs_mc $mc_epochs --num_epochs_data $data_epochs --extra_info "_${s_options[$s]}_${d_options[$d]}_${b_options[$b]}_" --switch_mask "$arg1" "$arg2" "$arg3"
EOL

      # Make the script executable
      chmod +x "$script_name"
      echo "sbatch $script_name" >> $runJobs
      b=$((b+1))
      i=$((i+1))
    done
    d=$((d+1))
  done
  s=$((s+1))
done
