#!/bin/bash

# Define the range of values from 0.1 to 1.2
start_value=0.1
end_value=2
increment=0.2
epochs=8

# Create a folder based on the current date (YYYY-MM-DD format)
current_date=$(date +"%Y_%m_%d")
working_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/"
output_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Output/Output${current_date}"
error_folder="/hpc/group/vossenlab/rck32/Lambda-GNNs/Slurm/Error/Error${current_date}"
shell_folder="${working_folder}Shells/shells_${current_date}"

runJobs="${working_folder}/runJobs.sh"

touch $runJobs
echo " " > $runJobs

# Create the output folder if it doesn't exist
if [ ! -d "$output_folder" ]; then
  mkdir "$output_folder"
fi

if [ ! -d "$error_folder" ]; then
  mkdir "$error_folder"
fi

if [ ! -d "$shell_folder" ]; then
  mkdir "$shell_folder"
fi

s_arg=(
    "--sidebands"
    "--no-sidebands"
)
d_arg=(
    "--sidebands"
    "--no-sidebands"
)
b_arg=(
    "--sidebands"
    "--no-sidebands"
)
for s in ${arguments[@]}; do
  for t in 

i=0
# Loop through the values and create the SLURM scripts
for value in $(seq $start_value $increment $end_value); do
  # Define the SLURM script filename based on the value
  value_string=$(printf "%.1f" $value | tr '.' '_')
  script_filename="latent_${value_string}.sh"
  if (( i <= 2 )); then
    partition="vossenlab-gpu"
  else
    partition="scavenger-gpu"
  fi
  # Create the SLURM script
  cat <<EOF > "${shell_folder}/$script_filename"
#!/bin/bash
#SBATCH -p $partition
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=${output_folder}/output_${value_string}.txt
#SBATCH --error=${error_folder}/error_${value_string}.txt
#SBATCH --mail-user=rck32@duke.edu
#SBATCH --mail-type=END
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
cd /hpc/group/vossenlab/rck32/Lambda-GNNs
source /hpc/group/vossenlab/rck32/miniconda3/bin/activate
source activate /hpc/group/vossenlab/rck32/miniconda3/envs/venv
/hpc/group/vossenlab/rck32/miniconda3/envs/venv/bin/python3 Distortion_NF.py --Date $current_date --num_epochs $epochs --Distortion $value
EOF

  # Make the SLURM script executable
  chmod +x "${shell_folder}/$script_filename"
  echo "sbatch ${shell_folder}/$script_filename" >> $runJobs
  i=$((i+1))
done