#!/bin/bash

# Script to check job status of all jobs in log file.
# Accepts file name as argument.
  
# Check arguments
FILENAME="jobs.txt"
if (($# > 0 )); then
    FILENAME=$1
    echo Checking $FILENAME
else
    name=`basename "$0"`
    echo "Usage: $name <filename>"
    exit 0
fi

slurm_pids=`grep -Eo '[0-9]{8}' $FILENAME`
slurm_pids=`echo $slurm_pids | sed "s;\n; ;g" | sed "s; ;,;g"`
echo "COMMAND: squeue -u $USER -j $slurm_pids"
squeue -u $USER -j $slurm_pids
