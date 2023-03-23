#!/bin/bash 
srun --pty -n 1 -c 2 --time=00:30:00 --mem=4G bash -l

module load anaconda3
source activate sbi

# squeue -u $USER
# scancel 2905690