#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=process_dataset_x
#SBATCH --output=./cluster/uzh/dataset/process/process_dataset_x_exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/process/process_dataset_x_exp_set_0_%a.err

# SLURM_ARRAY_TASK_ID=$1

CLUSTER=uzh
# DATA_PATH=/home/wehe/scratch/data/dataset-L0-exp-set-0-500sets.h5
# DATA_PATH=../data/dataset/dataset_L0_exp_set_0.h5
DATA_PATH=/home/wehe/scratch/data/dataset_L0_exp_set_0.h5
# module load anaconda3
# source activate sbi

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/process/process_dataset_x.log"

echo "print_log: ${PRINT_LOG}"

code ${PRINT_LOG}

# /data/wehe/conda/envs/sbi/bin/python3 -u ./src/dataset/process_dataset_x.py \
# --data_path ${DATA_PATH} &> ${PRINT_LOG}
# echo 'finished dataset x seqC process'

echo 'start uploading'
# cd "/home/wehe/data"
/home/wehe/data/gdrive files upload "${DATA_PATH}"

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952
# squeue -u $USER
# scancel --user=wehe
# squeue -u $USER
# squeue -u $USER

# SBATCH --gres=gpu:T4:1
# SBATCH --gres=gpu:V100:1
# SBATCH --gres=gpu:A100:1
# 0: 22G
# 1: 32G