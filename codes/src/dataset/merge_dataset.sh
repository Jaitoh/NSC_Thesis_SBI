#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=4

#SBATCH --job-name=merge_dataset
#SBATCH --output=./cluster/uzh/dataset/other_logs/merge_dataset_exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/other_logs/merge_dataset_exp_set_0_%a.err

# SLURM_ARRAY_TASK_ID=$1

CLUSTER=uzh
RUN_ID=exp_set_0

if [ "${CLUSTER}" == "uzh" ]; then
    DATA_PATH=/home/wehe/scratch/data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=/home/wehe/scratch/data/dataset/
else
    DATA_PATH=../data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=../data/dataset/
fi

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/merge_dataset_exp_set_0.log"

module load anaconda3
source activate sbi

echo "print_log: ${PRINT_LOG}"
echo "SEED: ${SEED}"

# mkdir -p $DATA_DIR
# mkdir -p $PRINT_DIR

python3 -u ./src/dataset/merge_dataset.py \
--data_dir ${DATA_DIR} &> ${PRINT_LOG}

echo 'finished dataset merge'

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