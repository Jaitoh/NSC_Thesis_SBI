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

if [ "${CLUSTER}" == "uzh" ]; then
    DATA_DIR=/home/wehe/scratch/data/dataset/
    MERGED_DATA_path=/home/wehe/scratch/data/dataset-L0-exp-set-0-500sets.h5
    # module load anaconda3
    # source activate sbi
else
    DATA_DIR=../data/dataset/dataset-L0-exp-set-0-500sets.h5
fi

PRINT_DIR="./cluster/${CLUSTER}/dataset/process"
PRINT_LOG="./cluster/${CLUSTER}/dataset/process/merge_dataset_exp_set_0.log"

echo "print_log: ${PRINT_LOG}"

# mkdir -p $DATA_DIR
# mkdir -p $PRINT_DIR

/data/wehe/conda/envs/sbi/bin/python3 -u ./src/dataset/merge_dataset.py \
--data_dir ${DATA_DIR} \
--merged_data_path ${MERGED_DATA_path} &> ${PRINT_LOG}

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