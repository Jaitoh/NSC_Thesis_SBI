#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=4

#SBATCH --job-name=process_dataset_x
#SBATCH --output=./cluster/uzh/dataset/other_logs/process_dataset_x_exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/other_logs/process_dataset_x_exp_set_0_%a.err

# SLURM_ARRAY_TASK_ID=$1

CLUSTER=snn

if [ "${CLUSTER}" == "uzh" ]; then
    # DATA_PATH=/home/wehe/scratch/data/dataset/dataset_L0_exp_set_0.h5
    DATA_PATH=../data/dataset/dataset_L0_exp_set_0.h5
    module load anaconda3
    source activate sbi
else
    DATA_PATH=../data/dataset/dataset_L0_exp_set_0.h5
fi

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/process_dataset_x_exp_set_0.log"

echo "print_log: ${PRINT_LOG}"

python3 -u ./src/dataset/process_dataset_x.py \
--data_path ${DATA_PATH} &> ${PRINT_LOG}

echo 'finished dataset x seqC process'

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